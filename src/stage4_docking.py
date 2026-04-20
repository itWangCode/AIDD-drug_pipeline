"""
Stage 4: UniProt → PDB → Protein Preparation → Cartesian-Product Docking
Multi-target × Multi-ligand docking with robust error handling.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DIR, RESULTS_DIR, FIGURES_DIR, PROCESSED_DIR,
    PDB_CONFIG, DOCKING_CONFIG, TOP_N_COMPOUNDS, TOP_N_DOCKING,
    PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv, Checkpoint, retry

logger = setup_logger("stage4_docking", "stage4.log")


# ─── UniProt → PDB IDs ────────────────────────────────────────────────────────
@retry(max_attempts=8, delay=3.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def uniprot_to_pdb_ids(uniprot_accession: str) -> list:
    """Fetch PDB IDs from RCSB for a given UniProt accession via REST API."""
    # RCSB Search API
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": uniprot_accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": PDB_CONFIG["method"],
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": PDB_CONFIG["resolution_cutoff"],
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "return_all_hits": True,
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }

    resp = requests.post(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        json=query, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    pdb_ids = [hit["identifier"] for hit in data.get("result_set", [])]
    logger.info(f"Found {len(pdb_ids)} PDB entries for UniProt {uniprot_accession}")
    return pdb_ids


@retry(max_attempts=6, delay=3.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def get_pdb_metadata(pdb_id: str) -> dict:
    """Fetch PDB entry metadata."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return {
        "pdb_id": pdb_id.upper(),
        "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [999])[0],
        "method": data.get("exptl", [{}])[0].get("method", ""),
        "title": data.get("struct", {}).get("title", ""),
        "n_chains": data.get("rcsb_entry_info", {}).get("deposited_polymer_entity_instance_count", 0),
        "deposit_date": data.get("rcsb_accession_info", {}).get("deposit_date", ""),
    }


@retry(max_attempts=6, delay=3.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def get_pdb_ligands(pdb_id: str) -> list:
    """Fetch ligand information for a PDB entry."""
    query = """
    {
      entry(entry_id: "%s") {
        nonpolymer_entities {
          pdbx_entity_nonpoly { comp_id name }
          nonpolymer_entity_instances {
            rcsb_nonpolymer_entity_instance_container_identifiers { auth_asym_id auth_seq_id }
          }
        }
      }
    }
    """ % pdb_id.upper()

    resp = requests.get(
        "https://data.rcsb.org/graphql",
        params={"query": query}, timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    entities = data.get("data", {}).get("entry", {}).get("nonpolymer_entities") or []

    ligands = []
    for ent in entities:
        poly = ent.get("pdbx_entity_nonpoly", {})
        comp_id = poly.get("comp_id", "")
        name = poly.get("name", "")
        for inst in ent.get("nonpolymer_entity_instances", []):
            ids = inst.get("rcsb_nonpolymer_entity_instance_container_identifiers", {})
            ligands.append({
                "pdb_id": pdb_id.upper(),
                "ligand_id": comp_id,
                "ligand_name": name,
                "chain_id": ids.get("auth_asym_id", "A"),
                "seq_id": ids.get("auth_seq_id", ""),
            })
    return ligands


@retry(max_attempts=5, delay=5.0, backoff=2.0, exceptions=(Exception,), logger=logger)
def download_pdb_file(pdb_id: str, out_dir: Path) -> Path:
    """Download PDB file from RCSB."""
    pdb_id = pdb_id.upper()
    out_path = out_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    logger.info(f"Downloaded {pdb_id}.pdb ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def select_top_pdb_entries(pdb_ids: list, n_top: int = 5) -> pd.DataFrame:
    """Select top N PDB entries by resolution with ligand info."""
    records = []
    for pdb_id in tqdm(pdb_ids[:50], desc="Fetching PDB metadata"):  # limit to 50
        try:
            meta = get_pdb_metadata(pdb_id)
            ligs = get_pdb_ligands(pdb_id)
            # Filter: only entries WITH ligands (MW > threshold)
            if ligs:
                meta["ligands"] = json.dumps(ligs)
                meta["n_ligands"] = len(ligs)
                records.append(meta)
            time.sleep(0.3)
        except Exception as e:
            logger.warning(f"{pdb_id}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("resolution").head(n_top)
    logger.info(f"Selected {len(df)} PDB entries")
    return df


# ─── Protein Preparation ──────────────────────────────────────────────────────
def prepare_protein_pdbqt(pdb_path: Path, out_path: Path, ph: float = 7.4) -> Path:
    """
    Convert PDB to PDBQT using OpenBabel (obabel).
    Falls back to raw conversion if obabel unavailable.
    """
    if out_path.exists():
        return out_path

    try:
        result = subprocess.run(
            ["obabel", str(pdb_path), "-O", str(out_path),
             "--partialcharge", "gasteiger", "-p", str(ph), "--addh"],
            capture_output=True, text=True, timeout=60
        )
        if out_path.exists():
            logger.info(f"Protein prepared: {out_path.name}")
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"obabel failed: {e}. Trying MGLTools/pythonsh approach.")

    # Fallback: use AutoDockTools prepare_receptor4.py if available
    try:
        result = subprocess.run(
            ["prepare_receptor4.py", "-r", str(pdb_path), "-o", str(out_path),
             "-A", "hydrogens", "-U", "nphs_lps_waters_nonstdres"],
            capture_output=True, text=True, timeout=120
        )
        if out_path.exists():
            return out_path
    except FileNotFoundError:
        pass

    # Last resort: write a basic PDBQT by renaming
    logger.warning(f"No converter available. Copying PDB as placeholder PDBQT.")
    import shutil
    shutil.copy(pdb_path, out_path)
    return out_path


def prepare_ligand_pdbqt(smiles: str, out_path: Path, mol_id: str, ph: float = 7.4) -> Path:
    """Convert SMILES to PDBQT using OpenBabel."""
    if out_path.exists():
        return out_path

    try:
        result = subprocess.run(
            ["obabel", f"-:{smiles}", "--gen3d", "--best", "-O", str(out_path),
             "--partialcharge", "gasteiger", "-p", str(ph), "--addh"],
            capture_output=True, text=True, timeout=60
        )
        if out_path.exists():
            logger.debug(f"Ligand prepared: {mol_id}")
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Ligand prep failed for {mol_id}: {e}")
    return None


# ─── Binding Site Definition ──────────────────────────────────────────────────
def define_binding_box_from_ligand(pdb_path: Path, ligand_resname: str, buffer: float = 5.0) -> dict:
    """Extract binding site coordinates from co-crystallized ligand."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if (line.startswith("HETATM") or line.startswith("ATOM")) and \
                    line[17:20].strip() == ligand_resname:
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except ValueError:
                    continue

    if not coords:
        logger.warning(f"Ligand '{ligand_resname}' not found in {pdb_path.name}")
        return {"center_x": 0, "center_y": 0, "center_z": 0,
                "size_x": 25, "size_y": 25, "size_z": 25}

    coords = np.array(coords)
    center = (coords.max(axis=0) + coords.min(axis=0)) / 2
    size = coords.max(axis=0) - coords.min(axis=0) + buffer

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }


# ─── Smina/Vina Docking ───────────────────────────────────────────────────────
def run_smina_docking(
    protein_pdbqt: Path,
    ligand_pdbqt: Path,
    out_sdf: Path,
    box: dict,
    exhaustiveness: int = 16,
    num_modes: int = 9,
) -> dict:
    """Run smina docking. Returns best pose score."""
    if out_sdf.exists():
        score = parse_smina_score(out_sdf)
        return {"docking_score": score, "out_file": str(out_sdf), "status": "cached"}

    cmd = [
        "smina",
        "--receptor", str(protein_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(out_sdf),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--num_modes", str(num_modes),
        "--exhaustiveness", str(exhaustiveness),
        "--energy_range", str(DOCKING_CONFIG["energy_range"]),
        "--scoring", "vinardo",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        score = parse_smina_output(result.stdout)
        return {"docking_score": score, "out_file": str(out_sdf), "status": "success",
                "stdout": result.stdout}
    except FileNotFoundError:
        # Try vina as fallback
        logger.warning("smina not found. Trying AutoDock Vina...")
        cmd[0] = "vina"
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            score = parse_smina_output(result.stdout)
            return {"docking_score": score, "out_file": str(out_sdf), "status": "vina"}
        except FileNotFoundError:
            logger.error("Neither smina nor vina found. Please install smina.")
            return {"docking_score": None, "out_file": None, "status": "no_software"}
    except subprocess.TimeoutExpired:
        logger.warning(f"Docking timeout for {ligand_pdbqt.stem}")
        return {"docking_score": None, "out_file": None, "status": "timeout"}


def parse_smina_output(stdout: str) -> float:
    """Parse best docking score from smina/vina stdout."""
    for line in stdout.split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except (ValueError, IndexError):
                pass
    return None


def parse_smina_score(sdf_file: Path) -> float:
    """Parse score from SDF file (property field)."""
    try:
        with open(sdf_file) as f:
            content = f.read()
        for line in content.split("\n"):
            if "minimizedAffinity" in line or "SCORE" in line.upper():
                for part in line.split():
                    try:
                        return float(part)
                    except ValueError:
                        continue
    except Exception:
        pass
    return None


# ─── Cartesian Product Docking ────────────────────────────────────────────────
def run_cartesian_docking(
    gene_pdb_map: dict,       # {gene: [(pdb_id, ligand_resname, box), ...]}
    gene_compounds: dict,     # {gene: DataFrame with [smiles, molecule_chembl_id, ...]}
    docking_dir: Path,
    top_n: int = TOP_N_COMPOUNDS,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Cartesian product docking: ALL top compounds × ALL PDB structures per gene.
    """
    docking_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for gene, pdb_entries in gene_pdb_map.items():
        df_compounds = gene_compounds.get(gene)
        if df_compounds is None or len(df_compounds) == 0:
            continue

        # Select top N compounds by composite score
        sort_col = "composite_score" if "composite_score" in df_compounds.columns else "pIC50"
        top_compounds = df_compounds.nlargest(min(top_n, len(df_compounds)), sort_col)
        logger.info(f"\n[{gene}] Docking {len(top_compounds)} compounds × {len(pdb_entries)} PDB structures")

        gene_dir = docking_dir / gene
        gene_dir.mkdir(exist_ok=True)

        for pdb_id, ligand_resname, box in pdb_entries:
            pdb_file = RAW_DIR / "pdb" / f"{pdb_id}.pdb"
            if not pdb_file.exists():
                logger.warning(f"PDB file not found: {pdb_file}. Skipping.")
                continue

            protein_pdbqt = gene_dir / f"{pdb_id}_protein.pdbqt"
            prepare_protein_pdbqt(pdb_file, protein_pdbqt)

            for _, compound in tqdm(
                top_compounds.iterrows(),
                total=len(top_compounds),
                desc=f"  {gene}/{pdb_id}"
            ):
                mol_id = str(compound.get("molecule_chembl_id", f"mol_{compound.name}"))
                smiles = str(compound["smiles"])

                ckpt_key = f"dock_{gene}_{pdb_id}_{mol_id}"
                ckpt = Checkpoint(ckpt_key)
                if ckpt.exists() and not force_rerun:
                    dock_result = ckpt.load()
                    all_results.append(dock_result)
                    continue

                ligand_pdbqt = gene_dir / f"{mol_id}_ligand.pdbqt"
                out_sdf = gene_dir / f"{pdb_id}_{mol_id}_docked.sdf"

                lig_path = prepare_ligand_pdbqt(smiles, ligand_pdbqt, mol_id)
                if lig_path is None:
                    logger.warning(f"Ligand prep failed: {mol_id}")
                    continue

                dock_result = run_smina_docking(
                    protein_pdbqt, lig_path, out_sdf, box,
                    exhaustiveness=DOCKING_CONFIG["exhaustiveness"],
                    num_modes=DOCKING_CONFIG["num_modes"],
                )
                dock_result.update({
                    "gene": gene,
                    "pdb_id": pdb_id,
                    "molecule_chembl_id": mol_id,
                    "smiles": smiles,
                    "pIC50": compound.get("pIC50"),
                    "qed": compound.get("qed"),
                    "composite_score": compound.get("composite_score"),
                })
                ckpt.save(dock_result)
                all_results.append(dock_result)

    df_results = pd.DataFrame(all_results)
    if len(df_results) > 0:
        safe_save_csv(df_results, RESULTS_DIR / "docking_results_all.csv")
    return df_results


# ─── Docking Plots ────────────────────────────────────────────────────────────
def set_nature_style():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": PLOT_CONFIG["font_size"],
        "axes.titlesize": PLOT_CONFIG["title_size"],
        "axes.labelsize": PLOT_CONFIG["label_size"],
        "xtick.labelsize": PLOT_CONFIG["tick_size"],
        "ytick.labelsize": PLOT_CONFIG["tick_size"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": PLOT_CONFIG["dpi"],
    })


def plot_docking_heatmap(df_results: pd.DataFrame, gene: str):
    """Heatmap of docking scores: compounds vs PDB structures."""
    import seaborn as sns
    set_nature_style()

    df_gene = df_results[
        (df_results["gene"] == gene) & df_results["docking_score"].notna()
    ].copy()

    if df_gene.empty:
        return

    # Pivot: rows=compounds, cols=PDB IDs
    pivot = df_gene.pivot_table(
        index="molecule_chembl_id",
        columns="pdb_id",
        values="docking_score",
        aggfunc="min"  # best (most negative) score
    )

    # Sort by mean score
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    n_rows = min(40, len(pivot))
    pivot = pivot.head(n_rows)

    fig_h = max(10, n_rows * 0.35)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), fig_h))

    sns.heatmap(
        pivot, ax=ax, cmap="RdYlGn_r", annot=len(pivot.columns) <= 6,
        fmt=".1f", linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Docking Score (kcal/mol)", "shrink": 0.8},
        annot_kws={"fontsize": 9},
    )
    ax.set_xlabel("PDB Structure", fontsize=14, fontweight="bold")
    ax.set_ylabel("Compound (ChEMBL ID)", fontsize=14, fontweight="bold")
    ax.set_title(f"Cartesian-Product Docking Scores — {gene}\n"
                 f"(n={n_rows} compounds × {len(pivot.columns)} structures)",
                 fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(fontsize=9)

    out = FIGURES_DIR / f"{gene}_docking_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved docking heatmap: {out}")


def plot_top_docking_compounds(df_results: pd.DataFrame, gene: str, top_n: int = 20):
    """Top compounds by best docking score — horizontal bar chart."""
    set_nature_style()
    df_gene = df_results[(df_results["gene"] == gene) & df_results["docking_score"].notna()]
    if df_gene.empty:
        return

    # Best docking score per compound (across all PDB structures)
    best = df_gene.groupby("molecule_chembl_id").agg(
        best_docking_score=("docking_score", "min"),
        pIC50=("pIC50", "first"),
        qed=("qed", "first"),
        composite_score=("composite_score", "first"),
    ).reset_index().nsmallest(top_n, "best_docking_score")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: docking scores
    ax = axes[0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(best)))
    bars = ax.barh(range(len(best)), best["best_docking_score"], color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(best)))
    ax.set_yticklabels([str(i)[:15] for i in best["molecule_chembl_id"]], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Best Docking Score (kcal/mol)", fontsize=14, fontweight="bold")
    ax.set_title(f"Top {top_n} Compounds by Docking Score\n{gene}", fontsize=14, fontweight="bold")
    for bar, val in zip(bars, best["best_docking_score"]):
        ax.text(val - 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="right", fontsize=9, color="white", fontweight="bold")

    # Right: multi-metric scatter
    ax2 = axes[1]
    sc = ax2.scatter(best["best_docking_score"], best["pIC50"], s=120,
                     c=best["qed"], cmap="RdYlGn", vmin=0, vmax=1,
                     edgecolors="white", linewidths=1.5, zorder=3)
    plt.colorbar(sc, ax=ax2, label="QED Score", shrink=0.85)
    for _, row in best.iterrows():
        ax2.annotate(str(row["molecule_chembl_id"])[:10],
                      (row["best_docking_score"], row["pIC50"]),
                      fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.set_xlabel("Best Docking Score (kcal/mol)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("pIC₅₀", fontsize=14, fontweight="bold")
    ax2.set_title(f"Docking Score vs pIC₅₀ (colored by QED)\n{gene}", fontsize=14, fontweight="bold")
    ax2.axhline(y=6.3, color="red", linestyle="--", alpha=0.7, label="pIC₅₀ = 6.3 (active)")
    ax2.legend(fontsize=11)

    plt.suptitle(f"Molecular Docking Results — {gene}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"{gene}_top_docking_compounds.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved top docking compounds: {out}")


# ─── Main Stage 4 Function ────────────────────────────────────────────────────
def run_docking_pipeline(
    scored_data: dict,
    uniprot_df: pd.DataFrame,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Stage 4: Retrieve PDB structures and run Cartesian-product docking.
    """
    pdb_dir = RAW_DIR / "pdb"
    pdb_dir.mkdir(exist_ok=True)
    docking_dir = RESULTS_DIR / "docking"
    docking_dir.mkdir(exist_ok=True)

    gene_pdb_map = {}
    gene_compounds = {}

    for gene, result in scored_data.items():
        df_scored = result["df_scored"] if isinstance(result, dict) else result
        uniprot_row = uniprot_df[uniprot_df["gene_symbol"] == gene]
        if uniprot_row.empty:
            logger.warning(f"[{gene}] No UniProt mapping found")
            continue

        uniprot_id = uniprot_row.iloc[0]["uniprot_accession"]

        # Get PDB IDs
        ckpt_pdb = Checkpoint(f"s4_pdb_{gene}")
        if ckpt_pdb.exists() and not force_rerun:
            pdb_meta_df = ckpt_pdb.load()
        else:
            try:
                pdb_ids = uniprot_to_pdb_ids(uniprot_id)
                pdb_meta_df = select_top_pdb_entries(pdb_ids, n_top=5)
                ckpt_pdb.save(pdb_meta_df)
            except Exception as e:
                logger.error(f"[{gene}] PDB retrieval failed: {e}")
                continue

        if pdb_meta_df.empty:
            logger.warning(f"[{gene}] No PDB structures found")
            continue

        # Download PDB files
        pdb_entries = []
        for _, row in pdb_meta_df.iterrows():
            pdb_id = row["pdb_id"]
            try:
                pdb_path = download_pdb_file(pdb_id, pdb_dir)
                ligands = json.loads(row.get("ligands", "[]"))
                if ligands:
                    lig_resname = ligands[0]["ligand_id"]
                    box = define_binding_box_from_ligand(pdb_path, lig_resname)
                else:
                    box = {"center_x": 0, "center_y": 0, "center_z": 0,
                           "size_x": 25, "size_y": 25, "size_z": 25}
                    lig_resname = ""
                pdb_entries.append((pdb_id, lig_resname, box))
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"[{gene}] Download failed for {pdb_id}: {e}")

        gene_pdb_map[gene] = pdb_entries
        gene_compounds[gene] = df_scored
        logger.info(f"[{gene}] {len(pdb_entries)} PDB structures ready")

    # Run Cartesian-product docking
    if any(gene_pdb_map.values()):
        docking_results = run_cartesian_docking(
            gene_pdb_map, gene_compounds, docking_dir, force_rerun=force_rerun
        )

        # Visualizations per gene
        for gene in scored_data:
            try:
                plot_docking_heatmap(docking_results, gene)
                plot_top_docking_compounds(docking_results, gene)
            except Exception as e:
                logger.warning(f"[{gene}] Docking plot failed: {e}")

        safe_save_csv(docking_results, RESULTS_DIR / "docking_results_all.csv")
        return docking_results

    logger.warning("No docking was performed (missing software or PDB structures)")
    return pd.DataFrame()


if __name__ == "__main__":
    print("Stage 4 module loaded. Run via main pipeline.")
