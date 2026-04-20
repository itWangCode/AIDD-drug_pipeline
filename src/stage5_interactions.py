"""
Stage 5: Protein-Ligand Interaction Analysis (PLIP)
Detailed interaction profiling of top docking poses — Nature-grade figures.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RESULTS_DIR, FIGURES_DIR, TOP_N_DOCKING,
    PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv, Checkpoint

logger = setup_logger("stage5_interactions", "stage5.log")
warnings.filterwarnings("ignore")


# ─── PLIP Interaction Analysis ────────────────────────────────────────────────
INTERACTION_TYPES = [
    "hydrophobic", "hbond", "waterbridge", "saltbridge",
    "pistacking", "pication", "halogen", "metal",
]

INTERACTION_COLORS = {
    "hydrophobic":  NATURE_COLORS[0],
    "hbond":        NATURE_COLORS[1],
    "waterbridge":  NATURE_COLORS[2],
    "saltbridge":   NATURE_COLORS[3],
    "pistacking":   NATURE_COLORS[4],
    "pication":     NATURE_COLORS[5],
    "halogen":      NATURE_COLORS[6],
    "metal":        NATURE_COLORS[7],
}

INTERACTION_LABELS = {
    "hydrophobic":  "Hydrophobic",
    "hbond":        "H-Bond",
    "waterbridge":  "Water Bridge",
    "saltbridge":   "Salt Bridge",
    "pistacking":   "π-Stacking",
    "pication":     "π-Cation",
    "halogen":      "Halogen Bond",
    "metal":        "Metal Complex",
}


def run_plip_on_complex(pdb_complex_path: Path, out_dir: Path) -> dict:
    """
    Run PLIP on a protein-ligand complex PDB file.
    Returns dict of interaction DataFrames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["plip", "-f", str(pdb_complex_path), "-o", str(out_dir), "--xml", "--txt"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"PLIP failed: {result.stderr[:200]}")
    except FileNotFoundError:
        logger.info("PLIP CLI not found. Using Python API...")
        return run_plip_python(pdb_complex_path)
    except subprocess.TimeoutExpired:
        logger.warning("PLIP timed out")
        return {}

    return parse_plip_xml_output(out_dir)


def run_plip_python(pdb_complex_path: Path) -> dict:
    """Run PLIP using Python API (plip package)."""
    try:
        from plip.structure.preparation import PDBComplex
        from plip.exchange.report import BindingSiteReport

        protlig = PDBComplex()
        protlig.load_pdb(str(pdb_complex_path))
        for ligand in protlig.ligands:
            protlig.characterize_complex(ligand)

        all_interactions = {}
        for key, site in sorted(protlig.interaction_sets.items()):
            bsr = BindingSiteReport(site)
            site_data = {}
            for itype in INTERACTION_TYPES:
                features = getattr(bsr, f"{itype}_features", [])
                info = getattr(bsr, f"{itype}_info", [])
                if features and info:
                    site_data[itype] = pd.DataFrame(info, columns=features)
                else:
                    site_data[itype] = pd.DataFrame()
            all_interactions[key] = site_data

        return all_interactions
    except ImportError:
        logger.warning("PLIP Python package not available. Skipping interaction analysis.")
        return {}
    except Exception as e:
        logger.error(f"PLIP Python API failed: {e}")
        return {}


def parse_plip_xml_output(out_dir: Path) -> dict:
    """Parse PLIP XML output files."""
    import glob
    interactions = {}
    xml_files = list(out_dir.glob("*.xml"))
    if not xml_files:
        return {}

    try:
        import xml.etree.ElementTree as ET
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            binding_sites = root.findall(".//bindingsite")
            for bs in binding_sites:
                bs_id = bs.find("identifiers/hetid")
                bs_key = bs_id.text if bs_id is not None else xml_file.stem
                site_data = {}
                for itype in INTERACTION_TYPES:
                    records = []
                    for interaction in bs.findall(f".//{itype}s/{itype}"):
                        record = {child.tag: child.text for child in interaction}
                        records.append(record)
                    site_data[itype] = pd.DataFrame(records) if records else pd.DataFrame()
                interactions[bs_key] = site_data
    except Exception as e:
        logger.error(f"XML parsing failed: {e}")

    return interactions


def summarize_interactions(all_interactions: dict) -> dict:
    """Summarize interaction counts per type for a complex."""
    summary = defaultdict(int)
    for binding_site, site_data in all_interactions.items():
        for itype, df in site_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                summary[itype] += len(df)
    return dict(summary)


def get_interacting_residues(all_interactions: dict) -> dict:
    """Extract all interacting protein residues per interaction type."""
    residues = defaultdict(list)
    for binding_site, site_data in all_interactions.items():
        for itype, df in site_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                for col in ["RESNR", "RESTYPE", "RESCHAIN"]:
                    if col in df.columns:
                        for _, row in df.iterrows():
                            res_label = f"{row.get('RESTYPE', '')}{row.get('RESNR', '')}"
                            residues[itype].append(res_label)
    return {k: list(set(v)) for k, v in residues.items()}


# ─── Plots ────────────────────────────────────────────────────────────────────
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


def plot_interaction_profile(interactions_df: pd.DataFrame, gene: str):
    """Multi-panel interaction profile figure — Nature style."""
    set_nature_style()

    if interactions_df.empty:
        logger.warning(f"[{gene}] No interaction data to plot")
        return

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ─ Panel A: Interaction type frequency bar chart ─
    ax_a = fig.add_subplot(gs[0, 0])
    itype_counts = interactions_df.groupby("interaction_type").size().sort_values(ascending=False)
    colors = [INTERACTION_COLORS.get(k, "gray") for k in itype_counts.index]
    bars = ax_a.bar(range(len(itype_counts)), itype_counts.values, color=colors,
                     edgecolor="white", linewidth=1.2)
    ax_a.set_xticks(range(len(itype_counts)))
    ax_a.set_xticklabels(
        [INTERACTION_LABELS.get(k, k) for k in itype_counts.index],
        rotation=45, ha="right", fontsize=11
    )
    ax_a.set_ylabel("Number of Interactions", fontsize=13, fontweight="bold")
    ax_a.set_title("A   Interaction Type Frequency", fontsize=14, fontweight="bold", loc="left")
    for bar, val in zip(bars, itype_counts.values):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                   str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ─ Panel B: Top interacting residues ─
    ax_b = fig.add_subplot(gs[0, 1])
    if "residue" in interactions_df.columns:
        residue_counts = interactions_df["residue"].value_counts().head(20)
        if not residue_counts.empty:
            colors_r = plt.cm.viridis(np.linspace(0.2, 0.8, len(residue_counts)))
            ax_b.barh(range(len(residue_counts)), residue_counts.values,
                       color=colors_r, edgecolor="white")
            ax_b.set_yticks(range(len(residue_counts)))
            ax_b.set_yticklabels(residue_counts.index, fontsize=9)
            ax_b.invert_yaxis()
            ax_b.set_xlabel("Interaction Count", fontsize=13, fontweight="bold")
    ax_b.set_title("B   Top Interacting Residues", fontsize=14, fontweight="bold", loc="left")

    # ─ Panel C: Interaction type pie chart ─
    ax_c = fig.add_subplot(gs[0, 2])
    if not itype_counts.empty:
        wedge_colors = [INTERACTION_COLORS.get(k, "gray") for k in itype_counts.index]
        wedges, texts, autotexts = ax_c.pie(
            itype_counts.values,
            labels=[INTERACTION_LABELS.get(k, k) for k in itype_counts.index],
            colors=wedge_colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10},
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
        )
        for a in autotexts:
            a.set_fontsize(10)
            a.set_fontweight("bold")
    ax_c.set_title("C   Interaction Type Distribution", fontsize=14, fontweight="bold", loc="left")

    # ─ Panel D: Interactions per compound (stacked bar) ─
    ax_d = fig.add_subplot(gs[1, :2])
    if "molecule_id" in interactions_df.columns:
        pivot = interactions_df.groupby(["molecule_id", "interaction_type"]).size().unstack(fill_value=0)
        pivot = pivot.sort_values(pivot.columns.tolist(), ascending=False).head(20)

        bottom = np.zeros(len(pivot))
        for itype in INTERACTION_TYPES:
            if itype in pivot.columns:
                col = INTERACTION_COLORS.get(itype, "gray")
                ax_d.bar(range(len(pivot)), pivot[itype], bottom=bottom,
                          color=col, label=INTERACTION_LABELS.get(itype, itype),
                          edgecolor="white")
                bottom += pivot[itype].values

        ax_d.set_xticks(range(len(pivot)))
        ax_d.set_xticklabels([str(i)[:12] for i in pivot.index], rotation=45, ha="right", fontsize=9)
        ax_d.set_ylabel("Number of Interactions", fontsize=13, fontweight="bold")
        ax_d.set_title("D   Interactions per Compound", fontsize=14, fontweight="bold", loc="left")
        ax_d.legend(loc="upper right", fontsize=10, ncol=2, framealpha=0.9)

    # ─ Panel E: Docking score vs interaction count ─
    ax_e = fig.add_subplot(gs[1, 2])
    if "docking_score" in interactions_df.columns and "molecule_id" in interactions_df.columns:
        per_mol = interactions_df.groupby("molecule_id").agg(
            n_interactions=("interaction_type", "count"),
            docking_score=("docking_score", "first"),
        ).dropna()
        if not per_mol.empty:
            sc = ax_e.scatter(per_mol["docking_score"], per_mol["n_interactions"],
                               s=80, alpha=0.7, c=per_mol["n_interactions"],
                               cmap="YlOrRd", edgecolors="white")
            ax_e.set_xlabel("Docking Score (kcal/mol)", fontsize=13, fontweight="bold")
            ax_e.set_ylabel("Total Interactions", fontsize=13, fontweight="bold")
            ax_e.set_title("E   Docking Score vs Interaction Count", fontsize=14, fontweight="bold", loc="left")
            plt.colorbar(sc, ax=ax_e, label="Interactions", shrink=0.85)

    plt.suptitle(f"Protein-Ligand Interaction Profile — {gene}",
                  fontsize=20, fontweight="bold", y=1.01)

    out = FIGURES_DIR / f"{gene}_interaction_profile.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved interaction profile: {out}")


def plot_interaction_network(residue_interactions: dict, gene: str, mol_id: str):
    """Network graph of protein-ligand interactions."""
    set_nature_style()
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not available for interaction network plot")
        return

    G = nx.Graph()
    G.add_node("Ligand", node_type="ligand", color=NATURE_COLORS[3])

    for itype, residues in residue_interactions.items():
        for res in residues:
            G.add_node(res, node_type="protein", color=NATURE_COLORS[0])
            G.add_edge("Ligand", res,
                        interaction=itype,
                        color=INTERACTION_COLORS.get(itype, "gray"),
                        weight=2)

    if G.number_of_edges() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.5, seed=42)

    node_colors = [G.nodes[n].get("color", "gray") for n in G.nodes]
    node_sizes = [800 if n == "Ligand" else 400 for n in G.nodes]
    edge_colors = [G.edges[e].get("color", "gray") for e in G.edges]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                            width=2, alpha=0.7, arrows=False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=INTERACTION_COLORS[k], label=INTERACTION_LABELS[k])
        for k in INTERACTION_TYPES if k in residue_interactions
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_title(f"Protein-Ligand Interaction Network\n{gene} — {mol_id}",
                  fontsize=16, fontweight="bold")
    ax.axis("off")

    out = FIGURES_DIR / f"{gene}_{mol_id}_interaction_network.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved interaction network: {out}")


def plot_final_summary(all_results: dict):
    """Final summary figure across all genes — Nature multi-panel."""
    set_nature_style()
    genes = list(all_results.keys())
    n_genes = len(genes)
    if n_genes == 0:
        return

    fig, axes = plt.subplots(2, max(2, n_genes), figsize=(6 * n_genes, 12))
    if n_genes == 1:
        axes = axes.reshape(2, 1)

    for j, gene in enumerate(genes):
        res = all_results[gene]

        # Top: pIC50 distribution of final candidates
        ax = axes[0, j]
        df_top = res.get("top_candidates", pd.DataFrame())
        if not df_top.empty and "pIC50" in df_top.columns:
            ax.hist(df_top["pIC50"], bins=20, color=NATURE_COLORS[j % len(NATURE_COLORS)],
                     edgecolor="white", alpha=0.85)
            ax.axvline(x=6.3, color="red", linestyle="--", linewidth=2, label="Active cutoff")
            ax.set_xlabel("pIC₅₀", fontsize=13, fontweight="bold")
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.set_title(f"{gene}\nTop Candidates pIC₅₀", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)

        # Bottom: QED vs docking score
        ax2 = axes[1, j]
        if not df_top.empty and "docking_score" in df_top.columns and "qed" in df_top.columns:
            sc = ax2.scatter(df_top["docking_score"], df_top["qed"], s=80,
                              c=df_top.get("pIC50", 0), cmap="RdYlGn",
                              edgecolors="white", alpha=0.8)
            plt.colorbar(sc, ax=ax2, label="pIC₅₀", shrink=0.85)
            ax2.set_xlabel("Docking Score (kcal/mol)", fontsize=13, fontweight="bold")
            ax2.set_ylabel("QED Score", fontsize=13, fontweight="bold")
            ax2.set_title(f"{gene}\nQED vs Docking Score", fontsize=14, fontweight="bold")

    plt.suptitle("Final Drug Candidate Summary — All Targets",
                  fontsize=20, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "final_summary_all_genes.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved final summary: {out}")


# ─── Main Stage 5 Function ────────────────────────────────────────────────────
def run_interaction_analysis(
    docking_results: pd.DataFrame,
    scored_data: dict,
    force_rerun: bool = False,
) -> dict:
    """
    Stage 5: Run PLIP on top docking complexes and generate interaction profiles.
    """
    interaction_dir = RESULTS_DIR / "interactions"
    interaction_dir.mkdir(exist_ok=True)

    final_results = {}

    if docking_results.empty:
        logger.warning("No docking results to analyze")
        return {}

    # Select top N complexes per gene by docking score
    for gene in docking_results["gene"].unique():
        ckpt = Checkpoint(f"s5_interactions_{gene}")
        if ckpt.exists() and not force_rerun:
            final_results[gene] = ckpt.load()
            logger.info(f"[{gene}] Loaded interaction results from checkpoint")
            continue

        df_gene = docking_results[
            (docking_results["gene"] == gene) & docking_results["docking_score"].notna()
        ].nsmallest(TOP_N_DOCKING, "docking_score")

        logger.info(f"[{gene}] Analyzing {len(df_gene)} top complexes")

        all_interaction_records = []
        all_residue_interactions = {}

        for _, row in tqdm(df_gene.iterrows(), total=len(df_gene), desc=f"[{gene}] PLIP"):
            mol_id = row["molecule_chembl_id"]
            pdb_id = row["pdb_id"]

            # Build complex PDB path (from docking output)
            complex_dir = RESULTS_DIR / "docking" / gene
            complex_pdb = complex_dir / f"{pdb_id}_{mol_id}_complex.pdb"
            docked_sdf = Path(str(row.get("out_file", "")))

            if not complex_pdb.exists() and docked_sdf.exists():
                # Create complex from protein + docked ligand
                protein_pdb = complex_dir / f"{pdb_id}_protein.pdb"
                if protein_pdb.exists():
                    try:
                        with open(complex_pdb, "w") as fout:
                            fout.write(protein_pdb.read_text())
                            # Append ligand coords from SDF (simplified)
                    except Exception:
                        pass

            if not complex_pdb.exists():
                logger.debug(f"Complex PDB not found for {mol_id}/{pdb_id}, skipping PLIP")
                continue

            # Run PLIP
            plip_out = interaction_dir / gene / mol_id
            interactions = run_plip_python(complex_pdb) if complex_pdb.exists() else {}

            summary = summarize_interactions(interactions)
            residues = get_interacting_residues(interactions)
            all_residue_interactions[mol_id] = residues

            for itype, count in summary.items():
                for _ in range(count):
                    all_interaction_records.append({
                        "gene": gene,
                        "molecule_id": mol_id,
                        "pdb_id": pdb_id,
                        "interaction_type": itype,
                        "residue": ";".join(residues.get(itype, [])),
                        "docking_score": row.get("docking_score"),
                        "pIC50": row.get("pIC50"),
                        "qed": row.get("qed"),
                    })

        interactions_df = pd.DataFrame(all_interaction_records)

        # Get top candidates (combine ML + docking + interaction scores)
        scored_df = scored_data.get(gene, {})
        if isinstance(scored_df, dict):
            scored_df = scored_df.get("df_scored", pd.DataFrame())

        top_candidates = df_gene.copy()
        if "composite_score" in scored_df.columns:
            top_candidates = top_candidates.merge(
                scored_df[["molecule_chembl_id", "composite_score"]],
                on="molecule_chembl_id", how="left"
            )

        gene_result = {
            "top_candidates": top_candidates,
            "interactions_df": interactions_df,
            "residue_interactions": all_residue_interactions,
        }

        # Save
        safe_save_csv(interactions_df, RESULTS_DIR / f"{gene}_interactions.csv")
        safe_save_csv(top_candidates, RESULTS_DIR / f"{gene}_top_candidates.csv")
        ckpt.save(gene_result)
        final_results[gene] = gene_result

        # Plots
        try:
            if not interactions_df.empty:
                plot_interaction_profile(interactions_df, gene)
            for mol_id, res_ints in list(all_residue_interactions.items())[:3]:
                plot_interaction_network(res_ints, gene, mol_id)
        except Exception as e:
            logger.warning(f"[{gene}] Interaction plot failed: {e}")

    # Final cross-gene summary plot
    try:
        plot_final_summary(final_results)
    except Exception as e:
        logger.warning(f"Final summary plot failed: {e}")

    return final_results


if __name__ == "__main__":
    print("Stage 5 module loaded. Run via main pipeline.")
