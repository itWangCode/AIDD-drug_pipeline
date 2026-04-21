"""
Stage 5 — Protein-Ligand Interaction Analysis
==============================================
Python 3.8 compatible. Reads docking results from Stage 4 (column names:
best_affinity_kcal, molecule_id, complex_pdb) and performs detailed
protein-ligand interaction profiling using PLIP.

Features:
  • Reads top-N complexes by best_affinity_kcal (not 'docking_score')
  • Locates complex PDB from the 'complex_pdb' column written by Stage 4
  • Runs PLIP via Python API (pip install plip) if available;
    falls back to geometry-based analysis when PLIP is not installed
  • Detects all 8 interaction types: hydrophobic, H-bond, water bridge,
    salt bridge, pi-stacking, pi-cation, halogen bond, metal complex
  • Per-gene outputs:
      {GENE}_interactions.csv          — all interaction records
      {GENE}_top_candidates.csv        — top compounds with full metadata
      interactions/{GENE}/             — per-complex interaction JSONs
  • Publication-quality figures (300 dpi PDF + PNG):
      {GENE}_interaction_profile.pdf   — 5-panel interaction summary
      {GENE}_interaction_heatmap.pdf   — compounds × interaction types
      {GENE}_{mol_id}_network.pdf      — per-complex interaction network
      final_summary_all_genes.pdf      — cross-gene overview
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, FIGURES_DIR, TOP_N_DOCKING, NATURE_COLORS, PLOT_CONFIG
from src.utils import setup_logger, safe_save_csv, Checkpoint

logger = setup_logger("stage5_interactions", "stage5.log")

# ── Column name produced by Stage 4 ──────────────────────────────────────────
AFFINITY_COL  = "best_affinity_kcal"   # Stage 4 writes this
MOL_ID_COL    = "molecule_id"          # Stage 4 writes this
COMPLEX_COL   = "complex_pdb"          # Stage 4 writes this
PDB_ID_COL    = "pdb_id"
GENE_COL      = "gene"

# ── Interaction types ─────────────────────────────────────────────────────────
INTERACTION_TYPES = [
    "hydrophobic", "hbond", "waterbridge", "saltbridge",
    "pistacking", "pication", "halogen", "metal",
]

INTERACTION_LABELS = {
    "hydrophobic":  "Hydrophobic",
    "hbond":        "H-Bond",
    "waterbridge":  "Water Bridge",
    "saltbridge":   "Salt Bridge",
    "pistacking":   "pi-Stacking",
    "pication":     "pi-Cation",
    "halogen":      "Halogen Bond",
    "metal":        "Metal Complex",
}

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


# ══════════════════════════════════════════════════════════════════════════════
# PLIP interaction analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_plip_on_complex(complex_pdb: Path) -> Dict:
    """
    Run PLIP protein-ligand interaction profiling on a complex PDB file.

    Returns a dict of {binding_site_key: {interaction_type: pd.DataFrame}}.
    Falls back to empty dict if PLIP is not installed or the file has
    no ligand atoms.

    PLIP install: pip install plip
    """
    try:
        from plip.structure.preparation import PDBComplex
        from plip.exchange.report import BindingSiteReport
    except ImportError:
        logger.debug("PLIP not installed (pip install plip). "
                     "Using geometry fallback for %s.", complex_pdb.name)
        return _geometry_fallback(complex_pdb)
    except Exception as e:
        logger.warning("PLIP import error: %s", e)
        return {}

    try:
        protlig = PDBComplex()
        protlig.load_pdb(str(complex_pdb))

        # Skip if no ligands detected
        if not protlig.ligands:
            logger.debug("No ligands found in %s", complex_pdb.name)
            return {}

        for ligand in protlig.ligands:
            try:
                protlig.characterize_complex(ligand)
            except Exception as e:
                logger.debug("characterize_complex failed for %s: %s", ligand, e)

        all_sites = {}
        for key, site in sorted(protlig.interaction_sets.items()):
            try:
                bsr = BindingSiteReport(site)
                site_data = {}
                for itype in INTERACTION_TYPES:
                    features = getattr(bsr, "{}_features".format(itype), [])
                    info     = getattr(bsr, "{}_info".format(itype), [])
                    if features and info:
                        try:
                            site_data[itype] = pd.DataFrame(info, columns=features)
                        except Exception:
                            site_data[itype] = pd.DataFrame()
                    else:
                        site_data[itype] = pd.DataFrame()
                all_sites[str(key)] = site_data
            except Exception as e:
                logger.debug("BindingSiteReport failed for %s: %s", key, e)

        return all_sites

    except Exception as e:
        logger.warning("PLIP failed on %s: %s", complex_pdb.name, e)
        return {}


def _geometry_fallback(complex_pdb: Path) -> Dict:
    """
    Geometry-based interaction detection when PLIP is not available.
    Identifies interactions by distance thresholds between protein and
    ligand atoms.

    Thresholds (Angstrom):
      Hydrophobic : C-C  <= 4.5
      H-bond      : N/O donor … acceptor <= 3.5
      Halogen     : halogen … O/N <= 3.5
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return {}

    try:
        mol = Chem.MolFromPDBFile(str(complex_pdb), removeHs=False, sanitize=False)
        if mol is None:
            return {}

        conf = mol.GetConformer()
        protein_atoms = []
        ligand_atoms  = []

        for atom in mol.GetAtoms():
            mi = atom.GetMonomerInfo()
            if mi is None:
                continue
            chain = mi.GetChainId()
            resn  = mi.GetResidueName().strip()
            if resn == "LIG" or chain == "L":
                ligand_atoms.append(atom.GetIdx())
            else:
                protein_atoms.append(atom.GetIdx())

        if not ligand_atoms or not protein_atoms:
            return {}

        interactions = defaultdict(list)
        HBOND_ELEMS    = {7, 8}        # N, O
        HYDRO_ELEMS    = {6}            # C
        HALOGEN_ELEMS  = {9, 17, 35, 53}  # F, Cl, Br, I

        for li in ligand_atoms:
            lpos  = conf.GetAtomPosition(li)
            latom = mol.GetAtomWithIdx(li)
            lelem = latom.GetAtomicNum()

            for pi in protein_atoms:
                ppos  = conf.GetAtomPosition(pi)
                patom = mol.GetAtomWithIdx(pi)
                pelem = patom.GetAtomicNum()
                mi    = patom.GetMonomerInfo()
                resn  = mi.GetResidueName().strip() if mi else "UNK"
                resi  = mi.GetResidueSequenceNumber() if mi else 0
                res_label = "{}{}".format(resn, resi)

                dist = (
                    (lpos.x - ppos.x) ** 2 +
                    (lpos.y - ppos.y) ** 2 +
                    (lpos.z - ppos.z) ** 2
                ) ** 0.5

                if dist > 5.0:
                    continue

                if lelem in HYDRO_ELEMS and pelem in HYDRO_ELEMS and dist <= 4.5:
                    interactions["hydrophobic"].append({
                        "RESNR": resi, "RESTYPE": resn, "residue": res_label, "DIST": dist
                    })
                if (lelem in HBOND_ELEMS or pelem in HBOND_ELEMS) and dist <= 3.5:
                    interactions["hbond"].append({
                        "RESNR": resi, "RESTYPE": resn, "residue": res_label,
                        "DIST_D-A": dist
                    })
                if lelem in HALOGEN_ELEMS and pelem in HBOND_ELEMS and dist <= 3.5:
                    interactions["halogen"].append({
                        "RESNR": resi, "RESTYPE": resn, "residue": res_label, "DIST": dist
                    })

        # Deduplicate by residue
        site_data = {}
        for itype in INTERACTION_TYPES:
            rows = interactions.get(itype, [])
            if rows:
                df = pd.DataFrame(rows).drop_duplicates(subset=["RESNR"])
                site_data[itype] = df
            else:
                site_data[itype] = pd.DataFrame()

        return {"AUTO_DETECTED": site_data} if site_data else {}

    except Exception as e:
        logger.warning("Geometry fallback failed: %s", e)
        return {}


def summarize_interactions(all_sites: Dict) -> Dict:
    """Count interactions per type across all binding sites."""
    counts = {}
    for site_data in all_sites.values():
        for itype, df in site_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                counts[itype] = counts.get(itype, 0) + len(df)
    return counts


def get_interacting_residues(all_sites: Dict) -> Dict:
    """Extract unique interacting residues per interaction type."""
    residues = defaultdict(set)
    for site_data in all_sites.values():
        for itype, df in site_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                for col in ["RESNR", "residue"]:
                    if col in df.columns:
                        for val in df[col].dropna():
                            residues[itype].add(str(val))
                        break
                if "RESTYPE" in df.columns and "RESNR" in df.columns:
                    for _, row in df.iterrows():
                        label = "{}{}".format(
                            str(row.get("RESTYPE", "")).strip(),
                            str(row.get("RESNR", "")).strip()
                        )
                        residues[itype].add(label)
    return {k: sorted(v) for k, v in residues.items()}


def _find_complex_pdb(row: pd.Series, gene: str) -> Optional[Path]:
    """
    Locate the complex PDB file for a docking result row.
    Uses 'complex_pdb' column from Stage 4, then tries constructed paths.
    """
    # 1. Direct path from Stage 4 output
    cp = row.get(COMPLEX_COL, "")
    if cp and Path(cp).exists():
        return Path(cp)

    # 2. Construct from standard Stage 4 layout:
    #    results/docking/{gene}/{pdb_id}/{mol_id}/{mol_id}_complex.pdb
    mol_id = str(row.get(MOL_ID_COL, ""))
    pdb_id = str(row.get(PDB_ID_COL, ""))
    if mol_id and pdb_id:
        p = RESULTS_DIR / "docking" / gene / pdb_id / mol_id / "{}_complex.pdb".format(mol_id)
        if p.exists():
            return p

    # 3. Top-6 folder
    if mol_id and pdb_id:
        top6_dir = RESULTS_DIR / "docking" / gene / pdb_id / "top6_best_compounds"
        for f in top6_dir.glob("*{}*complex.pdb".format(mol_id)):
            return f

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Publication-quality figures
# ══════════════════════════════════════════════════════════════════════════════

def _style() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "xtick.labelsize":   11,
        "ytick.labelsize":   11,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.3,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path.name)


def plot_interaction_profile(interactions_df: pd.DataFrame, gene: str,
                              top_candidates: pd.DataFrame) -> None:
    """
    5-panel interaction profile figure per gene.
    A: interaction type frequency bar
    B: interaction type pie
    C: top interacting residues horizontal bar
    D: stacked interactions per compound
    E: affinity vs total interactions scatter
    """
    _style()
    if interactions_df.empty:
        logger.warning("[%s] No interaction data to plot", gene)
        return

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.42)

    # ── A: Interaction type frequency ────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    itype_counts = (
        interactions_df.groupby("interaction_type").size()
        .reindex(INTERACTION_TYPES, fill_value=0)
        .sort_values(ascending=False)
    )
    itype_counts = itype_counts[itype_counts > 0]
    if not itype_counts.empty:
        colors_a = [INTERACTION_COLORS.get(k, "gray") for k in itype_counts.index]
        bars = ax_a.bar(range(len(itype_counts)), itype_counts.values,
                         color=colors_a, edgecolor="white", linewidth=1.2)
        ax_a.set_xticks(range(len(itype_counts)))
        ax_a.set_xticklabels(
            [INTERACTION_LABELS.get(k, k) for k in itype_counts.index],
            rotation=40, ha="right", fontsize=10
        )
        ax_a.set_ylabel("Number of Interactions", fontsize=13, fontweight="bold")
        ax_a.set_title("A   Interaction Type Frequency\n{} | n={} compounds".format(
            gene, interactions_df["molecule_id"].nunique()),
            fontsize=13, fontweight="bold", loc="left")
        for bar, v in zip(bars, itype_counts.values):
            ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                       str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── B: Pie chart ──────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if not itype_counts.empty:
        wedge_colors = [INTERACTION_COLORS.get(k, "gray") for k in itype_counts.index]
        wedges, texts, autotexts = ax_b.pie(
            itype_counts.values,
            labels=[INTERACTION_LABELS.get(k, k) for k in itype_counts.index],
            colors=wedge_colors, autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9},
            wedgeprops={"linewidth": 2, "edgecolor": "white"},
        )
        for a in autotexts:
            a.set_fontsize(9)
            a.set_fontweight("bold")
    ax_b.set_title("B   Interaction Type Distribution", fontsize=13, fontweight="bold", loc="left")

    # ── C: Top interacting residues ───────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    if "residue" in interactions_df.columns:
        res_counts = (
            interactions_df["residue"]
            .dropna()
            .str.split(";")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(20)
        )
        if not res_counts.empty:
            colors_c = plt.cm.viridis(np.linspace(0.2, 0.85, len(res_counts)))
            ax_c.barh(range(len(res_counts)), res_counts.values,
                       color=colors_c, edgecolor="white")
            ax_c.set_yticks(range(len(res_counts)))
            ax_c.set_yticklabels(res_counts.index, fontsize=9)
            ax_c.invert_yaxis()
            ax_c.set_xlabel("Interaction Count", fontsize=12, fontweight="bold")
    ax_c.set_title("C   Top Interacting Residues", fontsize=13, fontweight="bold", loc="left")

    # ── D: Stacked interactions per compound ──────────────────────────────────
    ax_d = fig.add_subplot(gs[1, :2])
    if "molecule_id" in interactions_df.columns:
        pivot = (
            interactions_df.groupby(["molecule_id", "interaction_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=INTERACTION_TYPES, fill_value=0)
        )
        # Sort by total interactions
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=False).drop("total", axis=1).head(20)

        bottom = np.zeros(len(pivot))
        for itype in INTERACTION_TYPES:
            if itype in pivot.columns and pivot[itype].sum() > 0:
                ax_d.bar(range(len(pivot)), pivot[itype].values, bottom=bottom,
                          color=INTERACTION_COLORS.get(itype, "gray"),
                          label=INTERACTION_LABELS.get(itype, itype),
                          edgecolor="white", linewidth=0.5)
                bottom += pivot[itype].values

        ax_d.set_xticks(range(len(pivot)))
        ax_d.set_xticklabels(
            [str(i)[:14] for i in pivot.index],
            rotation=45, ha="right", fontsize=8.5
        )
        ax_d.set_ylabel("Number of Interactions", fontsize=13, fontweight="bold")
        ax_d.set_title("D   Interactions per Compound (Top 20)",
                        fontsize=13, fontweight="bold", loc="left")
        ax_d.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)

    # ── E: Affinity vs interaction count ─────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    if not top_candidates.empty and AFFINITY_COL in top_candidates.columns:
        per_mol = (
            interactions_df.groupby("molecule_id")
            .size()
            .reset_index(name="n_interactions")
        )
        merged = top_candidates[[MOL_ID_COL, AFFINITY_COL, "pIC50", "qed"]].rename(
            columns={MOL_ID_COL: "molecule_id"}
        ).merge(per_mol, on="molecule_id", how="inner")

        if not merged.empty:
            sc = ax_e.scatter(
                pd.to_numeric(merged[AFFINITY_COL], errors="coerce"),
                merged["n_interactions"],
                s=80, alpha=0.75,
                c=pd.to_numeric(merged.get("qed", pd.Series()), errors="coerce"),
                cmap="RdYlGn", vmin=0, vmax=1,
                edgecolors="white", linewidths=0.5
            )
            plt.colorbar(sc, ax=ax_e, label="QED", shrink=0.85)
            ax_e.set_xlabel("Docking Affinity (kcal/mol)", fontsize=12, fontweight="bold")
            ax_e.set_ylabel("Total Interactions Detected", fontsize=12, fontweight="bold")
    ax_e.set_title("E   Affinity vs Interaction Count\n(color = QED Score)",
                    fontsize=13, fontweight="bold", loc="left")

    fig.suptitle(
        "Protein-Ligand Interaction Profile — {}\nPlotted from complex PDB files".format(gene),
        fontsize=18, fontweight="bold"
    )
    _save(fig, FIGURES_DIR / "{}_interaction_profile.pdf".format(gene))


def plot_interaction_heatmap(interactions_df: pd.DataFrame, gene: str) -> None:
    """Heatmap: compounds (rows) x interaction types (cols), value = count."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not installed; skipping interaction heatmap")
        return

    _style()
    if interactions_df.empty or "molecule_id" not in interactions_df.columns:
        return

    pivot = (
        interactions_df.groupby(["molecule_id", "interaction_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=INTERACTION_TYPES, fill_value=0)
    )
    pivot.columns = [INTERACTION_LABELS.get(c, c) for c in pivot.columns]
    pivot = pivot.loc[pivot.sum(axis=1).nlargest(min(30, len(pivot))).index]

    fig_h = max(8, len(pivot) * 0.42)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd",
        annot=True, fmt="d", linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Interaction Count", "shrink": 0.8},
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )
    ax.set_xlabel("Interaction Type", fontsize=13, fontweight="bold")
    ax.set_ylabel("Compound", fontsize=13, fontweight="bold")
    ax.set_title(
        "Protein-Ligand Interaction Heatmap — {}\n"
        "{} compounds | Values = number of contacts".format(gene, len(pivot)),
        fontsize=14, fontweight="bold"
    )
    plt.xticks(rotation=40, ha="right", fontsize=11, fontweight="bold")
    plt.yticks(fontsize=9)
    _save(fig, FIGURES_DIR / "{}_interaction_heatmap.pdf".format(gene))


def plot_interaction_network(residue_interactions: Dict, gene: str, mol_id: str) -> None:
    """Network diagram: ligand node connected to residue nodes by interaction type."""
    _style()
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed; skipping network plot (pip install networkx)")
        return

    G = nx.Graph()
    G.add_node("Ligand", node_type="ligand")

    edge_colors = []
    for itype, residues in residue_interactions.items():
        for res in residues:
            node = "{}".format(res)
            G.add_node(node, node_type="protein")
            G.add_edge("Ligand", node, interaction=itype,
                        color=INTERACTION_COLORS.get(itype, "gray"))

    if G.number_of_edges() == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G, k=2.8, seed=42)

    # Draw protein residue nodes
    prot_nodes = [n for n in G.nodes if n != "Ligand"]
    nx.draw_networkx_nodes(G, pos, nodelist=prot_nodes, ax=ax,
                            node_color=NATURE_COLORS[3], node_size=500, alpha=0.85)
    # Draw ligand node (larger, gold)
    nx.draw_networkx_nodes(G, pos, nodelist=["Ligand"], ax=ax,
                            node_color="gold", node_size=1200, alpha=0.95)

    # Draw edges by interaction type
    for itype in INTERACTION_TYPES:
        edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d.get("interaction") == itype]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                                    edge_color=INTERACTION_COLORS.get(itype, "gray"),
                                    width=2.5, alpha=0.75, style="solid")

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color=INTERACTION_COLORS.get(itype, "gray"),
                        label=INTERACTION_LABELS.get(itype, itype))
        for itype in INTERACTION_TYPES
        if any(d.get("interaction") == itype for _, _, d in G.edges(data=True))
    ]
    legend_items.append(mpatches.Patch(color="gold", label="Ligand ({})".format(mol_id[:12])))
    ax.legend(handles=legend_items, loc="lower right", fontsize=10, framealpha=0.9)

    ax.set_title(
        "Protein-Ligand Interaction Network\n{} — {}".format(gene, str(mol_id)[:20]),
        fontsize=15, fontweight="bold"
    )
    ax.axis("off")
    fig.patch.set_facecolor("white")

    out = FIGURES_DIR / "{}_{}_network.pdf".format(gene, str(mol_id)[:20])
    _save(fig, out)


def plot_final_summary(final_results: Dict) -> None:
    """Cross-gene overview: pIC50 histogram + affinity vs QED per gene."""
    _style()
    genes = [g for g, r in final_results.items()
             if r.get("top_candidates") is not None
             and not r["top_candidates"].empty]
    if not genes:
        return

    ncols = max(2, len(genes))
    fig, axes = plt.subplots(2, ncols, figsize=(8 * ncols, 12))
    if ncols == 1:
        axes = axes.reshape(2, 1)

    for j, gene in enumerate(genes):
        tc = final_results[gene]["top_candidates"].copy()
        tc[AFFINITY_COL] = pd.to_numeric(tc.get(AFFINITY_COL, pd.Series()), errors="coerce")
        tc["pIC50"]       = pd.to_numeric(tc.get("pIC50",       pd.Series()), errors="coerce")
        tc["qed"]         = pd.to_numeric(tc.get("qed",         pd.Series()), errors="coerce")

        # Top row: pIC50 histogram
        ax = axes[0, j]
        if tc["pIC50"].notna().any():
            ax.hist(tc["pIC50"].dropna(), bins=20,
                     color=NATURE_COLORS[j % len(NATURE_COLORS)],
                     edgecolor="white", alpha=0.85)
            ax.axvline(x=6.3, color="crimson", lw=2, ls="--", label="Active (pIC50=6.3)")
            ax.set_xlabel("pIC50", fontsize=13, fontweight="bold")
            ax.set_ylabel("Count", fontsize=13, fontweight="bold")
            ax.set_title("{}\nTop Candidates pIC50".format(gene), fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)

        # Bottom row: affinity vs QED
        ax2 = axes[1, j]
        if tc[AFFINITY_COL].notna().any() and tc["qed"].notna().any():
            sc = ax2.scatter(
                tc[AFFINITY_COL], tc["qed"], s=80, alpha=0.75,
                c=tc["pIC50"] if tc["pIC50"].notna().any() else "steelblue",
                cmap="RdYlGn", edgecolors="white", linewidths=0.5
            )
            if tc["pIC50"].notna().any():
                plt.colorbar(sc, ax=ax2, label="pIC50", shrink=0.85)
            ax2.set_xlabel("Docking Affinity (kcal/mol)", fontsize=13, fontweight="bold")
            ax2.set_ylabel("QED Score", fontsize=13, fontweight="bold")
            ax2.set_title("{}\nQED vs Affinity".format(gene), fontsize=14, fontweight="bold")

    fig.suptitle("Final Drug Candidate Summary — All Targets",
                  fontsize=20, fontweight="bold")
    plt.tight_layout()
    _save(fig, FIGURES_DIR / "final_summary_all_genes.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Interaction record builder
# ══════════════════════════════════════════════════════════════════════════════

def _interactions_to_records(
    all_sites: Dict,
    gene: str,
    mol_id: str,
    pdb_id: str,
    affinity: Optional[float],
    pic50: Optional[float],
    qed: Optional[float],
) -> List[Dict]:
    """Flatten all_sites dict into a list of flat interaction records."""
    records = []
    residues_seen = get_interacting_residues(all_sites)

    for itype in INTERACTION_TYPES:
        res_list = residues_seen.get(itype, [])
        if not res_list:
            continue
        for res in res_list:
            records.append({
                "gene":             gene,
                "molecule_id":      mol_id,
                "pdb_id":           pdb_id,
                "interaction_type": itype,
                "interaction_label": INTERACTION_LABELS.get(itype, itype),
                "residue":          res,
                "affinity_kcal":    affinity,
                "pIC50":            pic50,
                "qed":              qed,
            })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Main Stage 5 function
# ══════════════════════════════════════════════════════════════════════════════

def run_interaction_analysis(
    docking_results: pd.DataFrame,
    scored_data: Dict,
    force_rerun: bool = False,
) -> Dict:
    """
    Stage 5: Protein-ligand interaction profiling for top docking complexes.

    Reads docking results produced by Stage 4 (stage4_docking.py).
    Column names expected: best_affinity_kcal, molecule_id, complex_pdb, pdb_id, gene.

    For each gene:
      - Selects top TOP_N_DOCKING complexes by best_affinity_kcal
      - Runs PLIP (if installed) or geometry fallback on each complex PDB
      - Saves interaction records to CSV
      - Generates 5-panel interaction profile, heatmap, and network figures

    Parameters
    ----------
    docking_results : pd.DataFrame
        Output of Stage 4 run_docking_pipeline().
    scored_data : dict
        {gene: {"df_scored": pd.DataFrame}} from Stage 3.
    force_rerun : bool
        Re-analyse even if checkpoint exists.

    Returns
    -------
    dict
        {gene: {"top_candidates": DataFrame,
                "interactions_df": DataFrame,
                "residue_interactions": dict}}
    """
    interaction_dir = RESULTS_DIR / "interactions"
    interaction_dir.mkdir(exist_ok=True)

    final_results = {}

    # Guard: nothing to analyse
    if docking_results is None or (
        isinstance(docking_results, pd.DataFrame) and docking_results.empty
    ):
        logger.warning("No docking results to analyse in Stage 5")
        return {}

    # Normalise column names: accept both old and new naming
    df = docking_results.copy()

    # Map old column names → new if present
    rename_map = {
        "docking_score":       AFFINITY_COL,
        "molecule_chembl_id":  MOL_ID_COL,
        "out_file":            COMPLEX_COL,
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Validate required columns
    required = [AFFINITY_COL, MOL_ID_COL, GENE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(
            "Stage 5: docking_results is missing columns: %s\n"
            "  Available columns: %s\n"
            "  Expected: %s / %s / %s",
            missing, list(df.columns), AFFINITY_COL, MOL_ID_COL, GENE_COL
        )
        return {}

    df[AFFINITY_COL] = pd.to_numeric(df[AFFINITY_COL], errors="coerce")
    genes_in_results = df[GENE_COL].dropna().unique().tolist()
    logger.info("Stage 5: analysing %d genes: %s", len(genes_in_results), genes_in_results)

    for gene in genes_in_results:
        ckpt = Checkpoint("s5_interactions_{}".format(gene))
        if ckpt.exists() and not force_rerun:
            final_results[gene] = ckpt.load()
            logger.info("[%s] Loaded interaction results from checkpoint", gene)
            continue

        df_gene = (
            df[(df[GENE_COL] == gene) & df[AFFINITY_COL].notna()]
            .nsmallest(TOP_N_DOCKING, AFFINITY_COL)
            .reset_index(drop=True)
        )

        logger.info("[%s] Analysing %d top complexes (TOP_N_DOCKING=%d)",
                    gene, len(df_gene), TOP_N_DOCKING)

        all_records = []
        all_residues = {}
        gene_int_dir = interaction_dir / gene
        gene_int_dir.mkdir(exist_ok=True)

        for _, row in tqdm(df_gene.iterrows(), total=len(df_gene),
                            desc="[{}] PLIP".format(gene)):
            mol_id   = str(row.get(MOL_ID_COL, "unknown"))
            pdb_id   = str(row.get(PDB_ID_COL, ""))
            affinity = row.get(AFFINITY_COL)
            pic50    = row.get("pIC50")
            qed      = row.get("qed")

            # Locate complex PDB
            complex_pdb = _find_complex_pdb(row, gene)
            if complex_pdb is None:
                logger.debug("[%s/%s] complex PDB not found — skipping PLIP", gene, mol_id)
                continue

            logger.info("  [%s] %s | affinity=%.3f | complex=%s",
                        gene, mol_id, affinity if affinity else 0, complex_pdb.name)

            # Run PLIP (or geometry fallback)
            all_sites = run_plip_on_complex(complex_pdb)

            if not all_sites:
                logger.debug("  [%s/%s] No interactions detected", gene, mol_id)
                continue

            # Flatten to records
            records = _interactions_to_records(
                all_sites, gene, mol_id, pdb_id, affinity, pic50, qed
            )
            all_records.extend(records)

            # Store residue sets
            residues = get_interacting_residues(all_sites)
            all_residues[mol_id] = residues

            # Save per-complex JSON
            json_out = gene_int_dir / "{}_interactions.json".format(mol_id)
            try:
                with open(json_out, "w") as fj:
                    json.dump({
                        "mol_id":   mol_id,
                        "pdb_id":   pdb_id,
                        "affinity": affinity,
                        "pIC50":    pic50,
                        "residues": residues,
                        "n_interactions": {
                            itype: len(rlist)
                            for itype, rlist in residues.items()
                        },
                    }, fj, indent=2)
            except Exception as e:
                logger.debug("JSON save failed for %s: %s", mol_id, e)

        # Build DataFrames
        interactions_df  = pd.DataFrame(all_records)
        top_candidates   = df_gene.copy()

        # Merge composite_score from Stage 3 if available
        scored_df = scored_data.get(gene, {})
        if isinstance(scored_df, dict):
            scored_df = scored_df.get("df_scored", pd.DataFrame())
        if not scored_df.empty and "composite_score" in scored_df.columns:
            merge_col = MOL_ID_COL if MOL_ID_COL in scored_df.columns else "molecule_chembl_id"
            if merge_col in scored_df.columns:
                top_candidates = top_candidates.merge(
                    scored_df[[merge_col, "composite_score"]].rename(
                        columns={merge_col: MOL_ID_COL}),
                    on=MOL_ID_COL, how="left"
                )

        # Save CSVs
        safe_save_csv(interactions_df, RESULTS_DIR / "{}_interactions.csv".format(gene))
        safe_save_csv(top_candidates,  RESULTS_DIR / "{}_top_candidates.csv".format(gene))

        # Checkpoint
        gene_result = {
            "top_candidates":      top_candidates,
            "interactions_df":     interactions_df,
            "residue_interactions": all_residues,
        }
        ckpt.save(gene_result)
        final_results[gene] = gene_result

        # Figures
        try:
            plot_interaction_profile(interactions_df, gene, top_candidates)
        except Exception as e:
            logger.warning("[%s] plot_interaction_profile failed: %s", gene, e)

        try:
            plot_interaction_heatmap(interactions_df, gene)
        except Exception as e:
            logger.warning("[%s] plot_interaction_heatmap failed: %s", gene, e)

        # Network plots for top-3 complexes
        for mol_id, res_ints in list(all_residues.items())[:3]:
            if res_ints:
                try:
                    plot_interaction_network(res_ints, gene, mol_id)
                except Exception as e:
                    logger.warning("[%s/%s] network plot failed: %s", gene, mol_id, e)

    # Cross-gene summary
    try:
        plot_final_summary(final_results)
    except Exception as e:
        logger.warning("Final summary plot failed: %s", e)

    return final_results