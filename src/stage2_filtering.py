"""
Stage 2: Lipinski Ro5 + ADMET Filtering
Comprehensive drug-likeness screening with Nature-grade visualizations.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, FIGURES_DIR, RO5_CONFIG, ADMET_CONFIG,
    PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv

logger = setup_logger("stage2_filtering", "stage2.log")
warnings.filterwarnings("ignore")


# ─── Molecular Property Calculations ─────────────────────────────────────────
def calculate_properties(smiles: str) -> dict:
    """Calculate all relevant molecular properties from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "mw": Descriptors.ExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "qed": QED.qed(mol),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "stereo_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        }
    except Exception:
        return {}


def apply_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Apply property calculation to all compounds."""
    logger.info("Calculating molecular properties...")
    props = df["smiles"].apply(calculate_properties)
    props_df = pd.DataFrame(list(props))
    df = pd.concat([df.reset_index(drop=True), props_df], axis=1)
    # Remove compounds with failed property calculation
    df.dropna(subset=["mw", "logp", "qed"], inplace=True)
    logger.info(f"Properties calculated for {len(df)} compounds")
    return df


# ─── Lipinski Ro5 ─────────────────────────────────────────────────────────────
def check_ro5(row: pd.Series) -> dict:
    """Check Lipinski Ro5 compliance. Returns detailed violation info."""
    violations = []
    if row.get("mw", 0) > RO5_CONFIG["mw_max"]:
        violations.append(f"MW={row['mw']:.0f}>{RO5_CONFIG['mw_max']}")
    if row.get("hba", 0) > RO5_CONFIG["hba_max"]:
        violations.append(f"HBA={row['hba']:.0f}>{RO5_CONFIG['hba_max']}")
    if row.get("hbd", 0) > RO5_CONFIG["hbd_max"]:
        violations.append(f"HBD={row['hbd']:.0f}>{RO5_CONFIG['hbd_max']}")
    if row.get("logp", 0) > RO5_CONFIG["logp_max"]:
        violations.append(f"LogP={row['logp']:.2f}>{RO5_CONFIG['logp_max']}")
    return {
        "ro5_violations": len(violations),
        "ro5_violation_details": "; ".join(violations),
        "ro5_pass": len(violations) <= RO5_CONFIG["violations_allowed"],
    }


def apply_ro5(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Ro5 filter to DataFrame."""
    ro5_results = df.apply(check_ro5, axis=1)
    ro5_df = pd.DataFrame(list(ro5_results))
    df = pd.concat([df.reset_index(drop=True), ro5_df], axis=1)
    n_pass = df["ro5_pass"].sum()
    logger.info(f"Ro5 filter: {n_pass}/{len(df)} compounds pass ({n_pass/len(df)*100:.1f}%)")
    return df


# ─── ADMET Filter ─────────────────────────────────────────────────────────────
def check_admet(row: pd.Series) -> dict:
    """Apply ADMET property filters."""
    cfg = ADMET_CONFIG
    checks = {
        "admet_qed": row.get("qed", 0) >= cfg["qed_min"],
        "admet_tpsa": row.get("tpsa", 999) <= cfg["tpsa_max"],
        "admet_rotb": row.get("rotatable_bonds", 999) <= cfg["rotatable_bonds_max"],
        "admet_mw_range": cfg["mw_min"] <= row.get("mw", 0) <= cfg["mw_max"],
        "admet_logp_range": cfg["logp_min"] <= row.get("logp", 99) <= cfg["logp_max"],
        "admet_hbd": row.get("hbd", 999) <= cfg["hbd_max"],
        "admet_hba": row.get("hba", 999) <= cfg["hba_max"],
        "admet_rings": row.get("rings", 999) <= cfg["rings_max"],
    }
    checks["admet_pass"] = all(checks.values())
    return checks


def apply_admet(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ADMET filter to DataFrame."""
    admet_results = df.apply(check_admet, axis=1)
    admet_df = pd.DataFrame(list(admet_results))
    df = pd.concat([df.reset_index(drop=True), admet_df], axis=1)
    n_pass = df["admet_pass"].sum()
    logger.info(f"ADMET filter: {n_pass}/{len(df)} compounds pass ({n_pass/len(df)*100:.1f}%)")
    return df


# ─── PAINS & Unwanted Substructures ──────────────────────────────────────────
def apply_pains_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter PAINS (Pan Assay Interference Compounds) using RDKit."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    pains_flags = []
    pains_descriptions = []
    for smiles in df["smiles"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                pains_flags.append(True)
                pains_descriptions.append("invalid_smiles")
                continue
            entry = catalog.GetFirstMatch(mol)
            if entry is not None:
                pains_flags.append(True)
                pains_descriptions.append(entry.GetDescription())
            else:
                pains_flags.append(False)
                pains_descriptions.append("")
        except Exception:
            pains_flags.append(True)
            pains_descriptions.append("error")

    df["is_pains"] = pains_flags
    df["pains_description"] = pains_descriptions
    df["pains_pass"] = ~df["is_pains"]
    n_pains = df["is_pains"].sum()
    logger.info(f"PAINS filter: {n_pains} compounds flagged, {df['pains_pass'].sum()} pass")
    return df


# ─── Brenk Unwanted Substructures ────────────────────────────────────────────
BRENK_SMARTS = {
    "michael_acceptor": "[$(C=C!@CC=O),$(C=C-C=O)]",
    "aldehyde": "[CH1](=O)",
    "catechol": "c1ccc(O)c(O)c1",
    "para_hydroxy_styrene": "O-c1ccc(/C=C/)cc1",
    "nitro_group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "thiocarbonyl": "[#6]=S",
    "halopyridine": "[n]1ccccc1[F,Cl,Br,I]",
    "imine": "[$([CH2]=[NH]),$([CH]=[NH])]=C",
    "isolated_alkene": "[$([CH2]=[CH2]),$([CH2]=[CH]),$([CH]=[CH])]",
    "triple_bond": "C#C",
    "thiol": "[SH]",
    "phosphate": "P(=O)(O)(O)O",
    "sulfate": "OS(=O)(=O)O",
    "long_aliphatic": "CCCCCC",
    "aniline": "c-[NH2]",
}


def apply_brenk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter Brenk unwanted substructures."""
    # Compile patterns
    compiled = {}
    for name, smarts in BRENK_SMARTS.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                compiled[name] = pat
        except Exception:
            pass

    brenk_flags = []
    brenk_matched = []
    for smiles in df["smiles"]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                brenk_flags.append(True)
                brenk_matched.append("invalid")
                continue
            matched = [name for name, pat in compiled.items() if mol.HasSubstructMatch(pat)]
            brenk_flags.append(len(matched) > 0)
            brenk_matched.append(";".join(matched))
        except Exception:
            brenk_flags.append(True)
            brenk_matched.append("error")

    df["has_unwanted_substructure"] = brenk_flags
    df["unwanted_substructures"] = brenk_matched
    df["brenk_pass"] = ~df["has_unwanted_substructure"]
    n_brenk = df["has_unwanted_substructure"].sum()
    logger.info(f"Brenk filter: {n_brenk} compounds flagged, {df['brenk_pass'].sum()} pass")
    return df


# ─── Combined Filter Flag ─────────────────────────────────────────────────────
def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all filters and add a combined 'passed_all' flag."""
    df["passed_all_filters"] = (
        df["ro5_pass"] & df["admet_pass"] & df["pains_pass"] & df["brenk_pass"]
    )
    n_pass = df["passed_all_filters"].sum()
    logger.info(f"Combined filters: {n_pass}/{len(df)} compounds pass ALL filters ({n_pass/len(df)*100:.1f}%)")
    return df


# ─── Publication-Quality Visualizations ──────────────────────────────────────
def set_nature_style():
    """Apply Nature journal plot aesthetics."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": PLOT_CONFIG["font_size"],
        "axes.titlesize": PLOT_CONFIG["title_size"],
        "axes.labelsize": PLOT_CONFIG["label_size"],
        "xtick.labelsize": PLOT_CONFIG["tick_size"],
        "ytick.labelsize": PLOT_CONFIG["tick_size"],
        "legend.fontsize": PLOT_CONFIG["legend_size"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": PLOT_CONFIG["dpi"],
        "savefig.dpi": PLOT_CONFIG["dpi"],
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_ro5_radar(df_pass: pd.DataFrame, df_fail: pd.DataFrame, gene: str):
    """Nature-grade Ro5 radar chart."""
    set_nature_style()
    properties = ["mw", "logp", "hba", "hbd"]
    thresholds = [500, 5, 10, 5]
    scaled_threshold = 5
    labels = ["MW (Da)/100", "LogP", "HBA/2", "HBD"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))

    for ax, (subdf, title, color) in zip(axes, [
        (df_pass, f"Ro5-Compliant\n(n={len(df_pass)})", NATURE_COLORS[1]),
        (df_fail, f"Ro5-Violated\n(n={len(df_fail)})", NATURE_COLORS[0]),
    ]):
        # Scale values
        means, stds = [], []
        for prop, thresh in zip(properties, thresholds):
            vals = subdf[prop].dropna()
            if len(vals) == 0:
                means.append(0); stds.append(0)
                continue
            means.append(vals.mean() / thresh * scaled_threshold)
            stds.append(vals.std() / thresh * scaled_threshold)

        n_axes = len(properties)
        angles = [i / float(n_axes) * 2 * np.pi for i in range(n_axes)]
        angles += angles[:1]
        means_plot = means + [means[0]]
        stds_plot = stds + [stds[0]]

        ax.fill(angles, [scaled_threshold] * len(angles), alpha=0.15, color=NATURE_COLORS[3], label="Ro5 threshold")
        ax.plot(angles, means_plot, "o-", linewidth=2.5, color=color, markersize=8, label="Mean")
        ax.fill_between(angles,
                         [m - s for m, s in zip(means_plot, stds_plot)],
                         [m + s for m, s in zip(means_plot, stds_plot)],
                         alpha=0.2, color=color, label="±SD")

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 8)
        ax.set_yticks([scaled_threshold])
        ax.set_yticklabels(["5"], fontsize=11, color="gray")
        ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_rlabel_position(0)

    plt.suptitle(f"Lipinski Rule of Five Analysis — {gene}", fontsize=18, fontweight="bold", y=1.02)
    out = FIGURES_DIR / f"{gene}_ro5_radar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Ro5 radar: {out}")


def plot_property_distributions(df: pd.DataFrame, gene: str):
    """Property distribution violin/box plots — Nature style."""
    set_nature_style()
    props = {
        "mw": ("Molecular Weight (Da)", RO5_CONFIG["mw_max"]),
        "logp": ("LogP", RO5_CONFIG["logp_max"]),
        "hba": ("H-Bond Acceptors", RO5_CONFIG["hba_max"]),
        "hbd": ("H-Bond Donors", RO5_CONFIG["hbd_max"]),
        "tpsa": ("TPSA (Å²)", ADMET_CONFIG["tpsa_max"]),
        "qed": ("QED Score", ADMET_CONFIG["qed_min"]),
        "rotatable_bonds": ("Rotatable Bonds", ADMET_CONFIG["rotatable_bonds_max"]),
        "fsp3": ("Fsp³", None),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Categories
    df_pass = df[df["passed_all_filters"]]
    df_fail = df[~df["passed_all_filters"]]

    for ax, (prop, (label, threshold)) in zip(axes, props.items()):
        if prop not in df.columns:
            ax.set_visible(False)
            continue
        pass_vals = df_pass[prop].dropna().values
        fail_vals = df_fail[prop].dropna().values

        data_to_plot = [pass_vals, fail_vals] if len(fail_vals) > 0 else [pass_vals]
        colors_vp = [NATURE_COLORS[1], NATURE_COLORS[0]] if len(fail_vals) > 0 else [NATURE_COLORS[1]]
        labels_vp = ["Pass", "Fail"] if len(fail_vals) > 0 else ["Pass"]

        parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)),
                               showmedians=True, showextrema=False)
        for pc, col in zip(parts["bodies"], colors_vp):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        parts["cmedians"].set_colors("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(labels_vp)))
        ax.set_xticklabels(labels_vp, fontsize=13, fontweight="bold")
        ax.set_ylabel(label, fontsize=13, fontweight="bold")

        if threshold is not None:
            ax.axhline(y=threshold, color="crimson", linestyle="--", linewidth=1.5,
                        alpha=0.8, label=f"Threshold: {threshold}")
            ax.legend(fontsize=10)

        ax.set_title(label, fontsize=13, fontweight="bold", pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(f"Molecular Property Distributions — {gene}", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGURES_DIR / f"{gene}_property_distributions.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved property distributions: {out}")


def plot_filtering_funnel(filter_stats: dict, gene: str):
    """Waterfall funnel plot showing compound attrition at each filter stage."""
    set_nature_style()
    stages = list(filter_stats.keys())
    counts = list(filter_stats.values())
    colors = [NATURE_COLORS[i % len(NATURE_COLORS)] for i in range(len(stages))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: bar chart funnel
    ax = axes[0]
    bars = ax.barh(range(len(stages)), counts, color=colors, edgecolor="white",
                   linewidth=1.2, height=0.7)
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Compounds", fontsize=14, fontweight="bold")
    ax.set_title("Compound Attrition Funnel", fontsize=16, fontweight="bold")
    ax.invert_yaxis()

    # Labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={count:,}", va="center", ha="left", fontsize=12, fontweight="bold")

    # Right: pie chart of final pass/fail
    ax2 = axes[1]
    n_total = counts[0]
    n_final = counts[-1]
    n_removed = n_total - n_final
    sizes = [n_final, n_removed]
    labels_pie = [f"Passed All Filters\n(n={n_final:,})", f"Removed\n(n={n_removed:,})"]
    wedges, texts, autotexts = ax2.pie(
        sizes, labels=labels_pie, colors=[NATURE_COLORS[1], NATURE_COLORS[0]],
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 13},
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    for autotext in autotexts:
        autotext.set_fontsize(13)
        autotext.set_fontweight("bold")
    ax2.set_title(f"Overall Filter Outcome\n(from {n_total:,} compounds)", fontsize=14, fontweight="bold")

    plt.suptitle(f"Drug-Likeness Filtering Pipeline — {gene}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"{gene}_filtering_funnel.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved filtering funnel: {out}")


def plot_pIC50_distribution(df: pd.DataFrame, gene: str, pic50_cutoff: float = 6.3):
    """pIC50 histogram with active/inactive shading."""
    set_nature_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: histogram of all passed compounds
    df_passed = df[df["passed_all_filters"]]
    ax1.hist(df_passed["pIC50"], bins=50, color=NATURE_COLORS[3], edgecolor="white",
             alpha=0.85, linewidth=0.5)
    ax1.axvline(x=pic50_cutoff, color=NATURE_COLORS[0], linewidth=2.5, linestyle="--",
                label=f"Activity cutoff (pIC₅₀={pic50_cutoff})")
    ax1.fill_betweenx([0, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 100],
                       pic50_cutoff, ax1.get_xlim()[1] if ax1.get_xlim()[1] > 0 else 12,
                       alpha=0.1, color=NATURE_COLORS[1], label="Active region")
    ax1.set_xlabel("pIC₅₀", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of Compounds", fontsize=14, fontweight="bold")
    ax1.set_title(f"pIC₅₀ Distribution (Filtered Set)\nn={len(df_passed)}", fontsize=15, fontweight="bold")
    ax1.legend(fontsize=12)

    # Re-draw with correct y limit
    ylim = ax1.get_ylim()
    ax1.fill_betweenx([0, ylim[1]], pic50_cutoff, df_passed["pIC50"].max() + 0.5,
                       alpha=0.1, color=NATURE_COLORS[1])
    ax1.set_ylim(ylim)

    # Right: cumulative distribution
    sorted_pic50 = np.sort(df_passed["pIC50"].dropna())
    cumulative = np.arange(1, len(sorted_pic50) + 1) / len(sorted_pic50) * 100
    ax2.plot(sorted_pic50, cumulative, linewidth=2.5, color=NATURE_COLORS[3])
    ax2.axvline(x=pic50_cutoff, color=NATURE_COLORS[0], linewidth=2, linestyle="--",
                label=f"Cutoff (pIC₅₀={pic50_cutoff})")
    # Mark % above cutoff
    pct_active = (df_passed["pIC50"] >= pic50_cutoff).sum() / len(df_passed) * 100
    ax2.axhline(y=100 - pct_active, color=NATURE_COLORS[1], linewidth=1.5, linestyle=":",
                label=f"{pct_active:.1f}% active")
    ax2.set_xlabel("pIC₅₀", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cumulative Percentage (%)", fontsize=14, fontweight="bold")
    ax2.set_title(f"Cumulative pIC₅₀ Distribution\n{pct_active:.1f}% compounds are active", fontsize=15, fontweight="bold")
    ax2.legend(fontsize=12)
    ax2.set_ylim(0, 105)

    plt.suptitle(f"Bioactivity Profile — {gene}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"{gene}_pic50_distribution.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pIC50 distribution: {out}")


# ─── Main Stage 2 Function ────────────────────────────────────────────────────
def run_filtering(bioactivity_data: dict, force_rerun: bool = False) -> dict:
    """
    Stage 2: Apply Ro5, ADMET, PAINS, Brenk filters to all gene datasets.
    Returns dict of filtered DataFrames per gene.
    """
    from src.utils import Checkpoint

    filtered_data = {}

    for gene, df in bioactivity_data.items():
        ckpt = Checkpoint(f"s2_filtered_{gene}")
        if ckpt.exists() and not force_rerun:
            df_filtered = ckpt.load()
            logger.info(f"[{gene}] Loaded filtered data from checkpoint: {len(df_filtered)} → {df_filtered['passed_all_filters'].sum()} pass")
            filtered_data[gene] = df_filtered
            continue

        logger.info(f"\n{'='*60}\n[{gene}] Starting filtration pipeline (n={len(df)} compounds)")

        # Validate SMILES first
        df = df.copy()
        valid_mask = df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        df = df[valid_mask].reset_index(drop=True)
        logger.info(f"[{gene}] Valid SMILES: {len(df)}")

        n_initial = len(df)

        # Step 1: Properties
        df = apply_properties(df)
        n_props = len(df)

        # Step 2: Ro5
        df = apply_ro5(df)
        n_ro5 = df["ro5_pass"].sum()

        # Step 3: ADMET
        df = apply_admet(df)
        n_admet = df[df["ro5_pass"] & df["admet_pass"]].shape[0]

        # Step 4: PAINS
        df = apply_pains_filter(df)
        n_pains = df[df["ro5_pass"] & df["admet_pass"] & df["pains_pass"]].shape[0]

        # Step 5: Brenk
        df = apply_brenk_filter(df)

        # Combined
        df = apply_all_filters(df)
        n_final = df["passed_all_filters"].sum()

        filter_stats = {
            "Raw (valid SMILES)": n_initial,
            "After property calc.": n_props,
            "After Ro5": n_ro5,
            "After ADMET": n_admet,
            "After PAINS": n_pains,
            "Passed All Filters": n_final,
        }

        logger.info(f"[{gene}] Filter summary: {filter_stats}")

        # Save
        safe_save_csv(df, PROCESSED_DIR / f"{gene}_filtered.csv")
        ckpt.save(df)
        filtered_data[gene] = df

        # Visualizations
        try:
            df_pass = df[df["passed_all_filters"]]
            df_fail = df[~df["passed_all_filters"]]
            if len(df_pass) > 5:
                plot_ro5_radar(df_pass, df_fail, gene)
                plot_property_distributions(df, gene)
                plot_filtering_funnel(filter_stats, gene)
                plot_pIC50_distribution(df, gene)
        except Exception as e:
            logger.warning(f"[{gene}] Visualization failed: {e}")

    # Save combined
    if filtered_data:
        all_filtered = pd.concat(
            [df.assign(gene=g) for g, df in filtered_data.items()],
            ignore_index=True
        )
        safe_save_csv(all_filtered, PROCESSED_DIR / "all_filtered.csv")

    return filtered_data


if __name__ == "__main__":
    # Quick test with sample data
    import random
    sample = {
        "TRIM22": pd.DataFrame({
            "smiles": [
                "CC1=C(C(=O)Nc2ccccc2)C(c2ccccc2Cl)NC1=O",
                "COc1ccc(-c2nc(N3CCOCC3)c3ccccc3n2)cc1",
                "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1C",
            ],
            "pIC50": [7.2, 6.8, 5.9],
            "IC50_nM": [63.1, 158.5, 1259.0],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        })
    }
    result = run_filtering(sample)
    print("Done:", {g: df["passed_all_filters"].sum() for g, df in result.items()})
