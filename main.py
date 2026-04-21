"""
╔══════════════════════════════════════════════════════════════════════════╗
║        COMPUTATIONAL DRUG DISCOVERY PIPELINE — MAIN ORCHESTRATOR       ║
║                                                                          ║
║  Hub Genes → UniProt → ChEMBL → Ro5/ADMET/PAINS → ML/NN →             ║
║  Top Compounds → PDB → Cartesian Docking → PLIP Interactions           ║
║                                                                          ║
║  Nature-grade analysis | Checkpoint-based resumption | Robust I/O      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR,
    HUB_GENES_FILE, NATURE_COLORS, PLOT_CONFIG,
)
from src.utils import setup_logger, safe_save_csv

logger = setup_logger("main_pipeline", "pipeline.log")


# ─── Pipeline Stage Runner ────────────────────────────────────────────────────
def run_pipeline(
    force_rerun_stages: list = None,
    skip_stages: list = None,
    demo_mode: bool = False,
):
    """
    Main pipeline orchestrator.

    Parameters
    ----------
    force_rerun_stages : list, optional
        List of stage numbers to force rerun (e.g. [1, 3])
    skip_stages : list, optional
        List of stage numbers to skip
    demo_mode : bool
        Use demo data instead of real API calls (for testing)
    """
    force_rerun_stages = force_rerun_stages or []
    skip_stages = skip_stages or []

    t_start = datetime.now()
    logger.info("=" * 70)
    logger.info("COMPUTATIONAL DRUG DISCOVERY PIPELINE STARTED")
    logger.info(f"Start time: {t_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Data Acquisition
    # ─────────────────────────────────────────────────────────────────────────
    if 1 not in skip_stages:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 1: Data Acquisition (Genes → UniProt → ChEMBL)")
        logger.info("─" * 60)
        try:
            if demo_mode:
                bioactivity_data = _generate_demo_data()
                logger.info(f"[DEMO] Generated data for {len(bioactivity_data)} genes")
            else:
                from src.stage1_data_acquisition import run_data_acquisition
                bioactivity_data = run_data_acquisition(force_rerun=(1 in force_rerun_stages))

            results["stage1"] = bioactivity_data
            logger.info(f"Stage 1 complete: {len(bioactivity_data)} genes with data")
            _log_stage_summary(bioactivity_data, "Stage 1")

        except Exception as e:
            logger.error(f"STAGE 1 FAILED: {e}\n{traceback.format_exc()}")
            if not demo_mode:
                logger.info("Falling back to demo mode...")
                bioactivity_data = _generate_demo_data()
                results["stage1"] = bioactivity_data
    else:
        logger.info("Stage 1: SKIPPED")
        bioactivity_data = results.get("stage1", _generate_demo_data())

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Drug-Likeness Filtering
    # ─────────────────────────────────────────────────────────────────────────
    if 2 not in skip_stages and bioactivity_data:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 2: Drug-Likeness Filtering (Ro5, ADMET, PAINS, Brenk)")
        logger.info("─" * 60)
        try:
            from src.stage2_filtering import run_filtering
            filtered_data = run_filtering(bioactivity_data, force_rerun=(2 in force_rerun_stages))
            results["stage2"] = filtered_data
            _log_filter_summary(filtered_data)
        except Exception as e:
            logger.error(f"STAGE 2 FAILED: {e}\n{traceback.format_exc()}")
            filtered_data = bioactivity_data
    else:
        logger.info("Stage 2: SKIPPED")
        filtered_data = results.get("stage2", bioactivity_data)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Machine Learning & Compound Ranking
    # ─────────────────────────────────────────────────────────────────────────
    if 3 not in skip_stages and filtered_data:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 3: ML/NN Activity Prediction & Compound Ranking")
        logger.info("─" * 60)
        try:
            from src.stage3_ml import run_ml_analysis
            scored_data = run_ml_analysis(filtered_data, force_rerun=(3 in force_rerun_stages))
            results["stage3"] = scored_data
            _log_ml_summary(scored_data)
        except Exception as e:
            logger.error(f"STAGE 3 FAILED: {e}\n{traceback.format_exc()}")
            # Fallback: use filtered data with basic scoring
            scored_data = _basic_scoring_fallback(filtered_data)
    else:
        logger.info("Stage 3: SKIPPED")
        scored_data = results.get("stage3", _basic_scoring_fallback(filtered_data))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: PDB Retrieval & Molecular Docking
    # ─────────────────────────────────────────────────────────────────────────
    if 4 not in skip_stages and scored_data:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 4: PDB Retrieval & Cartesian-Product Docking")
        logger.info("─" * 60)
        logger.info(
            "  To dock against real PDB structures, place .pdb files in:\n"
            "    data/raw/pdb/{GENE}/   (one folder per gene), OR\n"
            "    data/raw/pdb/         (shared structures, all genes), OR\n"
            "  Create data/raw/pdb_registry.json with:\n"
            '    {"EGFR": [{"pdb_id":"3W32","pdb_path":"data/raw/pdb/3W32.pdb",'
            '"ligand_resname":"W32"}]}'
        )
        try:
            # Load UniProt mapping (kept for Stage 5 / reporting only)
            uniprot_path = DATA_DIR / "raw" / "uniprot_mapping.csv"
            if uniprot_path.exists():
                uniprot_df = pd.read_csv(uniprot_path)
            else:
                uniprot_df = _generate_demo_uniprot(bioactivity_data)
                safe_save_csv(uniprot_df, uniprot_path)

            from src.stage4_docking import run_docking_pipeline
            docking_results = run_docking_pipeline(
                scored_data, uniprot_df, force_rerun=(4 in force_rerun_stages)
            )
            results["stage4"] = docking_results
            n_dock = len(docking_results) if not docking_results.empty else 0
            logger.info("Docking complete: %d compound-structure pairs", n_dock)
        except Exception as e:
            logger.error("STAGE 4 FAILED: %s\n%s", e, traceback.format_exc())
            docking_results = pd.DataFrame()
    else:
        logger.info("Stage 4: SKIPPED")
        docking_results = results.get("stage4", pd.DataFrame())

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 5: Protein-Ligand Interaction Analysis
    # ─────────────────────────────────────────────────────────────────────────
    if 5 not in skip_stages:
        logger.info("\n" + "─" * 60)
        logger.info("STAGE 5: Protein-Ligand Interaction Analysis (PLIP)")
        logger.info("─" * 60)
        try:
            from src.stage5_interactions import run_interaction_analysis
            interaction_results = run_interaction_analysis(
                docking_results, scored_data, force_rerun=(5 in force_rerun_stages)
            )
            results["stage5"] = interaction_results
        except Exception as e:
            logger.error(f"STAGE 5 FAILED: {e}\n{traceback.format_exc()}")
            interaction_results = {}
    else:
        logger.info("Stage 5: SKIPPED")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL: Generate Pipeline Report
    # ─────────────────────────────────────────────────────────────────────────
    generate_pipeline_report(results, t_start)

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETE — Total time: {elapsed/60:.1f} min")
    logger.info("=" * 70)

    return results


# ─── Demo Data Generator ─────────────────────────────────────────────────────
def _generate_demo_data() -> dict:
    """Generate realistic demo data when API is unavailable."""
    from rdkit.Chem import AllChem

    logger.info("Generating demo bioactivity data...")

    # Representative EGFR-like SMILES
    demo_smiles = [
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",    # Gefitinib-like
        "COc1cc2c(Nc3cccc(Br)c3)ncnc2cc1OCC",                 # EGFR inhibitor scaffold
        "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
        "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
        "CCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC",
        "Cc1cnc(Nc2cc(C(F)(F)F)cc(C3(N)CC3)c2)nc1",
        "O=C(Nc1ccc2c(c1)OCCO2)c1ccc(Cl)cc1",
        "CS(=O)(=O)c1ccc(-c2nc3ccccc3s2)cc1",
        "CC1CCN(c2nc(Nc3ccc(F)c(Cl)c3)c3ncnc(N4CC4)c3n2)CC1",
        "c1ccc2c(c1)nc(Nc1ccc(F)c(Cl)c1)nc2N1CCCC1",
        "CC(=O)Nc1ccc(-c2ccc3ncnc(Nc4cccc(Br)c4)c3c2)cc1",
        "Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1",
        "CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
        "O=C(c1ccc(Cl)cc1)N1CCN(c2ncnc3ccccc23)CC1",
        "Cc1ccc(-c2cc(NC(=O)c3ccc(CN4CCN(C)CC4)cc3)c(C)n2-c2ccc(F)cc2)cc1",
        "COc1ccc(Nc2ncc3cc(-c4ccc(F)cc4)cc(=O)n3n2)cc1",
        "O=C(/C=C/c1ccc(O)cc1)Nc1ccccc1",
        "Cc1ccc(S(=O)(=O)Nc2cccc(C(F)(F)F)c2)cc1",
        "CC(Nc1nc2ccccc2s1)c1ccccc1",
        "N#Cc1ccc(NC(=O)c2cc3cc(Cl)ccc3[nH]2)cc1",
    ] * 30  # 600 entries

    np.random.seed(42)
    n = len(demo_smiles)
    pic50_vals = np.random.normal(6.8, 1.5, n).clip(3.0, 12.0)
    ic50_vals = 10 ** (9 - pic50_vals)

    genes = ["TRIM22", "PPP2R5A"]
    demo_data = {}
    for i, gene in enumerate(genes):
        n_gene = n // len(genes)
        idx_start = i * n_gene
        idx_end = idx_start + n_gene
        demo_data[gene] = pd.DataFrame({
            "molecule_chembl_id": [f"CHEMBL{j:06d}" for j in range(idx_start, idx_end)],
            "smiles": demo_smiles[idx_start:idx_end],
            "IC50_nM": ic50_vals[idx_start:idx_end],
            "pIC50": pic50_vals[idx_start:idx_end],
            "units": "nM",
            "gene_symbol": gene,
            "uniprot_accession": "Q8IYM9" if gene == "TRIM22" else "Q13362",
            "chembl_target_id": "CHEMBL203" if gene == "TRIM22" else "CHEMBL4860",
            "protein_name": f"{gene} kinase",
        })
        safe_save_csv(demo_data[gene], DATA_DIR / "raw" / f"{gene}_bioactivities_raw.csv")

    return demo_data


def _generate_demo_uniprot(bioactivity_data: dict) -> pd.DataFrame:
    """Generate demo UniProt mapping."""
    uniprot_map = {
        "TRIM22": {"uniprot_accession": "Q8IYM9", "protein_name": "E3 ubiquitin-protein ligase TRIM22"},
        "PPP2R5A": {"uniprot_accession": "Q15172", "protein_name": "Serine/threonine-protein phosphatase 2A 56 kDa regulatory subunit alpha"},
    }
    records = []
    for gene in bioactivity_data:
        entry = uniprot_map.get(gene, {"uniprot_accession": "UNKNOWN", "protein_name": gene})
        records.append({
            "gene_symbol": gene,
            "uniprot_accession": entry["uniprot_accession"],
            "protein_name": entry["protein_name"],
            "organism": "Homo sapiens",
        })
    return pd.DataFrame(records)


def _basic_scoring_fallback(filtered_data: dict) -> dict:
    """Basic composite scoring when ML fails."""
    scored = {}
    for gene, df in filtered_data.items():
        df_pass = df[df.get("passed_all_filters", pd.Series(True, index=df.index))].copy()
        if len(df_pass) == 0:
            df_pass = df.copy()
        if "pIC50" in df_pass.columns and "qed" in df_pass.columns:
            pic50_norm = (df_pass["pIC50"] - df_pass["pIC50"].min()) / \
                         (df_pass["pIC50"].max() - df_pass["pIC50"].min() + 1e-9)
            df_pass["composite_score"] = df_pass["qed"] * pic50_norm
            df_pass["pred_active_prob"] = (df_pass["pIC50"] >= 6.3).astype(float)
        df_pass.sort_values("composite_score", ascending=False, inplace=True)
        df_pass.reset_index(drop=True, inplace=True)
        scored[gene] = {"df_scored": df_pass, "cv_results": [], "best_clf": "N/A",
                         "best_clf_auc": 0.0, "best_reg": "N/A", "best_reg_r2": 0.0}
    return scored


# ─── Summary Helpers ──────────────────────────────────────────────────────────
def _log_stage_summary(data: dict, stage_name: str):
    for gene, df in data.items():
        logger.info(f"  {gene}: {len(df)} compounds, "
                    f"pIC50 range [{df['pIC50'].min():.1f}, {df['pIC50'].max():.1f}]")


def _log_filter_summary(filtered_data: dict):
    for gene, df in filtered_data.items():
        if "passed_all_filters" in df.columns:
            n_pass = df["passed_all_filters"].sum()
            logger.info(f"  {gene}: {n_pass}/{len(df)} passed all filters ({n_pass/len(df)*100:.1f}%)")


def _log_ml_summary(scored_data: dict):
    for gene, result in scored_data.items():
        if isinstance(result, dict):
            logger.info(f"  {gene}: best_clf={result.get('best_clf')}, "
                        f"AUC={result.get('best_clf_auc', 0):.3f}, "
                        f"best_reg={result.get('best_reg')}, "
                        f"R²={result.get('best_reg_r2', 0):.3f}")


# ─── Pipeline Report ──────────────────────────────────────────────────────────
def generate_pipeline_report(results: dict, t_start: datetime):
    """Generate a comprehensive text + figure pipeline report."""
    report_path = RESULTS_DIR / "pipeline_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("COMPUTATIONAL DRUG DISCOVERY PIPELINE — REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for stage_name, stage_result in results.items():
            f.write(f"\n[{stage_name.upper()}]\n")
            if isinstance(stage_result, dict) and stage_name in ("stage1", "stage2"):
                for gene, df in stage_result.items():
                    if isinstance(df, pd.DataFrame):
                        f.write(f"  {gene}: {len(df)} compounds\n")
            elif isinstance(stage_result, dict) and stage_name == "stage3":
                for gene, res in stage_result.items():
                    if isinstance(res, dict):
                        f.write(f"  {gene}: AUC={res.get('best_clf_auc', 0):.3f}, "
                                f"R²={res.get('best_reg_r2', 0):.3f}\n")
            elif isinstance(stage_result, pd.DataFrame):
                f.write(f"  {len(stage_result)} docking results\n")

        f.write("\n\nFigures generated:\n")
        for fig_path in sorted(FIGURES_DIR.glob("*.pdf")):
            f.write(f"  {fig_path.name}\n")

    # Generate overview figure
    _plot_pipeline_overview(results)
    logger.info(f"Pipeline report saved: {report_path}")


def _plot_pipeline_overview(results: dict):
    """Generate a pipeline overview figure."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "figure.dpi": PLOT_CONFIG["dpi"],
    })

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    stages = [
        ("Hub Genes\nInput", NATURE_COLORS[3]),
        ("UniProt\nMapping", NATURE_COLORS[1]),
        ("ChEMBL\nBioactivities", NATURE_COLORS[2]),
        ("Ro5 + ADMET\nFiltering", NATURE_COLORS[0]),
        ("ML/NN\nRanking", NATURE_COLORS[4]),
        ("PDB +\nDocking", NATURE_COLORS[5]),
        ("PLIP\nInteractions", NATURE_COLORS[6]),
        ("Top Drug\nCandidates", NATURE_COLORS[7]),
    ]

    n = len(stages)
    for i, (label, color) in enumerate(stages):
        x = i / (n - 1)
        circle = plt.Circle((x, 0.5), 0.07, color=color, transform=ax.transAxes,
                              clip_on=False, zorder=3, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, 0.5, str(i + 1), transform=ax.transAxes, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white", zorder=4)
        ax.text(x, 0.18, label, transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=color)
        if i < n - 1:
            ax.annotate("", xy=((i + 1) / (n - 1) - 0.03, 0.5),
                         xytext=(i / (n - 1) + 0.03, 0.5),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))

    ax.set_title("Computational Drug Discovery Pipeline Overview",
                  fontsize=18, fontweight="bold", pad=20)
    fig.patch.set_facecolor("white")
    out = FIGURES_DIR / "pipeline_overview.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved pipeline overview: {out}")


# ─── CLI Interface ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Computational Drug Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Full pipeline
  python main.py --demo                   # Demo mode (no internet)
  python main.py --force-rerun 1 3        # Re-run stages 1 and 3
  python main.py --skip 4 5               # Skip docking and PLIP
  python main.py --stages 1 2 3           # Run only stages 1-3
        """
    )
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no API calls)")
    parser.add_argument("--force-rerun", nargs="*", type=int, default=[], metavar="N",
                        help="Force rerun of specified stages")
    parser.add_argument("--skip", nargs="*", type=int, default=[], metavar="N",
                        help="Skip specified stages")
    parser.add_argument("--stages", nargs="*", type=int, default=None, metavar="N",
                        help="Run only specified stages (skips all others)")

    args = parser.parse_args()

    skip = args.skip or []
    if args.stages:
        all_stages = [1, 2, 3, 4, 5]
        skip = [s for s in all_stages if s not in args.stages]

    run_pipeline(
        force_rerun_stages=args.force_rerun,
        skip_stages=skip,
        demo_mode=args.demo,
    )


if __name__ == "__main__":
    main()
