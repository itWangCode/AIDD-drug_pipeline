# Computational Drug Discovery Pipeline

**Nature-grade | Checkpoint-based | Robust network handling**

## Pipeline Overview

```
Hub Genes → UniProt → ChEMBL → Ro5+ADMET+PAINS → ML/NN → 
PDB Retrieval → Cartesian-Product Docking → PLIP Interactions
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
conda install -c conda-forge openbabel   # for docking prep
# Install smina: https://github.com/mwojcikowski/smina
```

### 2. Set your hub genes
```bash
echo "TRIM22, PPP2R5A" > hubgenes.txt
# Or one per line:
echo -e "TRIM22\nPPP2R5A\nTP53\nEGFR" > hubgenes.txt
```

### 3. Run full pipeline
```bash
python main.py                    # Full pipeline (real API)
python main.py --demo             # Demo mode (no internet required)
python main.py --skip 4 5         # Skip docking + PLIP
python main.py --force-rerun 3    # Re-run ML stage only
```

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Gene→UniProt→ChEMBL→Bioactivities | `data/raw/` |
| 2 | Ro5 + ADMET + PAINS + Brenk filter | `data/processed/` |
| 3 | Morgan FP + RF/XGB/SVM/NN/GBM + Ranking | `data/results/*_ml_scored.csv` |
| 4 | PDB retrieval + Cartesian-product docking | `data/results/docking/` |
| 5 | PLIP protein-ligand interaction analysis | `data/results/interactions/` |

## Key Features

- **Checkpoint-based resumption**: Each stage saves to disk; resume after interruption
- **Exponential-backoff retry**: All API calls retry up to 8× with backoff
- **Multi-target × multi-ligand docking**: Cartesian product (N targets × M ligands)
- **5 ML classifiers** + 3 regressors with 5-fold CV
- **Composite ranking**: QED × normalized pIC₅₀ × predicted activity probability
- **25 publication-quality figures** (PDF + PNG at 300 DPI)

## Output Figures

| Figure | Description |
|--------|-------------|
| `pipeline_architecture.pdf` | Full pipeline schematic |
| `{GENE}_ro5_radar.pdf` | Lipinski Ro5 radar chart |
| `{GENE}_filtering_funnel.pdf` | Compound attrition waterfall |
| `{GENE}_property_distributions.pdf` | 8 molecular properties |
| `{GENE}_pic50_distribution.pdf` | Bioactivity histogram + CDF |
| `{GENE}_cv_performance.pdf` | 5-fold CV bar chart |
| `{GENE}_roc_curves.pdf` | Multi-model ROC curves |
| `{GENE}_*_regression.pdf` | pIC₅₀ scatter (predicted vs actual) |
| `{GENE}_chemical_space.pdf` | QED vs pIC₅₀ colored by LogP |
| `{GENE}_docking_heatmap.pdf` | Docking score matrix (after Stage 4) |
| `top_candidates_ranking.pdf` | Final ranked candidates |
| `ALL_FIGURES_COMBINED.pdf` | All figures in one PDF |

## External Software

For **docking** (Stage 4), install:
- **Smina**: `https://github.com/mwojcikowski/smina/releases`
- **OpenBabel**: `conda install -c conda-forge openbabel`

For **interaction analysis** (Stage 5):
- **PLIP**: `pip install plip`

## Configuration

Edit `config.py` to adjust:
- `PIC50_ACTIVE_CUTOFF` — activity threshold (default 6.3)
- `TOP_N_COMPOUNDS` — compounds for docking (default 40)
- `RO5_CONFIG` — Lipinski filter parameters
- `ADMET_CONFIG` — ADMET thresholds
- `ML_CONFIG` — cross-validation folds, train/test split
