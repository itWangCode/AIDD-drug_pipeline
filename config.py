"""
Drug Discovery Pipeline Configuration
Nature-grade computational drug discovery pipeline
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Input ────────────────────────────────────────────────────────────────────
HUB_GENES_FILE = BASE_DIR / "hubgenes.txt"

# ─── ChEMBL Query Parameters ──────────────────────────────────────────────────
CHEMBL_CONFIG = {
    "assay_type": "B",          # Binding assay
    "relation": "=",
    "standard_type": "IC50",
    "standard_units": "nM",
    "organism": "Homo sapiens",
    "max_retries": 10,
    "retry_delay": 5,           # seconds between retries
    "timeout": 30,
}

# ─── Lipinski Ro5 Thresholds ──────────────────────────────────────────────────
RO5_CONFIG = {
    "mw_max": 500,
    "hba_max": 10,
    "hbd_max": 5,
    "logp_max": 5,
    "violations_allowed": 1,    # allow up to 1 violation
}

# ─── ADMET Filter Thresholds ──────────────────────────────────────────────────
ADMET_CONFIG = {
    "qed_min": 0.4,
    "tpsa_max": 140,
    "rotatable_bonds_max": 10,
    "mw_min": 150,
    "mw_max": 600,
    "logp_min": -2,
    "logp_max": 6,
    "hbd_max": 5,
    "hba_max": 10,
    "rings_max": 6,
}

# ─── pIC50 Bioactivity Cutoff ─────────────────────────────────────────────────
PIC50_ACTIVE_CUTOFF = 6.3       # pIC50 ≥ 6.3 → active (IC50 ≤ 500 nM)
PIC50_INACTIVE_CUTOFF = 5.0     # pIC50 < 5.0 → inactive

# ─── Fingerprint Config ───────────────────────────────────────────────────────
FP_CONFIG = {
    "morgan_radius": 2,
    "morgan_nbits": 2048,
    "use_maccs": True,
    "use_morgan": True,
}

# ─── Machine Learning Config ──────────────────────────────────────────────────
ML_CONFIG = {
    "n_folds": 5,
    "test_size": 0.2,
    "random_state": 42,
    "n_jobs": -1,
}

# ─── Neural Network Config ────────────────────────────────────────────────────
NN_CONFIG = {
    "hidden_layers": [512, 256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15,
    "random_state": 42,
}

# ─── Top Compound Selection ───────────────────────────────────────────────────
TOP_N_COMPOUNDS = 40            # top compounds for docking
TOP_N_DOCKING = 10              # top docking results for PLIP analysis

# ─── PDB / Docking ───────────────────────────────────────────────────────────
PDB_CONFIG = {
    "resolution_cutoff": 3.0,
    "method": "X-RAY DIFFRACTION",
    "min_ligand_mw": 150,
    "n_chains": 1,
    "before_date": "2024-01-01T00:00:00Z",
}

DOCKING_CONFIG = {
    "exhaustiveness": 16,
    "num_modes": 9,
    "energy_range": 3,
    "buffer": 5.0,              # Å buffer around ligand
}

# ─── Plot Aesthetics ──────────────────────────────────────────────────────────
PLOT_CONFIG = {
    "dpi": 300,
    "font_size": 14,
    "title_size": 18,
    "label_size": 15,
    "tick_size": 12,
    "legend_size": 12,
    "fig_format": "pdf",        # also saves PNG
    "palette": "deep",
    "style": "whitegrid",
    "context": "paper",
}

# Nature journal color palette (accessible)
NATURE_COLORS = [
    "#E64B35",  # red
    "#4DBBD5",  # teal
    "#00A087",  # green
    "#3C5488",  # navy
    "#F39B7F",  # salmon
    "#8491B4",  # lavender
    "#91D1C2",  # mint
    "#DC0000",  # crimson
    "#7E6148",  # brown
    "#B09C85",  # beige
]
