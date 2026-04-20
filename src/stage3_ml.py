"""
Stage 3: Molecular Fingerprinting + Machine Learning + Neural Network
Activity classification and pIC50 regression with Nature-grade visualizations.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, r2_score, mean_absolute_error,
)
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR,
    ML_CONFIG, NN_CONFIG, FP_CONFIG, PIC50_ACTIVE_CUTOFF,
    PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv, Checkpoint

logger = setup_logger("stage3_ml", "stage3.log")
warnings.filterwarnings("ignore")


# ─── Fingerprint Generation ───────────────────────────────────────────────────
def smiles_to_fingerprints(smiles: str, method: str = "morgan") -> np.ndarray:
    """Generate molecular fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if method == "maccs":
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        elif method == "morgan":
            fpg = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_CONFIG["morgan_radius"],
                fpSize=FP_CONFIG["morgan_nbits"]
            )
            return np.array(fpg.GetFingerprint(mol))
        elif method == "morgan_count":
            fpg = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_CONFIG["morgan_radius"],
                fpSize=FP_CONFIG["morgan_nbits"]
            )
            return np.array(fpg.GetCountFingerprint(mol).ToList())
        elif method == "combined":
            fpg = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_CONFIG["morgan_radius"],
                fpSize=FP_CONFIG["morgan_nbits"]
            )
            morgan = np.array(fpg.GetFingerprint(mol))
            maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
            return np.concatenate([morgan, maccs])
        else:
            raise ValueError(f"Unknown fingerprint method: {method}")
    except Exception as e:
        logger.warning(f"Fingerprint failed for SMILES: {e}")
        return None


def generate_fingerprint_matrix(df: pd.DataFrame, method: str = "morgan") -> tuple:
    """Generate fingerprint matrix X and labels y from DataFrame."""
    fps = []
    valid_idx = []

    for i, smiles in enumerate(df["smiles"]):
        fp = smiles_to_fingerprints(str(smiles), method)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)

    if not fps:
        return np.array([]), pd.DataFrame()

    X = np.array(fps)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    logger.info(f"Generated {method} fingerprints: {X.shape} for {len(df_valid)} compounds")
    return X, df_valid


# ─── Label Assignment ─────────────────────────────────────────────────────────
def assign_activity_labels(df: pd.DataFrame, cutoff: float = PIC50_ACTIVE_CUTOFF) -> pd.DataFrame:
    """Assign binary activity labels based on pIC50 cutoff."""
    df = df.copy()
    df["active"] = (df["pIC50"] >= cutoff).astype(int)
    n_active = df["active"].sum()
    n_inactive = len(df) - n_active
    logger.info(f"Activity labels: {n_active} active, {n_inactive} inactive (cutoff pIC50={cutoff})")
    return df


# ─── Model Definitions ────────────────────────────────────────────────────────
def get_classifiers():
    """Return dict of sklearn classifiers."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            class_weight="balanced", random_state=ML_CONFIG["random_state"],
            n_jobs=ML_CONFIG["n_jobs"],
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1, use_label_encoder=False,
            eval_metric="logloss", random_state=ML_CONFIG["random_state"],
            verbosity=0,
        ),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale",
                        probability=True, random_state=ML_CONFIG["random_state"])),
        ]),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation="relu", solver="adam",
                alpha=1e-4, batch_size=64, learning_rate="adaptive",
                max_iter=200, random_state=ML_CONFIG["random_state"],
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
            )),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=ML_CONFIG["random_state"],
        ),
    }


def get_regressors():
    """Return dict of sklearn regressors for pIC50 prediction."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=ML_CONFIG["random_state"],
            n_jobs=ML_CONFIG["n_jobs"],
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=ML_CONFIG["random_state"], verbosity=0,
        ),
        "Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation="relu", solver="adam",
                alpha=1e-4, max_iter=200,
                random_state=ML_CONFIG["random_state"],
                early_stopping=True, n_iter_no_change=15,
            )),
        ]),
    }


# ─── Cross-Validation ─────────────────────────────────────────────────────────
def cross_validate_classifier(clf, X, y, gene: str, model_name: str) -> dict:
    """5-fold CV for classification with detailed metrics."""
    kf = KFold(n_splits=ML_CONFIG["n_folds"], shuffle=True, random_state=ML_CONFIG["random_state"])

    metrics = {"accuracy": [], "sensitivity": [], "specificity": [], "auc": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf_fold = _clone_model(clf)
        clf_fold.fit(X_train, y_train)
        y_pred = clf_fold.predict(X_test)
        y_prob = clf_fold.predict_proba(X_test)[:, 1]

        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["sensitivity"].append(recall_score(y_test, y_pred, zero_division=0))
        metrics["specificity"].append(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
        metrics["auc"].append(roc_auc_score(y_test, y_prob))

    summary = {
        "gene": gene,
        "model": model_name,
        **{f"mean_{k}": np.mean(v) for k, v in metrics.items()},
        **{f"std_{k}": np.std(v) for k, v in metrics.items()},
    }
    logger.info(f"[{gene}] {model_name}: AUC={summary['mean_auc']:.3f}±{summary['std_auc']:.3f}, "
                f"Sens={summary['mean_sensitivity']:.3f}, Spec={summary['mean_specificity']:.3f}")
    return summary


def _clone_model(clf):
    """Clone sklearn model."""
    from sklearn.base import clone
    return clone(clf)


# ─── Plots ────────────────────────────────────────────────────────────────────
def set_nature_style():
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
    })


def plot_roc_curves(models_test: dict, gene: str):
    """Multi-model ROC curves — Nature style."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (model_name, (y_test, y_prob)) in enumerate(models_test.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, linewidth=2.5, color=NATURE_COLORS[i % len(NATURE_COLORS)],
                label=f"{model_name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7, label="Random (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14, fontweight="bold")
    ax.set_title(f"ROC Curves — Activity Classification\n{gene}", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    out = FIGURES_DIR / f"{gene}_roc_curves.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curves: {out}")


def plot_cv_performance(cv_results: list, gene: str):
    """Cross-validation performance comparison — Nature style."""
    set_nature_style()
    df_cv = pd.DataFrame(cv_results)

    metrics = ["mean_accuracy", "mean_sensitivity", "mean_specificity", "mean_auc"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "AUC-ROC"]
    std_metrics = ["std_accuracy", "std_sensitivity", "std_specificity", "std_auc"]

    n_models = len(df_cv)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (metric, label, std_col) in enumerate(zip(metrics, metric_labels, std_metrics)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, df_cv[metric], width, yerr=df_cv[std_col],
                      label=label, color=NATURE_COLORS[i], alpha=0.85,
                      capsize=5, error_kw={"linewidth": 1.5, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(df_cv["model"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_title(f"5-Fold Cross-Validation Performance — {gene}", fontsize=16, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=11, framealpha=0.9)

    # Value labels on bars
    for bars_grp in ax.containers:
        ax.bar_label(bars_grp, fmt="%.2f", fontsize=9, padding=2)

    plt.tight_layout()
    out = FIGURES_DIR / f"{gene}_cv_performance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved CV performance: {out}")


def plot_regression_scatter(y_true, y_pred, gene: str, model_name: str):
    """Predicted vs actual pIC50 scatter plot — Nature style."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, p_val = stats.pearsonr(y_true, y_pred)

    # Scatter
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=40,
                         c=y_true, cmap="coolwarm", edgecolors="white",
                         linewidths=0.5, vmin=min(y_true), vmax=max(y_true))
    plt.colorbar(scatter, ax=ax, label="True pIC₅₀", shrink=0.8)

    # Perfect prediction line
    lims = [min(min(y_true), min(y_pred)) - 0.5, max(max(y_true), max(y_pred)) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=2, label="Perfect prediction", alpha=0.8)

    # Regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, m * x_line + b, "-", linewidth=2, color=NATURE_COLORS[0],
            alpha=0.8, label=f"Regression line")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("True pIC₅₀", fontsize=14, fontweight="bold")
    ax.set_ylabel("Predicted pIC₅₀", fontsize=14, fontweight="bold")
    ax.set_title(f"pIC₅₀ Regression — {model_name}\n{gene}", fontsize=16, fontweight="bold")

    stats_text = (f"R² = {r2:.3f}\nMAE = {mae:.3f}\nPearson r = {pearson_r:.3f} (p={p_val:.2e})")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=13,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    ax.legend(fontsize=12, loc="lower right")
    ax.set_aspect("equal")

    out = FIGURES_DIR / f"{gene}_{model_name.replace(' ', '_')}_regression.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved regression scatter: {out}")


def plot_feature_importance(clf, gene: str, method: str = "morgan", top_n: int = 30):
    """Feature importance from Random Forest — Nature style."""
    set_nature_style()
    model = clf
    # Unwrap pipeline if needed
    if hasattr(clf, "named_steps"):
        for step in clf.named_steps.values():
            if hasattr(step, "feature_importances_"):
                model = step
                break
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[indices]
    top_labels = [f"Bit {i}" for i in indices]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    bars = ax.barh(range(top_n), top_imp[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=14, fontweight="bold")
    ax.set_title(f"Top {top_n} {method.upper()} Fingerprint Bits — {gene}\n(Random Forest)", fontsize=15, fontweight="bold")
    ax.invert_yaxis()

    out = FIGURES_DIR / f"{gene}_feature_importance.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance: {out}")


def plot_qed_pic50_landscape(df_scored: pd.DataFrame, gene: str):
    """QED vs pIC50 scatter with compound ranking — Nature style."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df_valid = df_scored.dropna(subset=["qed", "pIC50"])

    # Left: scatter plot
    ax = axes[0]
    sc = ax.scatter(df_valid["pIC50"], df_valid["qed"], alpha=0.65, s=50,
                    c=df_valid["composite_score"] if "composite_score" in df_valid else df_valid["pIC50"],
                    cmap="RdYlGn", edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Composite Score", shrink=0.85)

    ax.axvline(x=PIC50_ACTIVE_CUTOFF, color=NATURE_COLORS[0], linewidth=2, linestyle="--", alpha=0.8, label=f"Active (pIC₅₀≥{PIC50_ACTIVE_CUTOFF})")
    ax.axhline(y=0.5, color=NATURE_COLORS[3], linewidth=2, linestyle="--", alpha=0.8, label="QED threshold (0.5)")
    ax.fill_between([PIC50_ACTIVE_CUTOFF, df_valid["pIC50"].max() + 0.5], 0.5, 1.0, alpha=0.08,
                     color=NATURE_COLORS[1], label="Optimal region")

    # Label top compounds
    if "composite_score" in df_valid:
        top10 = df_valid.nlargest(10, "composite_score")
        for _, row in top10.iterrows():
            ax.annotate(row.get("molecule_chembl_id", "")[:10],
                         (row["pIC50"], row["qed"]),
                         fontsize=7, ha="left", va="bottom",
                         xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("pIC₅₀", fontsize=14, fontweight="bold")
    ax.set_ylabel("QED Score", fontsize=14, fontweight="bold")
    ax.set_title(f"Chemical Space: QED vs pIC₅₀\n{gene}", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")

    # Right: top compounds ranked by composite score
    ax2 = axes[1]
    if "composite_score" in df_valid:
        top_n = min(20, len(df_valid))
        top_df = df_valid.nlargest(top_n, "composite_score").reset_index(drop=True)
        colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
        ax2.barh(range(top_n), top_df["composite_score"], color=colors_bar, edgecolor="white")
        ax2.set_yticks(range(top_n))
        ids = top_df.get("molecule_chembl_id", pd.Series(range(top_n))).tolist()
        ax2.set_yticklabels([str(i)[:15] for i in ids], fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("Composite Score (QED × Normalized pIC₅₀)", fontsize=13, fontweight="bold")
        ax2.set_title(f"Top {top_n} Compounds by Composite Score", fontsize=14, fontweight="bold")

    plt.suptitle(f"Compound Ranking Landscape — {gene}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / f"{gene}_qed_pic50_landscape.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved QED-pIC50 landscape: {out}")


# ─── Main Stage 3 Function ────────────────────────────────────────────────────
def run_ml_analysis(filtered_data: dict, force_rerun: bool = False) -> dict:
    """
    Stage 3: Fingerprint → ML classification + regression → compound ranking.
    Returns dict of scored DataFrames per gene.
    """
    scored_data = {}

    for gene, df_all in filtered_data.items():
        ckpt = Checkpoint(f"s3_ml_{gene}")
        if ckpt.exists() and not force_rerun:
            result = ckpt.load()
            logger.info(f"[{gene}] Loaded ML results from checkpoint")
            scored_data[gene] = result
            continue

        df = df_all[df_all["passed_all_filters"]].copy().reset_index(drop=True)
        logger.info(f"\n{'='*60}\n[{gene}] ML analysis: {len(df)} compounds")

        if len(df) < 30:
            logger.warning(f"[{gene}] Too few compounds ({len(df)}) for ML. Skipping.")
            continue

        # Generate fingerprints
        X, df_valid = generate_fingerprint_matrix(df, method="morgan")
        if X.shape[0] < 20:
            logger.warning(f"[{gene}] Insufficient valid fingerprints. Skipping.")
            continue

        # Activity labels
        df_valid = assign_activity_labels(df_valid)
        y_class = df_valid["active"].values
        y_reg = df_valid["pIC50"].values

        # Train-test split
        X_train, X_test, y_train_c, y_test_c, y_train_r, y_test_r, idx_train, idx_test = \
            train_test_split(X, y_class, y_reg, np.arange(len(df_valid)),
                             test_size=ML_CONFIG["test_size"],
                             random_state=ML_CONFIG["random_state"],
                             stratify=y_class if y_class.sum() >= 2 else None)

        # ── Classification ──
        classifiers = get_classifiers()
        cv_results = []
        models_roc = {}
        best_clf = None
        best_auc = 0.0

        for model_name, clf in classifiers.items():
            logger.info(f"[{gene}] Training classifier: {model_name}")
            try:
                # CV
                cv_summary = cross_validate_classifier(clf, X_train, y_train_c, gene, model_name)
                cv_results.append(cv_summary)

                # Final fit on train, evaluate on test
                clf.fit(X_train, y_train_c)
                y_prob = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test_c, y_prob)
                models_roc[model_name] = (y_test_c, y_prob)

                if auc > best_auc:
                    best_auc = auc
                    best_clf = (model_name, clf)

            except Exception as e:
                logger.error(f"[{gene}] {model_name} classifier failed: {e}")

        # ── Regression ──
        regressors = get_regressors()
        best_reg = None
        best_r2 = -999.0

        for model_name, reg in regressors.items():
            logger.info(f"[{gene}] Training regressor: {model_name}")
            try:
                reg.fit(X_train, y_train_r)
                y_pred = reg.predict(X_test)
                r2 = r2_score(y_test_r, y_pred)
                mae = mean_absolute_error(y_test_r, y_pred)
                logger.info(f"[{gene}] {model_name} Regressor: R²={r2:.3f}, MAE={mae:.3f}")
                plot_regression_scatter(y_test_r, y_pred, gene, model_name)
                if r2 > best_r2:
                    best_r2 = r2
                    best_reg = (model_name, reg)
            except Exception as e:
                logger.error(f"[{gene}] {model_name} regressor failed: {e}")

        # ── Score all filtered compounds ──
        df_valid["pred_active_prob"] = np.nan
        df_valid["pred_pIC50"] = np.nan

        if best_clf:
            _, clf = best_clf
            df_valid["pred_active_prob"] = clf.predict_proba(X)[:, 1]
            logger.info(f"[{gene}] Best classifier: {best_clf[0]} (test AUC={best_auc:.3f})")

        if best_reg:
            _, reg = best_reg
            df_valid["pred_pIC50"] = reg.predict(X)
            logger.info(f"[{gene}] Best regressor: {best_reg[0]} (test R²={best_r2:.3f})")

        # ── Composite Score: QED × normalized_pIC50 × active_prob ──
        pic50_min = df_valid["pIC50"].min()
        pic50_max = df_valid["pIC50"].max()
        pic50_norm = (df_valid["pIC50"] - pic50_min) / (pic50_max - pic50_min + 1e-9)

        df_valid["composite_score"] = (
            df_valid.get("qed", pd.Series(0.5, index=df_valid.index)).fillna(0.5)
            * pic50_norm
            * df_valid["pred_active_prob"].fillna(0.5)
        )

        df_valid.sort_values("composite_score", ascending=False, inplace=True)
        df_valid.reset_index(drop=True, inplace=True)

        # Save
        safe_save_csv(df_valid, RESULTS_DIR / f"{gene}_ml_scored.csv")

        # Plots
        try:
            if models_roc:
                plot_roc_curves(models_roc, gene)
            if cv_results:
                plot_cv_performance(cv_results, gene)
            if best_clf and hasattr(best_clf[1], "feature_importances_") or \
                    (hasattr(best_clf[1], "named_steps") if best_clf else False):
                plot_feature_importance(best_clf[1] if best_clf else None, gene)
            plot_qed_pic50_landscape(df_valid, gene)
        except Exception as e:
            logger.warning(f"[{gene}] Some plots failed: {e}")

        result = {
            "df_scored": df_valid,
            "cv_results": cv_results,
            "best_clf": best_clf[0] if best_clf else None,
            "best_clf_auc": best_auc,
            "best_reg": best_reg[0] if best_reg else None,
            "best_reg_r2": best_r2,
        }
        ckpt.save(result)
        scored_data[gene] = result

    return scored_data


if __name__ == "__main__":
    # Quick test
    from rdkit.Chem import AllChem
    smiles_list = [
        "CC1=C(C(=O)Nc2ccccc2)C(c2ccccc2Cl)NC1=O",
        "COc1ccc(-c2nc(N3CCOCC3)c3ccccc3n2)cc1",
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1C",
        "O=C(Nc1ccc(N2CCOCC2)cc1)c1ccc(F)cc1",
        "Cc1cnc(NC2CC(c3ccc(Cl)cc3)=NO2)nc1",
    ] * 20  # repeat for enough data

    sample_data = {
        "TRIM22": pd.DataFrame({
            "smiles": smiles_list,
            "pIC50": np.random.uniform(4, 10, len(smiles_list)),
            "IC50_nM": np.random.uniform(1, 10000, len(smiles_list)),
            "qed": np.random.uniform(0.3, 0.9, len(smiles_list)),
            "mw": np.random.uniform(200, 500, len(smiles_list)),
            "logp": np.random.uniform(0, 5, len(smiles_list)),
            "passed_all_filters": [True] * len(smiles_list),
            "molecule_chembl_id": [f"CHEMBL{i}" for i in range(len(smiles_list))],
        })
    }
    result = run_ml_analysis(sample_data)
    for gene, r in result.items():
        print(f"{gene}: best_clf={r['best_clf']}, AUC={r['best_clf_auc']:.3f}")
