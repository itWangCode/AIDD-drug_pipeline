"""
Stage 3 — ML Activity Prediction + SHAP Interpretability + External Validation
================================================================================
Python 3.8 compatible. Nature-grade scientific pipeline.

NEW vs original stage3_ml.py:
  SHAP Analysis     : TreeExplainer (RF/XGB), KernelExplainer (SVM/NN)
                      global importance bar, beeswarm, waterfall, reliability score
  External Validation: Murcko scaffold split, y-scrambling (n=100), AD z-score,
                       BEDROC, EF1%/EF5%/EF10%
  Enhanced Ranking  : composite_score includes SHAP_reliability (0.15 weight)
                      AD-filtered top candidates saved as separate CSV

Install: pip install shap
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, recall_score,
    r2_score, mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import xgboost as xgb

try:
    import shap
    # ── NumPy compatibility patch ─────────────────────────────────────────────
    # NumPy 1.20 deprecated np.bool/np.int/np.float/np.complex aliases.
    # NumPy 2.x still has them as attributes BUT accessing them raises a
    # DeprecationWarning/AttributeError at runtime inside SHAP.
    # Fix: ALWAYS force-overwrite these attributes with the Python builtins,
    # regardless of whether they already exist (do NOT use hasattr check).
    # This must run AFTER 'import shap' so shap's submodules are loaded and
    # share the same numpy module object; the patch then applies to all of them.
    import numpy as _np_patch
    _np_patch.bool    = bool      # force overwrite — no hasattr check
    _np_patch.int     = int
    _np_patch.float   = float
    _np_patch.complex = complex
    _np_patch.object  = object
    _np_patch.str     = str
    del _np_patch
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR,
    ML_CONFIG, FP_CONFIG, PIC50_ACTIVE_CUTOFF,
    PLOT_CONFIG, NATURE_COLORS,
)
from src.utils import setup_logger, safe_save_csv, Checkpoint

logger = setup_logger("stage3_ml", "stage3.log")
warnings.filterwarnings("ignore")

N_SHAP_BACKGROUND = 100
N_SHAP_EXPLAIN    = 200
N_SHAP_TOP_BITS   = 30
N_PERMUTATIONS    = 100
AD_PERCENTILE     = 95


# ══════════════════════════════════════════════════════════════════════════════
# Fingerprints
# ══════════════════════════════════════════════════════════════════════════════

def smiles_to_fingerprints(smiles, method="morgan"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if method == "maccs":
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        elif method == "morgan":
            fpg = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_CONFIG["morgan_radius"], fpSize=FP_CONFIG["morgan_nbits"])
            return np.array(fpg.GetFingerprint(mol))
        elif method == "combined":
            fpg = rdFingerprintGenerator.GetMorganGenerator(
                radius=FP_CONFIG["morgan_radius"], fpSize=FP_CONFIG["morgan_nbits"])
            return np.concatenate([np.array(fpg.GetFingerprint(mol)),
                                    np.array(MACCSkeys.GenMACCSKeys(mol))])
        else:
            raise ValueError("Unknown method: {}".format(method))
    except Exception as e:
        logger.warning("FP failed: %s", e)
        return None


def generate_fingerprint_matrix(df, method="morgan"):
    fps, valid_idx = [], []
    for i, smiles in enumerate(df["smiles"]):
        fp = smiles_to_fingerprints(str(smiles), method)
        if fp is not None:
            fps.append(fp); valid_idx.append(i)
    if not fps:
        return np.array([]), pd.DataFrame()
    X = np.array(fps, dtype=np.float32)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    logger.info("Fingerprints %s: %s for %d compounds", method, X.shape, len(df_valid))
    return X, df_valid


def bit_labels(n_bits, prefix="Bit"):
    return ["{}_{}".format(prefix, i) for i in range(n_bits)]


def assign_activity_labels(df, cutoff=PIC50_ACTIVE_CUTOFF):
    df = df.copy()
    df["active"] = (df["pIC50"] >= cutoff).astype(int)
    logger.info("Labels: %d active, %d inactive (cutoff=%.1f)",
                df["active"].sum(), (df["active"]==0).sum(), cutoff)
    return df


def _clone_model(clf):
    return clone(clf)


# ══════════════════════════════════════════════════════════════════════════════
# Scaffold split + Applicability Domain + y-Scrambling + Enrichment
# ══════════════════════════════════════════════════════════════════════════════

def _murcko(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return ""
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc else ""
    except Exception:
        return ""


def scaffold_split(df, test_frac=0.15, seed=42):
    """Murcko scaffold-based split: test set has different scaffolds than train."""
    rng = np.random.default_rng(seed)
    scaffolds = df["smiles"].apply(_murcko)
    groups = {}
    for idx, sc in scaffolds.items():
        groups.setdefault(sc, []).append(idx)
    scaffold_list = sorted(groups.values(), key=len, reverse=True)
    rng.shuffle(scaffold_list)
    n_test = int(len(df) * test_frac)
    test_idx = []
    for g in scaffold_list:
        if len(test_idx) >= n_test: break
        test_idx.extend(g)
    test_set = set(test_idx)
    train_idx = [i for i in range(len(df)) if i not in test_set]
    logger.info("Scaffold split: %d train, %d test", len(train_idx), len(test_idx))
    return np.array(train_idx), np.array(test_idx)


class ApplicabilityDomain:
    """z-score AD: inside if mean |z| <= 95th percentile of training set."""
    def __init__(self, percentile=AD_PERCENTILE):
        self.percentile = percentile
        self.mean_ = self.std_ = self.threshold_ = None

    def fit(self, X_train):
        self.mean_ = X_train.mean(axis=0)
        self.std_  = X_train.std(axis=0) + 1e-9
        z = np.abs((X_train - self.mean_) / self.std_).mean(axis=1)
        self.threshold_ = np.percentile(z, self.percentile)
        logger.info("AD threshold (z-score p%d): %.4f", self.percentile, self.threshold_)
        return self

    def predict(self, X):
        z = np.abs((X - self.mean_) / self.std_).mean(axis=1)
        return z <= self.threshold_

    def score(self, X):
        return np.abs((X - self.mean_) / self.std_).mean(axis=1)


def y_scrambling_test(clf, X_train, y_train, X_test, y_test,
                       n_permutations=N_PERMUTATIONS, seed=42):
    """
    Permutation test: shuffle y_train n times, compare AUC to real model.
    p-value = fraction of permuted AUCs >= real AUC.
    """
    rng = np.random.default_rng(seed)
    try:
        clf_r = clone(clf); clf_r.fit(X_train, y_train)
        real_auc = roc_auc_score(y_test, clf_r.predict_proba(X_test)[:,1])
    except Exception as e:
        logger.warning("y-scramble real model failed: %s", e)
        return {"real_auc":None,"perm_aucs":[],"p_value":None,"significant":False}

    perm_aucs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_train)
        try:
            clf_p = clone(clf); clf_p.fit(X_train, y_perm)
            perm_aucs.append(roc_auc_score(y_test, clf_p.predict_proba(X_test)[:,1]))
        except Exception:
            perm_aucs.append(0.5)
    perm_aucs = np.array(perm_aucs)
    p_value = float((perm_aucs >= real_auc).sum() + 1) / (len(perm_aucs) + 1)
    logger.info("y-Scrambling: real=%.3f, perm=%.3f+/-%.3f, p=%.4f",
                real_auc, perm_aucs.mean(), perm_aucs.std(), p_value)
    return {"real_auc":float(real_auc),"perm_aucs":perm_aucs.tolist(),
            "perm_mean":float(perm_aucs.mean()),"perm_std":float(perm_aucs.std()),
            "p_value":float(p_value),"significant":bool(p_value<0.05)}


def enrichment_factor(y_true, y_score, fraction=0.05):
    n = len(y_true); n_top = max(1, int(n*fraction))
    top = np.argsort(y_score)[::-1][:n_top]
    return float((y_true[top].sum()/n_top)/(y_true.sum()/n))


def bedroc_score(y_true, y_score, alpha=20.0):
    n = len(y_true); n_a = int(y_true.sum())
    if n_a==0 or n_a==n: return float(y_true.mean())
    order = np.argsort(y_score)[::-1]; y_s = y_true[order]; ra = n_a/n
    ri_sum = sum(np.exp(-alpha*i/n) for i,yi in enumerate(y_s,1) if yi==1)
    ri_max = ra*(1-np.exp(-alpha))/(1-np.exp(-alpha*n_a/n))
    ri_min = ra*(np.exp(alpha*ra)-1)/(np.exp(alpha)-1)*(1-np.exp(-alpha))
    return float(np.clip((ri_sum-ri_min)/(ri_max-ri_min),0,1))


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_shap_values(clf, X_train, X_explain, model_name):
    """
    Compute SHAP values for the active class (class 1).

    TreeExplainer for RF / XGBoost / GradientBoosting (exact, fast, O(n·T)).
    KernelExplainer for SVM / Neural Network (model-agnostic, slower).

    Key fixes for NumPy >= 1.20 / SHAP >= 0.41:
      - check_additivity=False  prevents np.bool deprecation inside SHAP
      - Handles both old API (list of arrays) and new API (single array ndim=3)
      - Robust pipeline unwrapping for sklearn Pipeline objects

    Returns float32 array of shape (n_samples, n_features) or None.
    """
    if not SHAP_AVAILABLE:
        return None

    # Unwrap sklearn Pipeline: apply all transforms, extract final estimator
    model = clf
    X_bg  = X_train.astype(np.float32)
    X_exp = X_explain.astype(np.float32)
    if hasattr(clf, "named_steps"):
        steps = list(clf.named_steps.values())
        for step in steps[:-1]:
            X_bg  = step.transform(X_bg).astype(np.float32)
            X_exp = step.transform(X_exp).astype(np.float32)
        model = steps[-1]

    try:
        # ── Tree-based models: RF, XGBoost, GradientBoosting ─────────────────
        if hasattr(model, "feature_importances_"):
            # Use smaller background sample to speed up interventional SHAP
            bg_size = min(100, len(X_bg))
            bg_idx  = np.random.choice(len(X_bg), bg_size, replace=False)
            bg_data = X_bg[bg_idx]

            explainer = shap.TreeExplainer(
                model,
                data=bg_data,
                feature_perturbation="interventional",
            )
            # check_additivity=False avoids np.bool / np.issubdtype issues
            sv = explainer.shap_values(X_exp, check_additivity=False)

            # Handle all output shapes:
            #   SHAP < 0.40  RF: list [class0_arr, class1_arr]
            #   SHAP >= 0.40 RF: single array (n, feat, 2)
            #   XGBoost binary: single array (n, feat)
            if isinstance(sv, list):
                shap_vals = np.array(sv[1] if len(sv) == 2 else sv[0], dtype=np.float32)
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_vals = sv[:, :, 1].astype(np.float32)   # class-1 slice
            else:
                shap_vals = np.array(sv, dtype=np.float32)

            logger.info("[SHAP] TreeExplainer %s: shape=%s", model_name, shap_vals.shape)
            return shap_vals

        # ── Black-box models: SVM, Neural Network ────────────────────────────
        else:
            bg_size = min(N_SHAP_BACKGROUND, len(X_bg))
            bg_idx  = np.random.choice(len(X_bg), bg_size, replace=False)
            bg_data = X_bg[bg_idx]

            def _predict_proba_1(x):
                x = np.array(x, dtype=np.float32)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x)[:, 1]
                return model.predict(x)

            explainer = shap.KernelExplainer(_predict_proba_1, bg_data)
            # KernelExplainer is O(n²) — keep sample small
            n_exp = min(50, len(X_exp))
            sv = explainer.shap_values(X_exp[:n_exp], silent=True)
            shap_vals = np.array(sv, dtype=np.float32)
            logger.info("[SHAP] KernelExplainer %s: shape=%s", model_name, shap_vals.shape)
            return shap_vals

    except Exception as e:
        logger.warning("[SHAP] Failed on %s: %s", model_name, e)
        return None


def compute_shap_reliability(shap_vals, y_prob):
    """
    SHAP Reliability Score: fraction of the top-30% most important features
    that push toward the active class. Range [0,1]; >0.6 = trustworthy.
    """
    if shap_vals is None or len(shap_vals)==0:
        return np.full(len(y_prob), np.nan)
    n = min(len(shap_vals), len(y_prob))
    sv = shap_vals[:n]
    threshold = np.percentile(np.abs(sv), 70)
    important = np.abs(sv) > threshold
    n_imp = important.sum(axis=1)
    n_pos = ((sv > 0) & important).sum(axis=1)
    return np.where(n_imp > 0, n_pos/n_imp, 0.5).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":13,
        "axes.titlesize":15,"axes.labelsize":13,
        "xtick.labelsize":11,"ytick.labelsize":11,"legend.fontsize":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":1.3,
        "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
        "figure.dpi":300,"savefig.dpi":300,
        "figure.facecolor":"white","axes.facecolor":"white",
    })


def _save(fig, path):
    fig.savefig(str(path), bbox_inches="tight", dpi=300)
    fig.savefig(str(path).replace(".pdf",".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info("Saved: %s", path.name)


def plot_shap_importance(shap_vals, feature_names, gene, model_name, top_n=N_SHAP_TOP_BITS):
    _style()
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]
    top_v = mean_abs[idx]; top_l = [feature_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(14,7))
    colors = plt.cm.RdYlGn(np.linspace(0.85,0.15,top_n))
    ax.barh(range(top_n), top_v[::-1], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_l[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Activity Prediction)",
                   fontsize=13, fontweight="bold")
    ax.set_title("SHAP Global Feature Importance — {}\n{} | Top-{} Morgan Bits".format(
        gene, model_name, top_n), fontsize=15, fontweight="bold", loc="left")
    ax.text(0.98,0.02,"Model: {}".format(model_name),
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4",facecolor="lightyellow",alpha=0.9))
    _save(fig, FIGURES_DIR/"{}_shap_importance.pdf".format(gene))


def plot_shap_beeswarm(shap_vals, X, feature_names, gene, model_name, top_n=20):
    _style()
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    sv_top = shap_vals[:,top_idx]; X_top = X[:,top_idx]
    labels = [feature_names[i] for i in top_idx]
    fig, ax = plt.subplots(figsize=(12,10))
    sc = None
    for ri, fi in enumerate(range(top_n-1,-1,-1)):
        sv = sv_top[:,fi]; feat = X_top[:,fi]
        y_j = ri + np.random.uniform(-0.35,0.35,len(sv))
        sc = ax.scatter(sv,y_j,c=feat,cmap="bwr",s=12,alpha=0.6,vmin=0,vmax=1,linewidths=0)
    if sc is not None:
        plt.colorbar(sc,ax=ax,label="Feature value (0=absent, 1=present)",shrink=0.6)
    ax.axvline(0,color="gray",lw=1.5,ls="--",alpha=0.8)
    ax.set_yticks(range(top_n)); ax.set_yticklabels(labels[::-1],fontsize=9)
    ax.set_xlabel("SHAP Value  (left=inactive, right=active)",fontsize=13,fontweight="bold")
    ax.set_title("SHAP Beeswarm — {}\n{} | Each dot = one compound".format(gene,model_name),
                  fontsize=15,fontweight="bold",loc="left")
    _save(fig, FIGURES_DIR/"{}_shap_beeswarm.pdf".format(gene))


def plot_shap_waterfall(shap_vals, X, compound_idx, feature_names, gene,
                         mol_id, expected_value, top_n=15):
    _style()
    sv = shap_vals[compound_idx]; feat = X[compound_idx]
    top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    top_sv  = sv[top_idx]; top_f = feat[top_idx]
    top_lbl = ["{} = {:.0f}".format(feature_names[i],top_f[j]) for j,i in enumerate(top_idx)]
    cumul = expected_value + np.cumsum(np.append(0,top_sv))
    fig, ax = plt.subplots(figsize=(12,8))
    for i,(sv_val,lbl) in enumerate(zip(top_sv,top_lbl)):
        col = NATURE_COLORS[1] if sv_val>0 else NATURE_COLORS[0]
        ax.barh(i,sv_val,left=cumul[i],color=col,edgecolor="white",height=0.7)
        ax.text(cumul[i]+sv_val/2,i,"{:+.4f}".format(sv_val),
                ha="center",va="center",fontsize=8,color="white",fontweight="bold")
    ax.set_yticks(range(top_n)); ax.set_yticklabels(top_lbl,fontsize=9)
    ax.axvline(expected_value,color="gray",lw=1.5,ls="--",
                label="Base value = {:.3f}".format(expected_value))
    ax.axvline(cumul[-1],color="black",lw=2,
                label="Prediction = {:.3f}".format(cumul[-1]))
    ax.set_xlabel("Predicted Active Probability",fontsize=13,fontweight="bold")
    ax.set_title("SHAP Waterfall — {}\nCompound: {}".format(gene,str(mol_id)[:20]),
                  fontsize=14,fontweight="bold",loc="left")
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=NATURE_COLORS[1],label="Pushes active (+)"),
                        mpatches.Patch(color=NATURE_COLORS[0],label="Pushes inactive (-)")],
               fontsize=10,loc="lower right")
    _save(fig, FIGURES_DIR/"{}_{}_shap_waterfall.pdf".format(gene,str(mol_id)[:20]))


def plot_y_scrambling(result, gene, model_name):
    _style()
    perm = np.array(result.get("perm_aucs",[])); real = result.get("real_auc",0)
    p    = result.get("p_value",1.0)
    if len(perm)==0: return
    fig,ax = plt.subplots(figsize=(9,6))
    ax.hist(perm,bins=25,color=NATURE_COLORS[7],edgecolor="white",alpha=0.85,
             label="Permuted AUC (n={})".format(len(perm)))
    ax.axvline(real,color=NATURE_COLORS[0],lw=3,label="Real AUC = {:.3f}".format(real))
    ax.axvline(np.percentile(perm,95),color="gray",lw=2,ls="--",label="95th percentile")
    ax.set_xlabel("AUC-ROC",fontsize=13,fontweight="bold")
    ax.set_ylabel("Count",fontsize=13,fontweight="bold")
    sig_str = "(*** p<0.001)" if p<0.001 else "(** p<0.01)" if p<0.01 else "(* p<0.05)" if p<0.05 else "(n.s.)"
    ax.set_title("y-Scrambling Test — {}\n{} | p={:.4f} {}".format(
        gene,model_name,p,sig_str),fontsize=14,fontweight="bold",loc="left")
    ax.legend(fontsize=11)
    _save(fig, FIGURES_DIR/"{}_y_scrambling.pdf".format(gene))


def plot_external_validation(ext, gene):
    _style()
    fig = plt.figure(figsize=(20,8))
    gs  = gridspec.GridSpec(1,4,figure=fig,wspace=0.40)

    ax_bar = fig.add_subplot(gs[0,:2])
    names = ["AUC-ROC","BEDROC","EF 1%","EF 5%"]
    vals  = [ext.get("scaffold_auc",0),ext.get("bedroc",0),
              ext.get("ef_1pct",0),ext.get("ef_5pct",0)]
    colors= [NATURE_COLORS[1] if v>=0.7 else NATURE_COLORS[4] if v>=0.5 else NATURE_COLORS[0]
              for v in vals]
    bars = ax_bar.bar(names,vals,color=colors,edgecolor="white",width=0.5)
    ax_bar.set_ylim(0,max(max(vals)*1.25,1.6))
    ax_bar.set_ylabel("Score",fontsize=13,fontweight="bold")
    ax_bar.set_title("External Validation Metrics — {}\n(Scaffold split, n_test={})".format(
        gene,ext.get("n_test","?")),fontsize=14,fontweight="bold",loc="left")
    for bar,v in zip(bars,vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.02,
                    "{:.3f}".format(v),ha="center",fontsize=12,fontweight="bold")

    ax_roc = fig.add_subplot(gs[0,2])
    fpr = ext.get("roc_fpr",[0,1]); tpr = ext.get("roc_tpr",[0,1])
    ax_roc.plot(fpr,tpr,lw=2.5,color=NATURE_COLORS[3],
                 label="AUC={:.3f}".format(ext.get("scaffold_auc",0)))
    ax_roc.fill_between(fpr,tpr,alpha=0.15,color=NATURE_COLORS[3])
    ax_roc.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.6)
    ax_roc.set_xlabel("FPR",fontsize=12,fontweight="bold")
    ax_roc.set_ylabel("TPR",fontsize=12,fontweight="bold")
    ax_roc.set_title("ROC (Scaffold Split)",fontsize=13,fontweight="bold")
    ax_roc.legend(fontsize=11); ax_roc.set_aspect("equal")

    ax_ad = fig.add_subplot(gs[0,3])
    ni = ext.get("ad_n_inside",0); no = ext.get("ad_n_outside",0)
    if ni+no>0:
        ax_ad.pie([ni,no],
                   labels=["Inside AD\n(n={})".format(ni),"Outside AD\n(n={})".format(no)],
                   colors=[NATURE_COLORS[2],NATURE_COLORS[0]],autopct="%1.1f%%",startangle=90,
                   textprops={"fontsize":11},wedgeprops={"linewidth":2,"edgecolor":"white"})
    ax_ad.set_title("Applicability Domain\n(p{}={})".format(
        AD_PERCENTILE,ext.get("ad_threshold_fmt","?")),fontsize=13,fontweight="bold")

    n_in = ni; n_tot = max(1,ni+no)
    fig.suptitle("External Validation — {}\ny-scramble p={:.4f} | BEDROC={:.3f} | AD={:.1f}%".format(
        gene,ext.get("y_scramble_p",1.0),ext.get("bedroc",0),n_in/n_tot*100),
        fontsize=17,fontweight="bold")
    _save(fig, FIGURES_DIR/"{}_external_validation.pdf".format(gene))


def plot_roc_curves(models_test, gene):
    _style(); fig,ax = plt.subplots(figsize=(8,8))
    for i,(name,(yt,yp)) in enumerate(models_test.items()):
        fpr,tpr,_ = roc_curve(yt,yp); auc=roc_auc_score(yt,yp)
        ax.plot(fpr,tpr,lw=2.5,color=NATURE_COLORS[i%len(NATURE_COLORS)],
                 label="{} AUC={:.3f}".format(name,auc))
    ax.plot([0,1],[0,1],"k--",lw=1.5,alpha=0.6)
    ax.fill_between([0,1],[0,1],alpha=0.05,color="gray")
    ax.set_xlabel("False Positive Rate",fontsize=14,fontweight="bold")
    ax.set_ylabel("True Positive Rate",fontsize=14,fontweight="bold")
    ax.set_title("ROC Curves — {}".format(gene),fontsize=16,fontweight="bold")
    ax.legend(loc="lower right",fontsize=11); ax.set_aspect("equal")
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    _save(fig, FIGURES_DIR/"{}_roc_curves.pdf".format(gene))


def plot_regression_scatter(y_true, y_pred, gene, model_name):
    _style(); r2=r2_score(y_true,y_pred); mae=mean_absolute_error(y_true,y_pred)
    pr,pv = stats.pearsonr(y_true,y_pred)
    fig,ax = plt.subplots(figsize=(8,8))
    sc=ax.scatter(y_true,y_pred,alpha=0.6,s=40,c=y_true,cmap="coolwarm",
                   edgecolors="white",linewidths=0.5)
    plt.colorbar(sc,ax=ax,label="True pIC50",shrink=0.8)
    lims=[min(min(y_true),min(y_pred))-0.5,max(max(y_true),max(y_pred))+0.5]
    ax.plot(lims,lims,"k--",lw=2,alpha=0.8); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("True pIC50",fontsize=14,fontweight="bold")
    ax.set_ylabel("Predicted pIC50",fontsize=14,fontweight="bold")
    ax.set_title("pIC50 Regression — {}\n{}".format(model_name,gene),
                  fontsize=15,fontweight="bold")
    ax.text(0.05,0.95,"R²={:.3f}\nMAE={:.3f}\nr={:.3f} (p={:.2e})".format(r2,mae,pr,pv),
            transform=ax.transAxes,fontsize=12,va="top",
            bbox=dict(boxstyle="round,pad=0.5",facecolor="wheat",alpha=0.8))
    _save(fig, FIGURES_DIR/"{}_{}_{}.pdf".format(gene,"regression",model_name.replace(" ","_")))


def plot_cv_performance(cv_results, gene):
    _style(); df_cv=pd.DataFrame(cv_results)
    metrics=["mean_accuracy","mean_sensitivity","mean_specificity","mean_auc"]
    labels=["Accuracy","Sensitivity","Specificity","AUC-ROC"]
    stds=["std_accuracy","std_sensitivity","std_specificity","std_auc"]
    n=len(df_cv); x=np.arange(n); w=0.18
    fig,ax = plt.subplots(figsize=(14,7))
    for i,(m,lbl,sd) in enumerate(zip(metrics,labels,stds)):
        vals=df_cv[m].values; errs=df_cv[sd].values
        bars=ax.bar(x+(i-1.5)*w,vals,w,yerr=errs,label=lbl,
                     color=NATURE_COLORS[i],alpha=0.85,capsize=5,
                     error_kw={"linewidth":1.5,"ecolor":"black"})
        for bar,v,e in zip(bars,vals,errs):
            ax.text(bar.get_x()+w/2,bar.get_height()+e+0.01,
                    "{:.2f}".format(v),ha="center",va="bottom",fontsize=8.5,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(df_cv["model"],fontsize=11,fontweight="bold")
    ax.set_ylabel("Score",fontsize=14,fontweight="bold"); ax.set_ylim(0,1.15)
    ax.axhline(0.5,color="gray",ls="--",alpha=0.5,label="Random")
    ax.set_title("5-Fold CV — {}".format(gene),fontsize=16,fontweight="bold")
    ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.06),ncol=5,fontsize=11)
    plt.tight_layout()
    _save(fig, FIGURES_DIR/"{}_cv_performance.pdf".format(gene))


def plot_shap_compound_selection(df, gene):
    _style()
    df = df.dropna(subset=["composite_score","shap_reliability"]).copy()
    if df.empty: return
    fig,axes = plt.subplots(1,2,figsize=(18,8))

    ax = axes[0]
    sc = ax.scatter(df["composite_score"],df["shap_reliability"],s=60,alpha=0.7,
                     c=df["pIC50"] if "pIC50" in df.columns else "steelblue",
                     cmap="RdYlGn",vmin=4,vmax=11,edgecolors="white",linewidths=0.5)
    plt.colorbar(sc,ax=ax,label="Measured pIC50",shrink=0.85)
    if "AD_inside" in df.columns:
        inad=df[df["AD_inside"]]
        ax.scatter(inad["composite_score"],inad["shap_reliability"],s=120,marker="*",
                    c="gold",edgecolors="darkorange",lw=1.5,zorder=5,label="Inside AD")
        ax.legend(fontsize=11)
    ax.axvline(df["composite_score"].median(),color="gray",lw=1.5,ls="--",alpha=0.7)
    ax.axhline(df["shap_reliability"].median(),color="gray",lw=1.5,ls="--",alpha=0.7)
    xl=ax.get_xlim(); yl=ax.get_ylim()
    ax.text(xl[1]*0.98,yl[1]*0.98,"High score\nHigh reliability\n(Best candidates)",
            ha="right",va="top",fontsize=10,color=NATURE_COLORS[2],
            bbox=dict(boxstyle="round,pad=0.3",facecolor="honeydew",alpha=0.85))
    ax.set_xlabel("Composite Score",fontsize=13,fontweight="bold")
    ax.set_ylabel("SHAP Reliability Score",fontsize=13,fontweight="bold")
    ax.set_title("SHAP Reliability vs Composite Score — {}".format(gene),
                  fontsize=14,fontweight="bold",loc="left")

    ax2 = axes[1]
    df["shap_composite"]=df["composite_score"]*0.6+df["shap_reliability"]*0.4
    top20=df.nlargest(min(20,len(df)),"shap_composite").reset_index(drop=True)
    pal=plt.cm.RdYlGn(np.linspace(0.3,0.9,len(top20)))[::-1]
    ax2.barh(range(len(top20)),top20["shap_composite"],color=pal,edgecolor="white",height=0.7)
    ax2.set_yticks(range(len(top20)))
    id_col="molecule_chembl_id" if "molecule_chembl_id" in top20.columns else None
    labels=[]
    for i,row in top20.iterrows():
        cid=str(row[id_col] if id_col else i)[:14]
        labels.append("#{} {}".format(i+1,cid))
    ax2.set_yticklabels(labels,fontsize=9); ax2.invert_yaxis()
    ax2.set_xlabel("SHAP-Composite (0.6*score + 0.4*reliability)",fontsize=12,fontweight="bold")
    ax2.set_title("Top-{} SHAP-Validated Candidates".format(len(top20)),
                   fontsize=14,fontweight="bold")
    if "AD_inside" in top20.columns:
        for i,row in top20.iterrows():
            ad_s="AD+" if row.get("AD_inside",False) else "AD-"
            col=NATURE_COLORS[2] if row.get("AD_inside",False) else NATURE_COLORS[0]
            ax2.text(0.01,i,ad_s,transform=ax2.get_yaxis_transform(),fontsize=8,color=col,va="center")

    plt.suptitle("SHAP-Based Reliability Screening — {}".format(gene),fontsize=18,fontweight="bold")
    plt.tight_layout()
    _save(fig, FIGURES_DIR/"{}_shap_compound_selection.pdf".format(gene))


# ══════════════════════════════════════════════════════════════════════════════
# Main Stage 3 function
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_analysis(filtered_data, force_rerun=False):
    """
    Stage 3: Morgan FP -> ML -> SHAP -> External Validation -> Compound Ranking.

    Per gene:
      1  Morgan fingerprints (2048 bits, radius 2)
      2  5-fold stratified CV on 5 classifiers + 3 regressors
      3  Scaffold split (Murcko, 15% external test)
      4  SHAP: global importance, beeswarm, waterfall for best model
      5  y-Scrambling: n=100 permutations, p<0.05 required
      6  Applicability Domain: z-score, 95th percentile
      7  Enrichment: BEDROC, EF1%, EF5%, EF10%
      8  SHAP Reliability Score per compound
      9  Composite score = 0.35*pIC50 + 0.25*QED + 0.25*P(active) + 0.15*SHAP_rel
     10  AD-filtered top candidates saved to {GENE}_top_AD_candidates.csv
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed — running without SHAP. pip install shap")

    scored_data = {}

    for gene, df_all in filtered_data.items():

        ckpt = Checkpoint("s3_ml_{}".format(gene))
        if ckpt.exists() and not force_rerun:
            result = ckpt.load()
            logger.info("[%s] Loaded from checkpoint", gene)
            scored_data[gene] = result
            continue

        passed_col = "passed_all_filters"
        df = (df_all[df_all[passed_col]].copy().reset_index(drop=True)
               if passed_col in df_all.columns else df_all.copy().reset_index(drop=True))

        logger.info("\n%s\n[%s] ML: %d compounds", "="*60, gene, len(df))
        if len(df) < 30:
            logger.warning("[%s] Too few (%d). Skipping.", gene, len(df)); continue

        X, df_valid = generate_fingerprint_matrix(df, method="morgan")
        if X.shape[0] < 20:
            logger.warning("[%s] Insufficient FPs. Skipping.", gene); continue

        feat_names = bit_labels(X.shape[1])
        df_valid = assign_activity_labels(df_valid)
        y_class  = df_valid["active"].values
        y_reg    = df_valid["pIC50"].values

        # Random split for model training
        X_tr, X_te, y_tr_c, y_te_c, y_tr_r, y_te_r = train_test_split(
            X, y_class, y_reg,
            test_size=ML_CONFIG["test_size"],
            random_state=ML_CONFIG["random_state"],
            stratify=y_class if y_class.sum() >= 2 else None,
        )

        # Scaffold split for external validation
        sc_train, sc_test = scaffold_split(df_valid, test_frac=0.15,
                                            seed=ML_CONFIG["random_state"])
        Xst = X[sc_train]; yst_c = y_class[sc_train]
        Xse = X[sc_test];  yse_c = y_class[sc_test]

        # Classifiers
        classifiers = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,max_depth=None,min_samples_split=5,
                class_weight="balanced",random_state=ML_CONFIG["random_state"],
                n_jobs=ML_CONFIG["n_jobs"]),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300,max_depth=6,learning_rate=0.05,
                subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",
                random_state=ML_CONFIG["random_state"],verbosity=0),
            "SVM (RBF)": Pipeline([
                ("scaler",StandardScaler()),
                ("svc",SVC(kernel="rbf",C=1.0,gamma="scale",probability=True,
                            random_state=ML_CONFIG["random_state"]))]),
            "Neural Network": Pipeline([
                ("scaler",StandardScaler()),
                ("mlp",MLPClassifier(
                    hidden_layer_sizes=(512,256,128,64),activation="relu",
                    solver="adam",alpha=1e-4,batch_size=64,
                    learning_rate="adaptive",max_iter=200,
                    random_state=ML_CONFIG["random_state"],
                    early_stopping=True,validation_fraction=0.1,n_iter_no_change=15))]),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200,learning_rate=0.05,max_depth=4,
                subsample=0.8,random_state=ML_CONFIG["random_state"]),
        }

        kf = StratifiedKFold(n_splits=ML_CONFIG["n_folds"],shuffle=True,
                              random_state=ML_CONFIG["random_state"])
        cv_results = {}; models_roc = {}; best_clf = None; best_auc = 0.0

        for mname, clf in classifiers.items():
            logger.info("[%s] Training: %s", gene, mname)
            try:
                fold_m = {"accuracy":[],"sensitivity":[],"specificity":[],"auc":[]}
                for tr,te in kf.split(X_tr, y_tr_c):
                    cf = _clone_model(clf); cf.fit(X_tr[tr],y_tr_c[tr])
                    yp = cf.predict(X_tr[te]); yb = cf.predict_proba(X_tr[te])[:,1]
                    fold_m["accuracy"].append(accuracy_score(y_tr_c[te],yp))
                    fold_m["sensitivity"].append(recall_score(y_tr_c[te],yp,zero_division=0))
                    fold_m["specificity"].append(recall_score(y_tr_c[te],yp,pos_label=0,zero_division=0))
                    fold_m["auc"].append(roc_auc_score(y_tr_c[te],yb))
                cv_results[mname] = {
                    "gene":gene,"model":mname,
                    **{"mean_{}".format(k):float(np.mean(v)) for k,v in fold_m.items()},
                    **{"std_{}".format(k): float(np.std(v))  for k,v in fold_m.items()},
                }
                clf.fit(X_tr, y_tr_c)
                yp_te = clf.predict_proba(X_te)[:,1]
                auc = roc_auc_score(y_te_c,yp_te)
                models_roc[mname] = (y_te_c,yp_te)
                logger.info("[%s] %s: CV_AUC=%.3f+/-%.3f test_AUC=%.3f",
                            gene,mname,cv_results[mname]["mean_auc"],
                            cv_results[mname]["std_auc"],auc)
                if auc > best_auc: best_auc=auc; best_clf=(mname,clf)
            except Exception as e:
                logger.error("[%s] %s: %s", gene, mname, e)

        # Regressors
        regressors = {
            "Random Forest": RandomForestRegressor(n_estimators=300,
                random_state=ML_CONFIG["random_state"],n_jobs=ML_CONFIG["n_jobs"]),
            "XGBoost": xgb.XGBRegressor(n_estimators=300,max_depth=6,
                learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                random_state=ML_CONFIG["random_state"],verbosity=0),
            "Neural Network": Pipeline([
                ("scaler",StandardScaler()),
                ("mlp",MLPRegressor(hidden_layer_sizes=(512,256,128,64),
                    activation="relu",solver="adam",alpha=1e-4,max_iter=200,
                    random_state=ML_CONFIG["random_state"],
                    early_stopping=True,n_iter_no_change=15))]),
        }
        best_reg=None; best_r2=-999.0
        for rname,reg in regressors.items():
            try:
                reg.fit(X_tr,y_tr_r)
                ypr=reg.predict(X_te)
                r2=r2_score(y_te_r,ypr); mae=mean_absolute_error(y_te_r,ypr)
                logger.info("[%s] Reg %s: R2=%.3f MAE=%.3f",gene,rname,r2,mae)
                try: plot_regression_scatter(y_te_r,ypr,gene,rname)
                except Exception: pass
                if r2>best_r2: best_r2=r2; best_reg=(rname,reg)
            except Exception as e:
                logger.error("[%s] Reg %s: %s",gene,rname,e)

        # SHAP
        shap_vals_global=None; shap_expected=float(y_tr_c.mean())
        best_mname = best_clf[0] if best_clf else "Unknown"
        if SHAP_AVAILABLE and best_clf:
            logger.info("[%s] SHAP for: %s",gene,best_mname)
            clf_b = best_clf[1]
            n_exp = min(N_SHAP_EXPLAIN,len(X))
            exp_idx = np.random.choice(len(X),n_exp,replace=False)
            shap_vals_global = compute_shap_values(clf_b,X_tr,X[exp_idx],best_mname)
            if shap_vals_global is not None:
                try: plot_shap_importance(shap_vals_global,feat_names,gene,best_mname)
                except Exception as e: logger.warning("[%s] SHAP importance: %s",gene,e)
                try: plot_shap_beeswarm(shap_vals_global,X[exp_idx],feat_names,gene,best_mname)
                except Exception as e: logger.warning("[%s] SHAP beeswarm: %s",gene,e)
                # Waterfall for top-3 predicted active compounds
                try:
                    yp_all = clf_b.predict_proba(X)[:,1]
                    top3 = np.argsort(yp_all)[::-1][:3]
                    for rank,tidx in enumerate(top3):
                        if tidx<len(shap_vals_global):
                            mol_id = (df_valid.get("molecule_chembl_id",
                                pd.Series(range(len(df_valid)))).iloc[tidx])
                            try: plot_shap_waterfall(shap_vals_global,X[exp_idx],
                                    min(tidx,len(shap_vals_global)-1),
                                    feat_names,gene,mol_id,shap_expected)
                            except Exception: pass
                        if rank>=2: break
                except Exception as e: logger.warning("[%s] SHAP waterfall: %s",gene,e)

        # y-Scrambling
        scramble={"real_auc":None,"perm_aucs":[],"p_value":None,"significant":False}
        if best_clf:
            try:
                scramble = y_scrambling_test(best_clf[1],X_tr,y_tr_c,X_te,y_te_c,
                                              n_permutations=N_PERMUTATIONS,
                                              seed=ML_CONFIG["random_state"])
                try: plot_y_scrambling(scramble,gene,best_mname)
                except Exception: pass
            except Exception as e: logger.warning("[%s] y-scramble: %s",gene,e)

        # AD
        ad = ApplicabilityDomain(percentile=AD_PERCENTILE)
        ad.fit(X_tr)
        ad_inside = ad.predict(X)
        ad_scores  = ad.score(X)

        # External validation (scaffold split)
        ext_metrics = {"n_train":len(sc_train),"n_test":len(sc_test)}
        if best_clf and len(sc_test)>0 and yse_c.sum()>0:
            try:
                clf_ext = clone(best_clf[1]); clf_ext.fit(Xst,yst_c)
                yp_ext  = clf_ext.predict_proba(Xse)[:,1]
                fpr_e,tpr_e,_ = roc_curve(yse_c,yp_ext)
                ext_metrics.update({
                    "scaffold_auc":float(roc_auc_score(yse_c,yp_ext)),
                    "bedroc":float(bedroc_score(yse_c,yp_ext)),
                    "ef_1pct":float(enrichment_factor(yse_c,yp_ext,0.01)),
                    "ef_5pct":float(enrichment_factor(yse_c,yp_ext,0.05)),
                    "ef_10pct":float(enrichment_factor(yse_c,yp_ext,0.10)),
                    "roc_fpr":fpr_e.tolist(),"roc_tpr":tpr_e.tolist(),
                    "ad_n_inside":int(ad_inside[sc_test].sum()),
                    "ad_n_outside":int((~ad_inside[sc_test]).sum()),
                    "ad_threshold_fmt":"{:.4f}".format(ad.threshold_),
                    "y_scramble_p":scramble.get("p_value",1.0),
                })
                logger.info("[%s] Ext val: AUC=%.3f BEDROC=%.3f EF1%%=%.2f EF5%%=%.2f",
                            gene,ext_metrics["scaffold_auc"],ext_metrics["bedroc"],
                            ext_metrics["ef_1pct"],ext_metrics["ef_5pct"])
                try: plot_external_validation(ext_metrics,gene)
                except Exception as e: logger.warning("[%s] ext val plot: %s",gene,e)
            except Exception as e: logger.warning("[%s] ext val: %s",gene,e)

        safe_save_csv(pd.DataFrame([ext_metrics]),
                       RESULTS_DIR/"{}_external_validation.csv".format(gene))

        # Score all compounds
        df_valid["pred_active_prob"] = np.nan
        df_valid["pred_pIC50"]       = np.nan
        df_valid["AD_inside"]        = ad_inside
        df_valid["AD_z_score"]       = ad_scores
        df_valid["shap_reliability"] = np.nan

        if best_clf:
            df_valid["pred_active_prob"] = best_clf[1].predict_proba(X)[:,1]
        if best_reg:
            df_valid["pred_pIC50"] = best_reg[1].predict(X)

        # SHAP reliability for all compounds (TreeExplainer only, fast)
        if SHAP_AVAILABLE and best_clf:
            try:
                clf_b2  = best_clf[1]
                model_b2 = clf_b2
                X_all    = X.astype(np.float32)

                # Unwrap pipeline transforms
                if hasattr(clf_b2, "named_steps"):
                    steps = list(clf_b2.named_steps.values())
                    for step in steps[:-1]:
                        X_all = step.transform(X_all).astype(np.float32)
                    model_b2 = steps[-1]

                if hasattr(model_b2, "feature_importances_"):
                    # check_additivity=False: avoids np.bool / np.issubdtype error
                    exp_all = shap.TreeExplainer(model_b2)
                    sv_all  = exp_all.shap_values(X_all, check_additivity=False)

                    if isinstance(sv_all, list):
                        sv_all = np.array(
                            sv_all[1] if len(sv_all) == 2 else sv_all[0],
                            dtype=np.float32
                        )
                    elif isinstance(sv_all, np.ndarray) and sv_all.ndim == 3:
                        sv_all = sv_all[:, :, 1].astype(np.float32)
                    else:
                        sv_all = np.array(sv_all, dtype=np.float32)

                    yp_all = best_clf[1].predict_proba(X)[:, 1]
                    df_valid["shap_reliability"] = compute_shap_reliability(sv_all, yp_all)
                    logger.info("[%s] SHAP reliability computed: %d compounds",
                                gene, len(df_valid))
            except Exception as e:
                logger.warning("[%s] SHAP reliability: %s", gene, e)

        # Composite score (SHAP reliability included)
        p_min=df_valid["pIC50"].min(); p_max=df_valid["pIC50"].max()
        p_norm=(df_valid["pIC50"]-p_min)/(p_max-p_min+1e-9)
        qed_c = df_valid.get("qed",pd.Series(0.5,index=df_valid.index)).fillna(0.5)
        shap_r=df_valid["shap_reliability"].fillna(0.5)

        df_valid["composite_score"] = (
            0.35*p_norm + 0.25*qed_c
            + 0.25*df_valid["pred_active_prob"].fillna(0.5)
            + 0.15*shap_r
        )
        df_valid["shap_composite"] = 0.60*df_valid["composite_score"] + 0.40*shap_r

        df_valid.sort_values("shap_composite",ascending=False,inplace=True)
        df_valid.reset_index(drop=True,inplace=True)

        safe_save_csv(df_valid, RESULTS_DIR/"{}_ml_scored.csv".format(gene))

        # Top candidates inside AD
        ad_top = df_valid[df_valid["AD_inside"]].nlargest(
            min(40,len(df_valid)),"shap_composite")
        safe_save_csv(ad_top, RESULTS_DIR/"{}_top_AD_candidates.csv".format(gene))
        logger.info("[%s] Top AD candidates: %d",gene,len(ad_top))

        # Figures
        try: plot_roc_curves(models_roc,gene)
        except Exception as e: logger.warning("[%s] ROC: %s",gene,e)
        try: plot_cv_performance(list(cv_results.values()),gene)
        except Exception as e: logger.warning("[%s] CV: %s",gene,e)
        try: plot_shap_compound_selection(df_valid,gene)
        except Exception as e: logger.warning("[%s] SHAP selection: %s",gene,e)

        logger.info(
            "[%s] Complete:\n"
            "  Best clf : %s (AUC=%.3f)\n"
            "  Best reg : %s (R2=%.3f)\n"
            "  Scaffold AUC: %.3f | BEDROC: %.3f\n"
            "  y-scramble p: %s | AD coverage: %.1f%%\n"
            "  AD top candidates: %d",
            gene,
            best_clf[0] if best_clf else "N/A", best_auc,
            best_reg[0] if best_reg else "N/A", best_r2,
            ext_metrics.get("scaffold_auc",0), ext_metrics.get("bedroc",0),
            "{:.4f}".format(scramble.get("p_value",1.0)),
            ad_inside.mean()*100, len(ad_top)
        )

        result = {
            "df_scored":     df_valid,
            "cv_results":    list(cv_results.values()),
            "best_clf":      best_clf[0] if best_clf else None,
            "best_clf_auc":  best_auc,
            "best_reg":      best_reg[0] if best_reg else None,
            "best_reg_r2":   best_r2,
            "ext_validation":ext_metrics,
            "y_scrambling":  scramble,
            "ad_model":      ad,
        }
        ckpt.save(result)
        scored_data[gene] = result

    return scored_data


if __name__ == "__main__":
    import numpy as np, pandas as pd
    np.random.seed(42)
    smiles_list = [
        "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
        "CCOc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCC",
        "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
        "CC1=C(C(=O)Nc2ccccc2)C(c2ccccc2Cl)NC1=O",
        "COc1ccc(-c2nc(N3CCOCC3)c3ccccc3n2)cc1",
        "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1C",
        "O=C(Nc1ccc(N2CCOCC2)cc1)c1ccc(F)cc1",
    ] * 20
    n = len(smiles_list)
    sample = {"EGFR": pd.DataFrame({
        "smiles": smiles_list,
        "pIC50":  np.random.uniform(4.5,10.5,n),
        "qed":    np.random.uniform(0.4,0.9,n),
        "mw":     np.random.uniform(280,490,n),
        "logp":   np.random.uniform(1,5,n),
        "passed_all_filters": [True]*n,
        "molecule_chembl_id": ["CHEMBL{:06d}".format(i) for i in range(n)],
    })}
    result = run_ml_analysis(sample, force_rerun=True)
    for gene,r in result.items():
        ev = r.get("ext_validation",{})
        print("\n[{}]".format(gene))
        print("  Best clf: {} AUC={:.3f}".format(r["best_clf"],r["best_clf_auc"]))
        print("  Best reg: {} R2={:.3f}".format(r["best_reg"],r["best_reg_r2"]))
        print("  Scaffold AUC: {:.3f}".format(ev.get("scaffold_auc",0)))
        print("  BEDROC: {:.3f}".format(ev.get("bedroc",0)))
        print("  y-scramble p: {}".format(r.get("y_scrambling",{}).get("p_value")))
        print("  AD top: {}".format(len(r["df_scored"][r["df_scored"]["AD_inside"]])))