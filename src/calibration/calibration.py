from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

__all__ = [
    # loaders / feature builders (pairwise, protocol-aligned)
    "merge_case_pairwise",
    "make_pair_features",
    "load_pairwise_for_scenario",
    # selection / modeling
    "pick_best_feature_on_dev",
    "fit_logistic",
    # metrics helpers
    "roc_arrays",
    "eer_from_curve",
    "apcer_bpcer_at_threshold",
]

# ---------- Loaders (comparison-level, per case) ----------

def merge_case_pairwise(project_root: Path, split: str, case: str) -> pd.DataFrame:
    """
    Merge pairs_<split>_<case>.parquet with dtw_cache_<split>_<case>.parquet on pair_id.
    Returns a DataFrame including: pair_id, label, d_raw, d_bound, path_len, len_A, len_B.
    """
    project_root = Path(project_root)
    pairs = pd.read_parquet(project_root / "data" / "pairs" / split / f"pairs_{split}_{case}.parquet", engine="pyarrow")
    cache = pd.read_parquet(project_root / "data" / "dtw_cache" / split / f"dtw_cache_{split}_{case}.parquet", engine="pyarrow")
    df = pairs.merge(cache, on="pair_id", how="inner", suffixes=("_pairs", "_cache"))
    # Normalize label column
    if "label_cache" in df.columns:
        df["label"] = df["label_cache"].astype(int)
    elif "label_pairs" in df.columns:
        df["label"] = df["label_pairs"].astype(int)
    else:
        df["label"] = df["label"].astype(int)
    # Keep only needed columns
    keep = ["pair_id","label","d_raw","d_bound","path_len","len_A","len_B"]
    for k in keep:
        if k not in df.columns:
            raise KeyError(f"Required column '{k}' missing in merged {split}/{case}")
    return df[keep].copy()

def load_pairwise_for_scenario(project_root: Path, split: str, scenario: str) -> pd.DataFrame:
    """
    Build a per-scenario table from per-case merges.
    Scenarios:
      - skilled_only  : genuine vs skilled
      - random_only   : genuine vs random
      - skilled_random: genuine vs (skilled ∪ random)
    """
    g = merge_case_pairwise(project_root, split, "genuine")
    s = merge_case_pairwise(project_root, split, "skilled")
    r = merge_case_pairwise(project_root, split, "random")
    if scenario == "skilled_only":
        return pd.concat([g, s], ignore_index=True)
    if scenario == "random_only":
        return pd.concat([g, r], ignore_index=True)
    if scenario == "skilled_random":
        return pd.concat([g, s, r], ignore_index=True)
    raise ValueError(f"Unknown scenario: {scenario}")

# ---------- Feature engineering (pairwise) ----------

def make_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    From comparison-level fields, create calibration-friendly features.
    All features are oriented as 'higher ⇒ more genuine' (i.e., negative of distances).
    """
    out = df.copy()
    # raw similarities
    out["s_raw"]   = -out["d_raw"].astype(float)
    out["s_bound"] = -out["d_bound"].astype(float) if "d_bound" in out else np.nan

    # normalizations (guard divide-by-zero)
    eps = 1e-12
    out["s_by_path"]   = -out["d_raw"] / np.maximum(out["path_len"], eps)
    out["s_by_lenA"]   = -out["d_raw"] / np.maximum(out["len_A"],   eps)
    out["s_by_lenB"]   = -out["d_raw"] / np.maximum(out["len_B"],   eps)
    out["s_by_avglen"] = -out["d_raw"] / np.maximum((out["len_A"] + out["len_B"]) / 2.0, eps)

    # Clean inf/nan
    for c in ["s_bound","s_by_path","s_by_lenA","s_by_lenB","s_by_avglen"]:
        if c in out.columns:
            col = out[c].astype(float)
            med = float(np.nanmedian(col))
            out[c] = np.where(np.isfinite(col), col, med)

    return out

# ---------- Model selection & training ----------

def pick_best_feature_on_dev(df_dev_feats: pd.DataFrame, candidates: List[str]) -> tuple[str, float]:
    y = df_dev_feats["label"].to_numpy(int)
    best_name, best_auc = candidates[0], -np.inf
    for c in candidates:
        if c not in df_dev_feats.columns:
            continue
        auc_c = float(roc_auc_score(y, df_dev_feats[c].to_numpy(float)))
        if auc_c > best_auc:
            best_auc, best_name = auc_c, c
    if not np.isfinite(best_auc):
        raise ValueError("No valid candidate features found for AUC selection.")
    return best_name, best_auc

def fit_logistic(df_feats: pd.DataFrame, feature_col: str, *, test_size: float = 0.2, seed: int = 42):
    X = df_feats[[feature_col]].to_numpy(float)
    y = df_feats["label"].to_numpy(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    clf = LogisticRegression(solver="liblinear", random_state=seed)
    clf.fit(Xtr, ytr)
    yprob_te = clf.predict_proba(Xte)[:, 1]
    return clf, (Xte, yte), yprob_te

# ---------- Metrics helpers ----------

def roc_arrays(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return fpr, tpr, thr, auc

def eer_from_curve(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    return float(thr[idx]), float((fpr[idx] + fnr[idx]) / 2.0)

def apcer_bpcer_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> tuple[float, float]:
    y_pred = (y_score >= thr).astype(int)    # positive = genuine
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    apcer = fp / max(fp + tn, 1)            # attack false accept (FPR)
    bpcer = fn / max(fn + tp, 1)            # bona fide false reject (FNR)
    return float(apcer), float(bpcer)
