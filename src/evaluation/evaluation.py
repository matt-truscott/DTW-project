import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score


def load_results(pairs_path: str | Path, cache_path: str | Path) -> pd.DataFrame:
    """
    Load pair metadata and DTW cache, merge on pair_id.
    If `pair_id` is missing from the pairs table, create it from the index.
    After merging, collapse any duplicated 'label' columns into a single 'label'.
    """
    pairs = pd.read_parquet(pairs_path)
    if "pair_id" not in pairs.columns:
        pairs = pairs.reset_index(drop=False).rename(columns={"index": "pair_id"})

    cache = pd.read_parquet(cache_path)

    # merge and force suffixes so we can disambiguate
    df = pairs.merge(
        cache,
        on="pair_id",
        how="inner",
        suffixes=("_pairs", "_cache")
    )

    # collapse the two label columns into one
    if "label_pairs" in df.columns and "label_cache" in df.columns:
        # they should be identical, but prefer the _pairs version
        df["label"] = df["label_pairs"]
        df.drop(["label_pairs", "label_cache"], axis=1, inplace=True)
    elif "label_pairs" in df.columns:
        df.rename(columns={"label_pairs": "label"}, inplace=True)
    elif "label_cache" in df.columns:
        df.rename(columns={"label_cache": "label"}, inplace=True)

    return df


def compute_metrics(
    df: pd.DataFrame,
    score_col: str = "d_raw",
    label_col: str = "label",
) -> dict:
    """
    Given merged DataFrame with:
      - a score column (higher → more genuine), and
      - a binary label column (1=genuine, 0=forgery),
    compute ROC curve, AUC, and EER.
    """
    y_true  = df[label_col].to_numpy(dtype=int)
    y_score = df[score_col].to_numpy(dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    fnr = 1.0 - tpr
    diffs = np.abs(fnr - fpr)
    idx = int(np.argmin(diffs))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return ax


def plot_det(fpr: np.ndarray, fnr: np.ndarray, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, fnr, label="DET")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    ax.legend()
    return ax


def save_metrics(metrics: dict, out_path: str | Path):
    """
    Save scalar metrics to either JSON or CSV, based on file extension.

    - Case-insensitive lookup of 'auc', 'eer', 'eer_threshold'.
    - If out_path ends with '.json', write JSON; else write CSV.
    """
    out_path = Path(out_path)

    # helper to pull values case-insensitively
    def get(key: str):
        for k, v in metrics.items():
            if k.lower() == key.lower():
                return v
        raise KeyError(f"Metric '{key}' not found in {list(metrics.keys())}")

    data = {
        "auc":           get("auc"),
        "eer":           get("eer"),
        "eer_threshold": get("eer_threshold"),
    }

    if out_path.suffix.lower() == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        df = pd.DataFrame({
            "metric": list(data.keys()),
            "value":  list(data.values()),
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
