# src/nn_utils/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Set

import json
import numpy as np
import pandas as pd

# Light deps from your project
from src.io.load_biosecurid import load_local
from src.nn_utils.data import resample_sequence

# Optional: custom layer for loading the saved Siamese model
try:
    from tensorflow.keras.models import load_model as _keras_load_model  # type: ignore
    try:
        from src.keras_layers.diff_dtw import DiffDTW  # type: ignore
        _CUSTOMS = {"DiffDTW": DiffDTW}
    except Exception:
        _CUSTOMS = {}
except Exception as _e:
    _keras_load_model = None  # type: ignore


__all__ = [
    "load_siamese_model",
    "score_split",
    "score_both_splits",
    "make_per_scenario_outputs",
]


# ---------------------------------------------------------------------
# Small utilities (Path handling, coercions, caching)
# ---------------------------------------------------------------------

def _resolve_path(p: str | Path, project_root: Optional[Path] = None) -> Path:
    """
    Return an absolute Path. If `p` is relative, resolve it against `project_root`
    (if provided) else against the current working directory.
    """
    pp = Path(p) if not isinstance(p, Path) else p
    if pp.is_absolute():
        return pp
    base = project_root if project_root is not None else Path.cwd()
    return (base / pp).resolve()


def _coerce_refs(x) -> List[str]:
    """
    Accept list/tuple/ndarray or a JSON-encoded string of paths.
    Returns a Python list[str].
    """
    if isinstance(x, list):
        return [str(p) for p in x]
    if isinstance(x, tuple):
        return [str(p) for p in x]
    if isinstance(x, np.ndarray):
        return [str(p) for p in x.tolist()]
    if isinstance(x, (bytes, str)):
        # Sometimes parquet stores lists as JSON strings
        try:
            obj = json.loads(x)  # type: ignore[arg-type]
            if isinstance(obj, list):
                return [str(p) for p in obj]
        except Exception:
            # fall through — handled below
            pass
    raise TypeError(f"Unexpected type for path_lf_refs (got {type(x)})")


def _preload_resampled(paths: Iterable[Path], seq_len: int) -> Dict[Path, np.ndarray]:
    """
    Load LocalFunctions matrices once and resample to (seq_len, 9).
    Keys and lookups are done strictly with Path objects.
    """
    cache: Dict[Path, np.ndarray] = {}
    for p in paths:
        arr = load_local(p)  # (T, 9), float
        cache[p] = resample_sequence(arr, seq_len)  # (seq_len, 9), float32
    return cache


@dataclass(frozen=True)
class _IndexRec:
    pair_id: int
    query_label: str
    scenario: str
    n_refs: int = 4


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def load_siamese_model(model_path: Path):
    """
    Load a saved Keras model (expects the DiffDTW custom layer if it was used).
    """
    if _keras_load_model is None:
        raise ImportError("TensorFlow / Keras is not available in this environment.")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return _keras_load_model(str(model_path), custom_objects=_CUSTOMS or None)


def score_split(
    proc_root: Path,
    split: str,
    *,
    model,
    seq_len: int = 100,
    batch_size: int = 32,
    project_root: Optional[Path] = None,
    out_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Compute Siamese probabilities (mean/min/median over 4 refs) for all rows in
    `processed/pairs_{split}.parquet` and write `siam_cache_{split}.parquet`.

    Returns the path to the written parquet.
    """
    proc_root = Path(proc_root)
    project_root = Path(project_root) if project_root is not None else proc_root.parents[0]
    pairs_path = proc_root / f"pairs_{split}.parquet"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs file: {pairs_path}")

    out_path = out_path or (proc_root / f"siam_cache_{split}.parquet")
    if out_path.exists() and not overwrite:
        return out_path

    # Load and normalize columns
    df = pd.read_parquet(pairs_path, engine="pyarrow").copy()
    if "pair_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "pair_id"})

    must = {"pair_id", "path_lf_query", "path_lf_refs", "query_label"}
    missing = must - set(df.columns)
    if missing:
        raise KeyError(f"pairs_{split}.parquet missing columns: {sorted(missing)}")

    # Ensure list[str] in the refs column
    df["path_lf_refs"] = df["path_lf_refs"].map(_coerce_refs)

    # --------- Preload/resample once per unique path ---------
    uniq_paths: Set[Path] = set()
    for r in df.itertuples(index=False):
        # query
        uniq_paths.add(_resolve_path(getattr(r, "path_lf_query"), project_root))
        # refs (already list[str])
        for rp in getattr(r, "path_lf_refs"):
            uniq_paths.add(_resolve_path(rp, project_root))

    cache = _preload_resampled(uniq_paths, seq_len)

    # --------- Batch predict (4 pairs per row) ---------
    results: List[Dict[str, object]] = []
    chunk_q: List[np.ndarray] = []
    chunk_r: List[np.ndarray] = []
    indices: List[_IndexRec] = []

    def _flush():
        if not indices:
            return
        q_batch = np.stack(chunk_q, dtype=np.float32)
        r_batch = np.stack(chunk_r, dtype=np.float32)
        probs = model.predict([q_batch, r_batch], batch_size=batch_size, verbose=0)[:, 1]
        k: int = 0
        for rec in indices:
            pid = int(rec.pair_id)
            n = int(rec.n_refs)  # always 4 in our protocol
            vals = probs[k:k + n]
            k = k + n  # explicit int math (Pylance-friendly)

            if len(vals) != 4:
                raise RuntimeError(f"Expected 4 predictions per pair_id={pid}, got {len(vals)}")

            results.append({
                "pair_id": pid,
                "query_label": str(rec.query_label),
                "scenario": str(rec.scenario),
                "split": str(split),
                "siam_ref1": float(vals[0]),
                "siam_ref2": float(vals[1]),
                "siam_ref3": float(vals[2]),
                "siam_ref4": float(vals[3]),
                "siam_mean": float(np.mean(vals)),
                "siam_min":  float(np.min(vals)),
                "siam_median": float(np.median(vals)),
            })
        # clear containers
        chunk_q.clear()
        chunk_r.clear()
        indices.clear()

    # Build 4 comparisons per row (q vs each ref)
    for r in df.itertuples(index=False):
        q_path = _resolve_path(getattr(r, "path_lf_query"), project_root)
        refs: List[str] = getattr(r, "path_lf_refs")
        rpaths: List[Path] = [_resolve_path(p, project_root) for p in refs]

        q_seq = cache[q_path]
        for refp in rpaths:
            chunk_q.append(q_seq)
            chunk_r.append(cache[refp])

        indices.append(
            _IndexRec(
                pair_id=int(getattr(r, "pair_id")),
                query_label=str(getattr(r, "query_label")),
                scenario=str(getattr(r, "scenario", "skilled_random")),
                n_refs=4,
            )
        )

        # Optional: flush in chunks to bound memory (here every ~4096 pairs)
        if len(chunk_q) >= 4096:
            _flush()

    _flush()

    out_df = pd.DataFrame.from_records(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False, engine="pyarrow")
    return out_path


def score_both_splits(
    proc_root: Path,
    *,
    model,
    seq_len: int = 100,
    batch_size: int = 32,
    project_root: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    Convenience wrapper to produce siam_cache_{dev,test}.parquet.
    """
    paths = {}
    for split in ("dev", "test"):
        paths[split] = score_split(
            proc_root,
            split,
            model=model,
            seq_len=seq_len,
            batch_size=batch_size,
            project_root=project_root,
            overwrite=overwrite,
        )
    return paths


# ---------------------------------------------------------------------
# Per-scenario CSVs and summary (so the notebook stays tiny)
# ---------------------------------------------------------------------

_SCENARIO_VIEWS = {
    "skilled_only": ["genuine", "skilled"],
    "random_only":  ["genuine", "random"],
    "skilled_random": ["genuine", "skilled", "random"],
}

def _subset_for_view(df: pd.DataFrame, view: str) -> pd.DataFrame:
    keep = _SCENARIO_VIEWS[view]
    out = df[df["query_label"].isin(keep)].copy()
    out["scenario"] = view
    return out

def _roc(y_true: np.ndarray, y_score: np.ndarray):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))
    return fpr, tpr, thr, auc

def _eer_from_roc(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> Tuple[float, float]:
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float(thr[i]), float((fnr[i] + fpr[i]) / 2.0)

def _tpr_at(fpr: np.ndarray, tpr: np.ndarray, xs=(0.01, 0.05)) -> Dict[str, float]:
    return {f"TPR@FPR={x:.2f}": float(np.interp(x, fpr, tpr)) for x in xs}

def make_per_scenario_outputs(
    proc_root: Path,
    figures_dir: Path,
    metrics_dir: Path,
    *,
    score_col: str = "siam_mean",
) -> pd.DataFrame:
    """
    Merge pairs + siam_cache for dev/test, write the spec'd per-scenario CSVs,
    draw ROC/DET for test, and return a summary DataFrame.
    """
    proc_root = Path(proc_root)
    figures_dir = Path(figures_dir); figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(metrics_dir); metrics_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    pairs_dev  = pd.read_parquet(proc_root / "pairs_dev.parquet",  engine="pyarrow")
    pairs_test = pd.read_parquet(proc_root / "pairs_test.parquet", engine="pyarrow")
    siam_dev   = pd.read_parquet(proc_root / "siam_cache_dev.parquet",  engine="pyarrow")
    siam_test  = pd.read_parquet(proc_root / "siam_cache_test.parquet", engine="pyarrow")

    dev  = pairs_dev.merge(siam_dev,  on="pair_id", how="inner", suffixes=("", "_siam"))
    test = pairs_test.merge(siam_test, on="pair_id", how="inner", suffixes=("", "_siam"))

    # Standard score/label
    for df in (dev, test):
        df["score"] = df[score_col].astype(float)
        df["label"] = (df["query_label"] == "genuine").astype(int)

    def _write_csv(df: pd.DataFrame, split: str, view: str) -> Path:
        # Keep the exact columns requested by the spec
        cols = [
            "user", "scenario", "query_user", "query_session", "query_attempt",
            "ref_attempts", "siam_mean", "siam_min", "siam_median", "pair_id",
        ]
        safe = {c for c in cols if c in df.columns}
        out = df[list(safe)].copy()
        out.rename(columns={
            "siam_mean": "siam_score_mean",
            "siam_min": "siam_score_min",
            "siam_median": "siam_score_median",
        }, inplace=True)
        csv_path = metrics_dir / f"siamese_{view}_{split}.csv"
        out.to_csv(csv_path, index=False)
        return csv_path

    def _plot_roc_det(fpr, tpr, auc, title_prefix, out_prefix) -> Tuple[Path, Path]:
        import matplotlib.pyplot as plt
        # ROC
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right"); ax.set_title(f"{title_prefix} ROC")
        rp = figures_dir / f"{out_prefix}_roc.png"
        fig.tight_layout(); fig.savefig(rp, dpi=250); plt.close(fig)
        # DET
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.plot(fpr, 1 - tpr, label="DET")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("False Negative Rate")
        ax.legend(); ax.set_title(f"{title_prefix} DET")
        dp = figures_dir / f"{out_prefix}_det.png"
        fig.tight_layout(); fig.savefig(dp, dpi=250); plt.close(fig)
        return rp, dp

    summary_rows: List[Dict[str, object]] = []

    for view in ("skilled_only", "random_only", "skilled_random"):
        dev_v  = _subset_for_view(dev, view)
        test_v = _subset_for_view(test, view)

        csv_dev  = _write_csv(dev_v,  "dev",  view)
        csv_test = _write_csv(test_v, "test", view)

        # Dev ROC + threshold (choose here)
        fpr_d, tpr_d, thr_d, auc_d = _roc(dev_v["label"].to_numpy(int), dev_v["score"].to_numpy(float))
        thr_dev, eer_dev = _eer_from_roc(fpr_d, tpr_d, thr_d)
        tprs_d = _tpr_at(fpr_d, tpr_d)

        # Test ROC + EER
        fpr_t, tpr_t, thr_t, auc_t = _roc(test_v["label"].to_numpy(int), test_v["score"].to_numpy(float))
        _, eer_t = _eer_from_roc(fpr_t, tpr_t, thr_t)
        tprs_t = _tpr_at(fpr_t, tpr_t)

        # APCER/BPCER on test at dev threshold
        y_true = test_v["label"].to_numpy(int)
        y_pred = (test_v["score"].to_numpy(float) >= thr_dev).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        apcer = fp / max(fp + tn, 1)   # FPR
        bpcer = fn / max(fn + tp, 1)   # FNR

        roc_path, det_path = _plot_roc_det(
            fpr_t, tpr_t, auc_t,
            title_prefix=f"Siamese — {view} (TEST)",
            out_prefix=f"siam_{view}_test"
        )

        summary_rows.append({
            "scenario": view,
            "auc_dev":  float(auc_d),
            "eer_dev":  float(eer_dev),
            "thr_dev":  float(thr_dev),
            "auc_test": float(auc_t),
            "eer_test": float(eer_t),
            "apcer_test": float(apcer),
            "bpcer_test": float(bpcer),
            "TPR@FPR=0.01_dev": float(tprs_d["TPR@FPR=0.01"]),
            "TPR@FPR=0.05_dev": float(tprs_d["TPR@FPR=0.05"]),
            "TPR@FPR=0.01_test": float(tprs_t["TPR@FPR=0.01"]),
            "TPR@FPR=0.05_test": float(tprs_t["TPR@FPR=0.05"]),
            "roc_test_png": str(roc_path),
            "det_test_png": str(det_path),
            "csv_dev": str(csv_dev),
            "csv_test": str(csv_test),
        })

    summary = pd.DataFrame(summary_rows)
    (metrics_dir / "siamese_summary.json").write_text(
        json.dumps(json.loads(summary.to_json(orient="records")), indent=2)
    )
    return summary
