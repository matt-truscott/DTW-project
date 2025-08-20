# src/dtw/compute_dtw.py
from __future__ import annotations

# ---- bootstrap: ensure 'src' importable in spawned workers; cap BLAS threads ----
import os, sys, json
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import math
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist

try:
    from dtaidistance import dtw_ndim
    _HAVE_DTAI = True
except Exception:
    _HAVE_DTAI = False

try:
    import dtw_python_cuda as _dtw_cuda  # type: ignore
    _HAVE_CUDA = True
except Exception:
    _HAVE_CUDA = False

from src.dtw.dtwAlgorithm import dp
from src.io.load_biosecurid import load_local

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

Backend = str
__all__ = ["build_cache", "compute_pair_dtw"]

# -----------------------------------------------------------------------------


def _resolve_project_path(rel: str) -> Path:
    return (PROJECT_ROOT / Path(str(rel).replace("\\", "/"))).resolve()

def _select_backend(preferred: Backend | None = None) -> Backend:
    if preferred == "cuda" and _HAVE_CUDA:
        return "cuda"
    if preferred in {"dtaidistance", None} and _HAVE_DTAI:
        return "dtaidistance"
    return "python"

def _bounded_dtw(dist_mat: np.ndarray, window: int) -> float:
    n, m = dist_mat.shape
    window = max(window, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j0 = max(1, i - window)
        j1 = min(m, i + window)
        row = dist_mat[i - 1]
        ci = cost[i]
        cim1 = cost[i - 1]
        for j in range(j0, j1 + 1):
            d = row[j - 1]
            ci[j] = d + min(cim1[j], ci[j - 1], cim1[j - 1])
    return float(cost[n, m])

def compute_pair_dtw(
    a: np.ndarray, b: np.ndarray, *, backend: Backend | None = None, window: int = 10
) -> Tuple[float, float, int, int, int]:
    backend = _select_backend(backend)

    if backend == "cuda":
        _, cost = _dtw_cuda.warping_paths(a.astype(float), b.astype(float))
        path = _dtw_cuda.warping_path(a.astype(float), b.astype(float))
        d_raw = float(cost[-1, -1])
        path_len = len(path)
        _, cost_b = _dtw_cuda.warping_paths(a.astype(float), b.astype(float), window=window)
        d_bound = float(cost_b[-1, -1])

    elif backend == "dtaidistance":
        x = a.astype(float); y = b.astype(float)
        _, cost = dtw_ndim.warping_paths_fast(x, y)
        path = dtw_ndim.warping_path(x, y)
        d_raw = float(cost[-1, -1]); path_len = len(path)
        _, cost_b = dtw_ndim.warping_paths_fast(x, y, window=window)
        d_bound = float(cost_b[-1, -1])

    else:
        dist_mat = cdist(a, b)
        path, cost = dp(dist_mat)
        d_raw = float(cost[-1, -1]); path_len = len(path)
        d_bound = _bounded_dtw(dist_mat, window)

    return d_raw, d_bound, path_len, len(a), len(b)

# -----------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _load_local_cached(path: str) -> np.ndarray:
    abs_path = _resolve_project_path(path)
    if not abs_path.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {abs_path} (from '{path}')")
    return load_local(abs_path)

def _infer_split_case_from_path(p: Path) -> tuple[str, str]:
    """Infer (split, case) from a pairs file name like 'pairs_dev_genuine.parquet'."""
    stem = p.stem  # e.g., "pairs_dev_genuine"
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return "", ""

def _worker_pairwise(task: Dict[str, Any]) -> Dict[str, Any]:
    pid = int(task["pair_id"])
    a = _load_local_cached(task["pathA"])
    b = _load_local_cached(task["pathB"])
    d_raw, d_bound, plen, la, lb = compute_pair_dtw(a, b, backend=task["backend"], window=task["window"])

    # Invariant: bounded DTW should never be cheaper than raw
    if not (d_bound >= d_raw):
        raise AssertionError(f"Bounded DTW {d_bound:.6f} < raw DTW {d_raw:.6f} for pair_id={pid}")

    return {
        "pair_id": pid,
        "label": int(task["label"]),
        "d_raw": d_raw,
        "d_bound": d_bound,
        "path_len": plen,
        "len_A": la,
        "len_B": lb,
        "backend": task["backend"] or _select_backend(None),
        "window": int(task["window"]),
        "mode": "pairwise",
        "split": task.get("split", ""),
        "case": task.get("case", ""),
    }

def _worker_q2refs(task: Dict[str, Any]) -> Dict[str, Any]:
    pid = int(task["pair_id"])
    q = _load_local_cached(task["q"])
    dists: List[float] = []; lens_ref: List[int] = []
    for rp in task["refs"]:
        r = _load_local_cached(rp)
        _, d_b, _, _, lr = compute_pair_dtw(q, r, backend=task["backend"], window=task["window"])
        dists.append(d_b); lens_ref.append(lr)
    return {
        "pair_id": pid,
        "query_label": task["query_label"],
        "scenario": task.get("scenario", "skilled_random"),
        "split": task.get("split", ""),
        "d_ref1": dists[0], "d_ref2": dists[1], "d_ref3": dists[2], "d_ref4": dists[3],
        "d_mean": float(np.mean(dists)), "d_min": float(np.min(dists)), "d_median": float(np.median(dists)),
        "len_q": len(q),
        "len_r1": lens_ref[0], "len_r2": lens_ref[1], "len_r3": lens_ref[2], "len_r4": lens_ref[3],
        "backend": task["backend"] or _select_backend(None),
        "window": int(task["window"]),
        "mode": "q2refs",
    }

# -----------------------------------------------------------------------------


def build_cache(
    pairs_path: Path,
    cache_path: Path,
    *,
    backend: Backend | None = None,
    window: int = 10,
    n_jobs: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    progress_every: int = 500,
) -> None:
    """
    Auto-detect pairs format and compute cache.
      - Legacy pairwise → [pair_id, label, d_raw, d_bound, ...]
      - Query→refs      → [pair_id, query_label, d_ref1..4, d_mean, d_min, d_median, ...]
    Skips pair_ids already present in cache unless overwrite=True.
    """
    pairs_path = Path(pairs_path)
    cache_path = Path(cache_path)

    df = pd.read_parquet(pairs_path, engine="pyarrow")

    # Normalize column names FIRST so detection & checks are consistent
    if {"path_ref", "path_query"}.issubset(df.columns):
        df = df.rename(columns={"path_ref": "pathA", "path_query": "pathB"})

    has_pairwise = {"pathA", "pathB"}.issubset(df.columns)
    has_q2refs   = {"path_lf_query", "path_lf_refs"}.issubset(df.columns)
    if not (has_pairwise or has_q2refs):
        raise ValueError(
            "Unrecognised pairs file: need either {path_ref,path_query}/{pathA,pathB} "
            "or {path_lf_query,path_lf_refs}."
        )

    # Infer split/case for pairwise files from filename; q2refs already carries split/scenario
    if has_pairwise:
        split_inf, case_inf = _infer_split_case_from_path(pairs_path)
    else:
        split_inf, case_inf = "", ""

    # Lightweight path existence check on a sample
    sample_idx = df.index if len(df) <= 100 else df.sample(100, random_state=0).index
    if has_pairwise:
        _cols = ["pathA", "pathB"]
        for col in _cols:
            for p in df.loc[sample_idx, col].astype(str):
                if not _resolve_project_path(p).exists():
                    raise FileNotFoundError(
                        f"Path in column '{col}' does not exist under {PROJECT_ROOT}: {p}"
                    )
    else:
        for p in df.loc[sample_idx, "path_lf_query"].astype(str):
            if not _resolve_project_path(p).exists():
                raise FileNotFoundError(f"Query path does not exist under {PROJECT_ROOT}: {p}")
        # refs is a list-like; check the first row's 4 refs thoroughly
        first_refs = df["path_lf_refs"].iloc[0]
        if isinstance(first_refs, (str, bytes)):
            try:
                first_refs = json.loads(first_refs)
            except Exception:
                pass
        for rp in list(first_refs)[:4]:
            if not _resolve_project_path(str(rp)).exists():
                raise FileNotFoundError(f"Ref path does not exist under {PROJECT_ROOT}: {rp}")

    # Ensure pair_id exists and is int
    if "pair_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "pair_id"})
    df["pair_id"] = df["pair_id"].astype("int64")

    # Skip already-cached
    if not overwrite and cache_path.exists():
        done = set(pd.read_parquet(cache_path, engine="pyarrow")["pair_id"].tolist())
        df = df[~df["pair_id"].isin(done)]
        if df.empty:
            print("Nothing to do; all pairs already cached.")
            return

    n_jobs = n_jobs or max(1, os.cpu_count() or 1)
    tasks: List[Dict[str, Any]] = []

    if has_pairwise:
        use = df[["pair_id", "pathA", "pathB", "label"]].copy()
        use["label"] = use["label"].astype("int64")
        use["pathA"] = use["pathA"].astype(str)
        use["pathB"] = use["pathB"].astype(str)

        for _, r in use.iterrows():
            tasks.append({
                "pair_id": int(r["pair_id"]),
                "pathA": str(r["pathA"]),
                "pathB": str(r["pathB"]),
                "label": int(r["label"]),
                "backend": backend,
                "window": window,
                "split": split_inf,
                "case": case_inf,
            })

        worker = _worker_pairwise
        out_cols_order = [
            "pair_id", "label", "d_raw", "d_bound", "path_len", "len_A", "len_B",
            "backend", "window", "mode", "split", "case"
        ]

    else:
        sort_cols = [c for c in ["user", "pair_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)

        use = df[["pair_id", "path_lf_query", "path_lf_refs", "query_label", "scenario", "split"]].copy()
        use["path_lf_query"] = use["path_lf_query"].astype(str)

        def _coerce_refs(x) -> List[str]:
            if isinstance(x, list):   return [str(p) for p in x]
            if isinstance(x, tuple):  return [str(p) for p in x]
            if isinstance(x, np.ndarray): return [str(p) for p in x.tolist()]
            if isinstance(x, (bytes, str)):
                try:
                    v = json.loads(x)
                    if isinstance(v, list):
                        return [str(p) for p in v]
                except Exception:
                    pass
            raise TypeError(f"Unexpected type for path_lf_refs: {type(x)}")

        use["path_lf_refs"] = use["path_lf_refs"].map(_coerce_refs)
        bad = use["path_lf_refs"].map(len) != 4
        if bad.any():
            raise ValueError(f"Rows without 4 refs: {bad.sum()}")

        for _, r in use.iterrows():
            tasks.append({
                "pair_id": int(r["pair_id"]),
                "q": str(r["path_lf_query"]),
                "refs": [str(p) for p in r["path_lf_refs"]],
                "query_label": str(r["query_label"]),
                "scenario": str(r["scenario"]),
                "split": str(r["split"]),
                "backend": backend,
                "window": window,
            })

        worker = _worker_q2refs
        out_cols_order = [
            "pair_id", "query_label", "scenario", "split",
            "d_ref1", "d_ref2", "d_ref3", "d_ref4",
            "d_mean", "d_min", "d_median",
            "len_q", "len_r1", "len_r2", "len_r3", "len_r4",
            "backend", "window", "mode"
        ]

    results: List[Dict[str, Any]] = []
    backend_name = backend or _select_backend(None)
    mode = "pairwise" if has_pairwise else "q2refs"

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        if show_progress and tqdm is not None:
            desc = f"DTW {mode} | backend={backend_name} | w={window} | procs={n_jobs}"
            for f in tqdm(as_completed(futures), total=len(futures), unit="pair", desc=desc, leave=True):
                results.append(f.result())
        else:
            for i, f in enumerate(as_completed(futures), start=1):
                results.append(f.result())
                if show_progress and (i % progress_every == 0):
                    print(f"[DTW {mode}] {i}/{len(futures)} done…")

    df_new = pd.DataFrame.from_records(results)[out_cols_order]

    if cache_path.exists() and not overwrite:
        df_all = pd.concat([pd.read_parquet(cache_path, engine="pyarrow"), df_new], ignore_index=True)
    else:
        df_all = df_new

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Use compression for smaller/faster I/O
    df_all.to_parquet(cache_path, index=False, engine="pyarrow", compression="zstd")
