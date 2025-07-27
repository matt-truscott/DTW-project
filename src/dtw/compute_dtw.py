"""
Utilities for computing DTW distances and caching results.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pyarrow.parquet as pq

# allow running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# optional high-perf backends
try:
    from dtaidistance import dtw_ndim
    _HAVE_DTAI = True
except ImportError:
    _HAVE_DTAI = False

try:
    import dtw_python_cuda as _dtw_cuda  # type: ignore
    _HAVE_CUDA = True
except ImportError:
    _HAVE_CUDA = False

from src.dtw.dtwAlgorithm import dp
from src.io.load_biosecurid import load_local


Backend = str


def _select_backend(preferred: Backend | None = None) -> Backend:
    if preferred == "cuda" and _HAVE_CUDA:
        return "cuda"
    if preferred in {"dtaidistance", None} and _HAVE_DTAI:
        return "dtaidistance"
    return "python"


def compute_pair_dtw(
    a: np.ndarray,
    b: np.ndarray,
    *,
    backend: Backend | None = None,
    window: int = 10,
) -> Tuple[float, float, int, int, int]:
    backend = _select_backend(backend)

    if backend == "cuda":
        _, cost = _dtw_cuda.warping_paths(a.astype(float), b.astype(float))
        path  = _dtw_cuda.warping_path(a.astype(float), b.astype(float))
        d_raw = float(cost[-1, -1])
        path_len = len(path)
        _, cost_b = _dtw_cuda.warping_paths(a.astype(float), b.astype(float), window=window)
        d_bound = float(cost_b[-1, -1])

    elif backend == "dtaidistance":
        x = a.astype(float)
        y = b.astype(float)
        _, cost = dtw_ndim.warping_paths_fast(x, y)   # no use_ndim here
        path  = dtw_ndim.warping_path(x, y)
        d_raw = float(cost[-1, -1])
        path_len = len(path)
        _, cost_b = dtw_ndim.warping_paths_fast(x, y, window=window)
        d_bound = float(cost_b[-1, -1])

    else:
        dist_mat = cdist(a, b)
        path, cost = dp(dist_mat)
        d_raw = float(cost[-1, -1])
        path_len = len(path)
        d_bound = _bounded_dtw(dist_mat, window)

    return d_raw, d_bound, path_len, len(a), len(b)


def _bounded_dtw(dist_mat: np.ndarray, window: int) -> float:
    """Sakoe–Chiba‐band DTW on a precomputed distance matrix."""
    n, m = dist_mat.shape
    window = max(window, abs(n - m))
    cost = np.full((n+1, m+1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n+1):
        j0 = max(1, i-window)
        j1 = min(m, i+window)
        for j in range(j0, j1+1):
            d = dist_mat[i-1, j-1]
            cost[i, j] = d + min(
                cost[i-1, j],   # insertion
                cost[i, j-1],   # deletion
                cost[i-1, j-1]  # match
            )

    return float(cost[n, m])


def _append_records(records: list[dict], cache_path: Path) -> None:
    df = pd.DataFrame.from_records(records)
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(cache_path, index=False)


def build_cache(
    pairs_path: Path,
    cache_path: Path,
    *,
    chunk_size: int = 10_000,
    backend:    Backend | None = None,
    window:     int = 10,
) -> None:
    """
    Compute DTW distances for all pairs in `pairs_path`.
    Expects columns [pair_id, ..., pathA, pathB, label].
    Appends to `cache_path`, skipping existing pair_id.
    """
    pairs_df = pd.read_parquet(pairs_path)
    if "pair_id" not in pairs_df.columns:
        pairs_df["pair_id"] = np.arange(len(pairs_df))

    done = set()
    if cache_path.exists():
        done = set(pd.read_parquet(cache_path)["pair_id"].tolist())

    records: list[dict] = []
    for _, row in pairs_df.iterrows():
        pid = int(row.pair_id)
        if pid in done:
            continue

        # load the two sequences
        a = load_local(Path(row.pathA))
        b = load_local(Path(row.pathB))

        d_raw, d_bound, plen, la, lb = compute_pair_dtw(
            a, b, backend=backend, window=window
        )

        records.append({
            "pair_id":  pid,
            "label":    int(row.label),
            "d_raw":    d_raw,
            "d_bound":  d_bound,
            "path_len": plen,
            "len_ref":  la,
            "len_qry":  lb,
        })

        if len(records) >= chunk_size:
            _append_records(records, cache_path)
            done.update(r["pair_id"] for r in records)
            records.clear()

    if records:
        _append_records(records, cache_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compute DTW cache")
    ap.add_argument("pairs",  type=Path)
    ap.add_argument("cache",  type=Path)
    ap.add_argument("--chunk-size", type=int,   default=10_000)
    ap.add_argument(
        "--backend",
        choices=["cuda","dtaidistance","python"],
        default=None
    )
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    build_cache(
        pairs_path = args.pairs,
        cache_path = args.cache,
        chunk_size = args.chunk_size,
        backend    = args.backend,
        window     = args.window,
    )