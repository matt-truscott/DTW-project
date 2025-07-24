"""Utilities for computing DTW distances and caching results."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import pyarrow.parquet as pq

try:
    from dtaidistance import dtw_ndim
    _HAVE_DTAI = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_DTAI = False

try:  # pragma: no cover - optional dependency
    import dtw_python_cuda as _dtw_cuda  # type: ignore
    _HAVE_CUDA = True
except Exception:  # pragma: no cover - optional dependency
    _HAVE_CUDA = False

from .core import dp
from ..io.load_biosecurid import load_local


Backend = str


def _select_backend(preferred: Backend | None = None) -> Backend:
    """Select an available DTW backend."""
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
    """Compute raw and bounded DTW distances for two feature matrices.

    Parameters
    ----------
    a, b : ndarray shape (n_samples, n_features)
        Signature feature matrices.
    backend : {{'cuda', 'dtaidistance', 'python', None}}, optional
        Which DTW backend to use. ``None`` selects the best available option.

    Returns
    -------
    d_raw : float
        Unconstrained DTW cost.
    d_bound : float
        DTW cost using a Sakoe-Chiba window of ``window``.
    path_len : int
        Length of the optimal unconstrained warping path.
    len_a : int
        Length of the reference sequence.
    len_b : int
        Length of the query sequence.
    """
    backend = _select_backend(backend)

    if backend == "cuda":  # pragma: no cover - requires optional dependency
        dist, cost = _dtw_cuda.warping_paths(a.astype(float), b.astype(float))
        path = _dtw_cuda.warping_path(a.astype(float), b.astype(float))
        d_raw = float(cost[-1, -1])
        path_len = len(path)
        # bounded distance not implemented for CUDA backend
        _, cost_b = _dtw_cuda.warping_paths(a.astype(float), b.astype(float), window=window)
        d_bound = float(cost_b[-1, -1])
    elif backend == "dtaidistance":  # pragma: no cover - requires optional dependency
        x = a.astype(float).T
        y = b.astype(float).T
        _, cost = dtw_ndim.warping_paths_fast(x, y)
        path = dtw_ndim.warping_path(x, y)
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

    n_a = len(a)
    n_b = len(b)
    return d_raw, d_bound, int(path_len), int(n_a), int(n_b)


def _bounded_dtw(dist_mat: np.ndarray, window: int) -> float:
    """Simple Sakoe-Chiba band DTW on a precomputed distance matrix."""
    n, m = dist_mat.shape
    window = max(window, abs(n - m))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            d = dist_mat[i - 1, j - 1]
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])


def _append_records(records: Iterable[dict], cache_path: Path) -> None:
    """Append records to a Parquet cache."""
    df = pd.DataFrame.from_records(records)
    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(cache_path, index=False)


def build_cache(
    pairs_path: Path,
    catalog_path: Path,
    cache_path: Path,
    *,
    chunk_size: int = 10_000,
    backend: Backend | None = None,
    window: int = 10,
) -> None:
    """Compute DTW distances for all pairs in ``pairs_path``.

    Existing results in ``cache_path`` are reused and only missing ``pair_id``
    rows are processed.
    """
    cat = pd.read_parquet(catalog_path)
    lookup = cat.set_index(["user_id", "sample_id"])["local_path"].to_dict()

    computed: set[int] = set()
    if cache_path.exists():
        computed = set(pd.read_parquet(cache_path)["pair_id"].tolist())

    pf = pq.ParquetFile(pairs_path)
    records: list[dict] = []
    offset = 0
    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        df["pair_id"] = np.arange(offset, offset + len(df))
        offset += len(df)
        for row in df.itertuples(index=False):
            if row.pair_id in computed:
                continue
            path_a = Path(lookup[(row.userA, row.sigA)])
            path_b = Path(lookup[(row.userB, row.sigB)])
            a = load_local(path_a)
            b = load_local(path_b)
            d_raw, d_bound, plen, la, lb = compute_pair_dtw(
                a, b, backend=backend, window=window
            )
            records.append(
                {
                    "pair_id": int(row.pair_id),
                    "label": int(row.label),
                    "d_raw": d_raw,
                    "d_bound": d_bound,
                    "path_len": int(plen),
                    "len_ref": int(la),
                    "len_qry": int(lb),
                }
            )
        if len(records) >= chunk_size:
            _append_records(records, cache_path)
            computed.update(r["pair_id"] for r in records)
            records.clear()
    if records:
        _append_records(records, cache_path)


if __name__ == "__main__":  # pragma: no cover - CLI helper
    import argparse
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description="Compute DTW cache")
    ap.add_argument("pairs", type=Path)
    ap.add_argument("catalog", type=Path)
    ap.add_argument("cache", type=Path)
    ap.add_argument("--chunk-size", type=int, default=10_000)
    ap.add_argument(
        "--backend",
        choices=["cuda", "dtaidistance", "python"],
        default=None,
        help="Preferred DTW backend",
    )
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    build_cache(
        args.pairs,
        args.catalog,
        args.cache,
        chunk_size=args.chunk_size,
        backend=args.backend,
        window=args.window,
    )
