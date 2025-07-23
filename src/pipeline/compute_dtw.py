"""CLI helper for building the DTW cache."""
from __future__ import annotations

from pathlib import Path
import argparse

from ..dtw.compute_dtw import build_cache


def main():  # pragma: no cover - CLI wrapper
    ap = argparse.ArgumentParser(description="Compute DTW distances")
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog.parquet"))
    ap.add_argument("--cache", type=Path, default=Path("cache/dtw.parquet"))
    ap.add_argument("--backend", choices=["cuda", "dtaidistance", "python"], default=None)
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    build_cache(args.pairs, args.catalog, args.cache, backend=args.backend, window=args.window)


if __name__ == "__main__":
    main()
