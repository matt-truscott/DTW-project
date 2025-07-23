"""Wrapper CLI for generating labelled pairs."""
from __future__ import annotations

from pathlib import Path
import argparse

from ..pairing.make_pairs import main as _main


def main():  # pragma: no cover - simple CLI
    ap = argparse.ArgumentParser(description="Generate labelled BiosecurID pairs")
    ap.add_argument("--out", type=Path, default=Path("data/pairs.parquet"))
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    _main(args.catalog, args.out, max_pairs=200_000)


if __name__ == "__main__":
    main()
