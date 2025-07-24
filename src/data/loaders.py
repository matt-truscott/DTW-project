"""Legacy wrappers for building the BiosecurID catalog."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .catalog import build_catalog


def check_integrity(config_path: Path = Path("config.yaml")) -> pd.DataFrame:
    """Compatibility wrapper for the old CLI."""
    return build_catalog(config=config_path)


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Build BiosecurID catalog")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--global-dir", type=Path)
    ap.add_argument("--local-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/catalog.parquet"))
    args = ap.parse_args()

    build_catalog(
        global_dir=args.global_dir,
        local_dir=args.local_dir,
        config=args.config if not args.global_dir and not args.local_dir else None,
        out_path=args.out,
    )
