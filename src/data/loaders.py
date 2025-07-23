"""Data discovery utilities for BiosecurID."""
from __future__ import annotations

import yaml
from pathlib import Path
import pandas as pd

from ..io.load_biosecurid import parse_filename


def check_integrity(config_path: Path = Path("config.yaml")) -> pd.DataFrame:
    """Parse feature directories and build a catalog."""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    gdir = Path(cfg["GlobalFeatures"]).expanduser()
    ldir = Path(cfg["LocalFunctions"]).expanduser()

    records = []
    missing = []
    for gfile in sorted(gdir.glob("u*.mat")):
        try:
            user, sess, samp = parse_filename(gfile.name)
        except ValueError:
            continue
        lfile = ldir / gfile.name
        if not lfile.exists():
            missing.append(gfile.name)
            continue
        records.append(
            {
                "user_id": user,
                "session_id": sess,
                "sample_id": samp,
                "global_path": str(gfile.resolve()),
                "local_path": str(lfile.resolve()),
            }
        )
    df = pd.DataFrame.from_records(records)
    out_path = Path("data/catalog.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    if missing:
        print(f"Warning: {len(missing)} local files missing")
    print(f"✓ wrote catalog with {len(df)} entries → {out_path}")
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Check dataset integrity")
    ap.add_argument("command", choices=["check-integrity"], nargs="?")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = ap.parse_args()

    if args.command == "check-integrity":
        check_integrity(args.config)
