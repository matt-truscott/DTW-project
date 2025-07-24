"""Build a BiosecurID catalog of feature file paths."""
from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import pandas as pd

from ..io.load_biosecurid import parse_filename


def build_catalog(
    *,
    global_dir: Path | None = None,
    local_dir: Path | None = None,
    config: Path | None = None,
    out_path: Path = Path("data/catalog.parquet"),
) -> pd.DataFrame:
    """Scan feature directories and write a catalog Parquet file."""
    if config is not None:
        with open(config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        global_dir = Path(cfg["GlobalFeatures"]).expanduser()
        local_dir = Path(cfg["LocalFunctions"]).expanduser()
    if global_dir is None or local_dir is None:
        raise ValueError("global_dir and local_dir must be provided")

    records: list[dict] = []
    missing: list[str] = []
    for gfile in sorted(Path(global_dir).glob("u*.mat")):
        try:
            user, sess, samp = parse_filename(gfile.name)
        except ValueError:
            continue
        lfile = Path(local_dir) / gfile.name
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    if missing:
        print(f"Warning: {len(missing)} local files missing")
    print(f"\u2713 wrote catalog with {len(df)} entries \u2192 {out_path}")
    return df


def main() -> None:  # pragma: no cover - CLI helper
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


if __name__ == "__main__":  # pragma: no cover
    main()
