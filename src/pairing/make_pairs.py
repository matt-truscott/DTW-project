"""Generate labelled pairs of signature file paths for verification models."""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd

from src.io.load_biosecurid import parse_filename

GENUINE_SAMPLES = {1, 2, 6, 7}
FORGERY_SAMPLES = {3, 4, 5}


def _ensure_metadata(df: pd.DataFrame, path_col: str) -> pd.DataFrame:
    """Ensure user/session/sample columns exist by parsing filenames."""
    if {"user_id", "session_id", "sample_id"}.issubset(df.columns):
        return df

    def _parse(row_path: str, idx: int) -> int:
        return parse_filename(Path(row_path).name)[idx]

    df = df.copy()
    df["user_id"] = df[path_col].apply(lambda p: _parse(p, 0))
    df["session_id"] = df[path_col].apply(lambda p: _parse(p, 1))
    df["sample_id"] = df[path_col].apply(lambda p: _parse(p, 2))
    return df


def _pair_record(row_a, row_b, label: int, path_col: str) -> dict:
    return {
        "userA": int(row_a.user_id),
        "sessionA": int(row_a.session_id),
        "sampleA": int(row_a.sample_id),
        "pathA": row_a[path_col],
        "userB": int(row_b.user_id),
        "sessionB": int(row_b.session_id),
        "sampleB": int(row_b.sample_id),
        "pathB": row_b[path_col],
        "label": int(label),
    }


def generate_pairs(df: pd.DataFrame, *, weak_forgery: bool = False, path_col: str = "local_path") -> pd.DataFrame:
    """Generate labelled signature pairs from a catalog DataFrame."""
    df = _ensure_metadata(df, path_col)
    records: List[dict] = []

    # group by user for efficiency
    for user_id, user_df in df.groupby("user_id"):
        genuine = user_df[user_df["sample_id"].isin(GENUINE_SAMPLES)]

        # Genuine pairs (across sessions as well)
        for row_a, row_b in combinations(genuine.itertuples(index=False), 2):
            records.append(_pair_record(row_a, row_b, 1, path_col))

        # Skilled forgeries per session
        for session_id, sess_df in user_df.groupby("session_id"):
            gens = sess_df[sess_df["sample_id"].isin(GENUINE_SAMPLES)]
            forgs = sess_df[sess_df["sample_id"].isin(FORGERY_SAMPLES)]
            for row_g in gens.itertuples(index=False):
                for row_f in forgs.itertuples(index=False):
                    records.append(_pair_record(row_g, row_f, 0, path_col))

    if weak_forgery:
        # cross-user genuine pairs labelled as forgeries
        users = list(df.groupby("user_id").groups.keys())
        for i, user_a in enumerate(users):
            df_a = df[(df["user_id"] == user_a) & df["sample_id"].isin(GENUINE_SAMPLES)]
            for user_b in users[i + 1 :]:
                df_b = df[(df["user_id"] == user_b) & df["sample_id"].isin(GENUINE_SAMPLES)]
                for row_a in df_a.itertuples(index=False):
                    for row_b in df_b.itertuples(index=False):
                        records.append(_pair_record(row_a, row_b, 0, path_col))

    return pd.DataFrame.from_records(records)


def balance_subsample(pairs: pd.DataFrame, *, max_pairs: int = 200_000, random_state: int | None = None) -> pd.DataFrame:
    """Balance classes and optionally subsample to a maximum size."""
    rng = np.random.default_rng(random_state)
    g = pairs[pairs.label == 1]
    f = pairs[pairs.label == 0]
    n = min(len(g), len(f))
    if 2 * n > max_pairs:
        n = max_pairs // 2
    g = g.sample(n=n, random_state=rng.integers(0, 1e9)) if len(g) > n else g
    f = f.sample(n=n, random_state=rng.integers(0, 1e9)) if len(f) > n else f
    result = pd.concat([g, f], ignore_index=True)
    result = result.sample(frac=1.0, random_state=rng.integers(0, 1e9)).reset_index(drop=True)
    return result


def main(catalog_path: Path, output_path: Path, *, weak_forgery: bool = False, max_pairs: int = 200_000) -> None:
    catalog = pd.read_parquet(catalog_path)
    path_col = "local_path" if "local_path" in catalog.columns else catalog.columns[-1]
    pairs = generate_pairs(catalog, weak_forgery=weak_forgery, path_col=path_col)
    balanced = balance_subsample(pairs, max_pairs=max_pairs)
    balanced.to_parquet(output_path)
    print(f"Wrote {len(balanced)} pairs to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate signature pairs")
    parser.add_argument("catalog", type=Path, help="Path to biosecurid_catalog.parquet")
    parser.add_argument("output", type=Path, help="Destination pairs_meta.parquet")
    parser.add_argument("--weak-forgery", action="store_true", help="Include cross-user genuine pairs as forgeries")
    parser.add_argument("--max-pairs", type=int, default=200_000, help="Maximum number of pairs after subsampling")
    args = parser.parse_args()

    main(args.catalog, args.output, weak_forgery=args.weak_forgery, max_pairs=args.max_pairs)
