"""Generate labelled BiosecurID pairs (genuine/genuine, genuine/forgery)."""

from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import List, Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #

GENUINE_SAMPLES: set[int] = {1, 2, 6, 7}
FORGERY_SAMPLES: set[int] = {3, 4, 5}

# --------------------------------------------------------------------------- #
# 1. filename parser                                                          #
# --------------------------------------------------------------------------- #

def parse_filename(fname: str) -> tuple[int, int, int]:
    """
    Parse BiosecurID file name ``uXXXXsYYYY_sgZZZZ.*``

    Returns
    -------
    user_id, session_id, sample_id
    """
    import re
    m = re.match(r"u(\d{4})s(\d{4})_sg(\d{4})", fname)
    if not m:
        raise ValueError(f"Bad BiosecurID name: {fname!s}")
    return tuple(map(int, m.groups()))  # type: ignore[return-value]

# --------------------------------------------------------------------------- #
# 2. core pair generation                                                     #
# --------------------------------------------------------------------------- #

def _ensure_metadata(df: pd.DataFrame, path_col: str) -> pd.DataFrame:
    """Add user_id / session_id / sample_id if missing."""
    if {"user_id", "session_id", "sample_id"}.issubset(df.columns):
        return df

    vals = np.stack(df[path_col].map(lambda p: parse_filename(Path(p).name)))
    df = df.copy()
    df[["user_id", "session_id", "sample_id"]] = vals
    return df


def _pair_record(a, b, label: int, path_col: str) -> dict:
    # `a` and `b` are tuples in positional order (user, sess, samp, path, ...)
    return dict(
        userA=int(a[0]),
        sessionA=int(a[1]),
        sampleA=int(a[2]),
        pathA=a[path_col],
        userB=int(b[0]),
        sessionB=int(b[1]),
        sampleB=int(b[2]),
        pathB=b[path_col],
        label=int(label),
    )


def generate_pairs(
    df: pd.DataFrame,
    *,
    path_col: str = "local_path",
    weak_forgery: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with columns [pathA, pathB, label, …]."""
    df = _ensure_metadata(df, path_col)

    # we will iterate with .itertuples(name=None) → (user, sess, samp, path, …)
    cols_for_tuple = ["user_id", "session_id", "sample_id", path_col]
    tdf = df[cols_for_tuple]

    records: list[dict] = []

    for _, u_df in tdf.groupby("user_id"):
        genuine = u_df[u_df.sample_id.isin(GENUINE_SAMPLES)]

        # (A) genuine–genuine across sessions
        for a, b in combinations(genuine.itertuples(index=False, name=None), 2):
            records.append(_pair_record(a, b, 1, 3))  # path_col index = 3

        # (B) skilled forgeries within session
        for _, s_df in u_df.groupby("session_id"):
            gen = s_df[s_df.sample_id.isin(GENUINE_SAMPLES)]
            forg = s_df[s_df.sample_id.isin(FORGERY_SAMPLES)]
            for g in gen.itertuples(index=False, name=None):
                for f in forg.itertuples(index=False, name=None):
                    records.append(_pair_record(g, f, 0, 3))

    if weak_forgery:
        users = sorted(tdf.user_id.unique())
        for i, ua in enumerate(users[:-1]):
            ga = tdf[(tdf.user_id == ua) & tdf.sample_id.isin(GENUINE_SAMPLES)]
            for ub in users[i + 1 :]:
                gb = tdf[(tdf.user_id == ub) & tdf.sample_id.isin(GENUINE_SAMPLES)]
                for a in ga.itertuples(index=False, name=None):
                    for b in gb.itertuples(index=False, name=None):
                        records.append(_pair_record(a, b, 0, 3))

    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# 3. balance / subsample                                                      #
# --------------------------------------------------------------------------- #

def balance_subsample(
    pairs: pd.DataFrame,
    *,
    max_pairs: int = 200_000,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Downsample to a balanced ≤ `max_pairs` set."""
    rng = np.random.default_rng(random_state)
    g = pairs[pairs.label == 1]
    f = pairs[pairs.label == 0]

    n = min(len(g), len(f), max_pairs // 2)
    rs1, rs2, rs3 = rng.integers(0, 2**32, size=3)

    g = g.sample(n=n, random_state=int(rs1)) if len(g) > n else g
    f = f.sample(n=n, random_state=int(rs2)) if len(f) > n else f

    out = pd.concat([g, f]).sample(frac=1.0, random_state=int(rs3)).reset_index(drop=True)
    return out

# --------------------------------------------------------------------------- #
# 4. CLI                                                                     #
# --------------------------------------------------------------------------- #

def main(
    catalog_path: Path,
    output_path: Path,
    *,
    weak_forgery: bool = False,
    max_pairs: int = 200_000,
):
    cat = pd.read_parquet(catalog_path)
    path_col = "local_path" if "local_path" in cat.columns else cat.columns[-1]
    pairs = generate_pairs(cat, path_col=path_col, weak_forgery=weak_forgery)
    pairs = balance_subsample(pairs, max_pairs=max_pairs)
    pairs.to_parquet(output_path)
    print(f"✓ wrote {len(pairs):,} balanced pairs → {output_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate labelled BiosecurID pairs")
    ap.add_argument("catalog", type=Path)
    ap.add_argument("output",  type=Path)
    ap.add_argument("--weak-forgery", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=200_000)
    args = ap.parse_args()

    main(args.catalog, args.output, weak_forgery=args.weak_forgery, max_pairs=args.max_pairs)
