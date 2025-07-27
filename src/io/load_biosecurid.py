"""
Module to ingest all BiosecurID .mat files into a single analysis-ready catalog.
"""

import re
from pathlib import Path
import pandas as pd
import scipy.io as sio

# regex to parse filenames like u1001s0001_sg0001.mat
FILENAME_PATTERN = re.compile(
    r"^u(?P<user>\d{4})s(?P<session>\d{4})_sg(?P<sample>\d{4})\.mat$"
)
# which sample numbers are genuine
GENUINE_SET = {1, 2, 6, 7}


def parse_filename(fname: str):
    """
    Parse a BiosecurID filename uXXXXsYYYY_sgZZZZ.mat
    into (user_id, session_id, sample_id) as ints.
    """
    m = FILENAME_PATTERN.match(fname)
    if not m:
        raise ValueError(f"Filename not in expected format: {fname}")
    return (
        int(m.group('user')),
        int(m.group('session')),
        int(m.group('sample'))
    )


def load_global(mat_path: Path):
    """
    Load a globalFeatures vector from a .mat file.
    Returns a numpy array of shape (40,).
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat['globalFeatures']


def load_local(mat_path: Path):
    """
    Load a localFunctions matrix from a .mat file.
    Returns a numpy array of shape (n_samples, 9).
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat['localFunctions']


def build_catalog(
    processed_root: Path,
    output_catalog: Path
):
    """
    Walks processed_root/u*/{GlobalFeatures,LocalFunctions} for all .mat files and
    builds a DataFrame with columns:
      - file_path: relative path under processed_root
      - user_id, session_id, sample_id
      - feature: 'globalfeatures' or 'localfunctions'
      - label: 'genuine' or 'forgery'

    Persists the DataFrame to Parquet at output_catalog.
    """
    processed_root = Path(processed_root)
    records = []

    for user_dir in sorted(processed_root.glob("u*")):
        for feat in ("GlobalFeatures", "LocalFunctions"):
            feat_dir = user_dir / feat
            if not feat_dir.exists():
                continue
            for mat_file in sorted(feat_dir.glob("u*.mat")):
                user, session, sample = parse_filename(mat_file.name)
                label = "genuine" if sample in GENUINE_SET else "forgery"
                rel_path = mat_file.relative_to(processed_root)
                records.append({
                    "file_path":  str(rel_path),
                    "user_id":    user,
                    "session_id": session,
                    "sample_id":  sample,
                    "feature":    feat.lower(),
                    "label":      label
                })

    df = pd.DataFrame.from_records(records)
    df.to_parquet(output_catalog, index=False)
    return df


if __name__ == "__main__":
    project_root   = Path(__file__).resolve().parents[2]
    processed_root = project_root / "data" / "processed"
    catalog_path   = project_root / "data" / "biosecurid_catalog.parquet"

    df = build_catalog(processed_root, catalog_path)
    print(f"Catalog written with {len(df)} entries to {catalog_path}")