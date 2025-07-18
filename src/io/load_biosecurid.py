"""
load_biosecurid.py

Module to ingest all BiosecurID .mat files into a single analysis-ready catalog.
"""
import re
from pathlib import Path
import pandas as pd
import scipy.io as sio

FILENAME_PATTERN = re.compile(r"u(?P<user>\d{4})s(?P<session>\d{4})_sg(?P<sample>\d{4})\.mat$")


def parse_filename(fname: str):
    """
    Parse a BiosecurID filename uXXXXsYYYY_sgZZZZ.mat
    into user_id, session_id, sample_id (as ints).
    """
    m = FILENAME_PATTERN.search(fname)
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
    Returns a numpy array of shape (G,).
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat['globalFeatures']


def load_local(mat_path: Path):
    """
    Load a localFunctions matrix from a .mat file.
    Returns a numpy array of shape (n_samples, F).
    """
    mat = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    return mat['localFunctions']


def build_catalog(
    global_dir: Path,
    local_dir: Path,
    output_catalog: Path
):
    """
    Scan both GlobalFeatures and LocalFunctions directories, parse metadata,
    and build a master DataFrame with columns:
      - user_id, session_id, sample_id
      - global_path, local_path
    Persists catalog to Parquet.
    """
    records = []

    # assume both dirs contain same set of uXXXXsYYYY_sgZZZZ.mat files
    for gfile in sorted(global_dir.glob('u*.mat')):
        user, session, sample = parse_filename(gfile.name)
        lfile = local_dir / gfile.name
        if not lfile.exists():
            continue  # or raise

        records.append({
            'user_id': user,
            'session_id': session,
            'sample_id': sample,
            'global_path': str(gfile.resolve()),
            'local_path': str(lfile.resolve()),
        })

    df = pd.DataFrame.from_records(records)
    df.to_parquet(output_catalog)
    return df


if __name__ == '__main__':
    project_root = Path(__file__).parents[2]
    glob_dir = project_root / 'data' / 'processed' / 'GlobalFeatures'
    loc_dir  = project_root / 'data' / 'raw' / 'LocalFunctions'
    catalog  = project_root / 'data' / 'biosecurid_catalog.parquet'
    df = build_catalog(glob_dir, loc_dir, catalog)
    print(f"Catalog written with {len(df)} entries to {catalog}")
