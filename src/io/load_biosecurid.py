"""
load_biosecurid.py

Module to ingest all BiosecurID .mat files into a single analysis-ready catalog.
"""
import re
from pathlib import Path
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


