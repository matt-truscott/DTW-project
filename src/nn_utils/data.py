# src/nn_utils/data.py
"""
Data utilities for the Siamese‐DTW extension.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np

from src.io.load_biosecurid import load_local  # returns (T, 9) float array

# Project root (…/DTW-project). Useful when pairs store project-relative paths like "data/processed/…"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path, processed_root: Path) -> Path:
    """
    Resolve `p` to an absolute path.

    Rules:
    - If `p` is absolute → return as-is.
    - If `p` starts with "data/" (or "data\\") → treat as project-relative under PROJECT_ROOT.
    - Otherwise → treat as path relative to `processed_root`.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp

    # Normalise to string for prefix checks (Windows backslashes are fine).
    s = str(pp)
    if s.lower().startswith("data/") or s.lower().startswith("data\\"):
        return (PROJECT_ROOT / pp).resolve()

    return (processed_root / pp).resolve()


def resample_sequence(seq: np.ndarray, target_length: int) -> np.ndarray:
    """
    Linearly resample a (n_samples, n_features) array to shape (target_length, n_features).
    Returns float32 for TF.
    """
    n, f = seq.shape
    if n == target_length:
        return seq.astype(np.float32, copy=False)

    xp = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_length)
    out = np.zeros((target_length, f), dtype=np.float32)
    for j in range(f):
        out[:, j] = np.interp(x_new, xp, seq[:, j]).astype(np.float32, copy=False)
    return out


def load_siamese_data(pairs_df, processed_root: Path, sequence_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pairs_df must have ['pathA','pathB','label'] (paths may be absolute or project-relative).
    Returns X1, X2, y with shapes:
      - X1, X2: (N, sequence_length, 9)  float32
      - y:      (N,) int
    """
    processed_root = Path(processed_root)
    X1, X2, y = [], [], []

    for _, row in pairs_df.iterrows():
        pA = _resolve_path(str(row["pathA"]), processed_root)
        pB = _resolve_path(str(row["pathB"]), processed_root)
        seqA = load_local(pA)  # (T, 9)
        seqB = load_local(pB)  # (T, 9)
        X1.append(resample_sequence(seqA, sequence_length))
        X2.append(resample_sequence(seqB, sequence_length))
        y.append(int(row["label"]))

    return (
        np.stack(X1, dtype=np.float32),
        np.stack(X2, dtype=np.float32),
        np.asarray(y, dtype=int),
    )
