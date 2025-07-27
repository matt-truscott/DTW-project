"""
Data utilities for Siamese‐DTW neural extension.
"""

import numpy as np
from pathlib import Path
from src.io.load_biosecurid import load_local

def resample_sequence(seq: np.ndarray, target_length: int) -> np.ndarray:
    """
    Linearly resample a (n_samples, n_features) array to shape (target_length, n_features).
    """
    n, f = seq.shape
    if n == target_length:
        return seq
    xp = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, target_length)
    out = np.zeros((target_length, f), dtype=float)
    for j in range(f):
        out[:, j] = np.interp(x_new, xp, seq[:, j])
    return out

def load_siamese_data(
    pairs_df,
    processed_root: Path,
    sequence_length: int
):
    """
    Given a DataFrame with columns ['pathA','pathB','label'],
    load each localFunctions .mat, resample to fixed length,
    and return (X1, X2, y) where:
      - X1, X2 are arrays of shape (N, sequence_length, n_features)
      - y is (N,) labels 0/1
    """
    X1, X2, y = [], [], []
    for _, row in pairs_df.iterrows():
        pA = processed_root / row["pathA"]
        pB = processed_root / row["pathB"]
        seqA = load_local(pA)
        seqB = load_local(pB)
        X1.append(resample_sequence(seqA, sequence_length))
        X2.append(resample_sequence(seqB, sequence_length))
        y.append(int(row["label"]))
    return np.stack(X1), np.stack(X2), np.array(y, dtype=int)


