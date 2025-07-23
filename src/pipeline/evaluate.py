"""Simple evaluation script for the regression model."""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve


def load_data(cache: Path):
    df = pd.read_parquet(cache)
    X = df[["d_raw", "d_bound", "path_len", "len_ref", "len_qry"]].to_numpy()
    y = df["label"].to_numpy()
    return X, y


def compute_eer(y_true, scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Evaluate model")
    ap.add_argument("--model", type=Path, default=Path("models/bounded_dtw.h5"))
    ap.add_argument("--cache", type=Path, default=Path("cache/dtw.parquet"))
    args = ap.parse_args()

    X, y = load_data(args.cache)
    model = tf.keras.models.load_model(args.model)
    scores = model.predict(X, verbose=0).squeeze()
    eer = compute_eer(y, scores)
    print(f"EER: {eer:.4f}")


if __name__ == "__main__":
    main()
