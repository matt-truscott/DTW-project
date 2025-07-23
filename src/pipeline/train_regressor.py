"""Train a small regression model to calibrate DTW scores."""
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf


def load_features(cache_path: Path):
    df = pd.read_parquet(cache_path)
    X = df[["d_raw", "d_bound", "path_len", "len_ref", "len_qry"]].to_numpy()
    y = df["label"].to_numpy()
    return X, y


def build_model(input_dim: int = 5) -> tf.keras.Model:
    inputs = tf.keras.Input((input_dim,))
    x = tf.keras.layers.Dense(8, activation="relu")(inputs)
    x = tf.keras.layers.Dense(4, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train score calibration model")
    ap.add_argument("--cache", type=Path, default=Path("cache/dtw.parquet"))
    ap.add_argument("--model", type=Path, default=Path("models/bounded_dtw.h5"))
    args = ap.parse_args()

    X, y = load_features(args.cache)
    n = len(X)
    val_split = max(1, int(0.2 * n))
    X_train, X_val = X[:-val_split], X[-val_split:]
    y_train, y_val = y[:-val_split], y[-val_split:]

    model = build_model(X.shape[1])
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=[cb], verbose=0)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
    print(f"✓ model saved to {args.model}")


if __name__ == "__main__":
    main()
