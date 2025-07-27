"""
Training utilities for Siamese‐DTW neural extension.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

def train_siamese_model(
    model: Model,
    X1: np.ndarray,
    X2: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 32,
    epochs: int = 10,
    validation_split: float = 0.2
) -> tf.keras.callbacks.History:
    """
    Train the Siamese DTW model on paired data.
    Returns the Keras History object.
    """
    return model.fit(
        x=[X1, X2],
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split
    )
