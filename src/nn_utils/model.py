"""
Model utilities for Siamese‐DTW neural extension.
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Flatten
from tensorflow.keras.models import Model
from src.keras_layers.diff_dtw import DiffDTW

def build_embedding_net(
    hidden_dims: tuple[int, ...] = (7, 5),
    input_shape: tuple[int, int] = None
) -> tf.keras.Model:
    """
    Returns a Sequential network mapping (L, F) → (embedding_dim,)
    where hidden_dims defines the intermediate Dense layers.
    """
    layers = []
    layers.append(Flatten())
    for dim in hidden_dims:
        layers.append(Dense(dim, activation="relu"))
    return tf.keras.Sequential(layers, name="EmbeddingNet")

def build_siamese_dtw_model(
    sequence_length: int,
    n_features: int,
    hidden_dims: tuple[int, ...] = (7, 5),
    gamma: float = 1.0
) -> tf.keras.Model:
    """
    Build & compile a Siamese network with:
     - Shared embedding net
     - Differentiable DTW layer
     - Sigmoid output for binary classification
    """
    # Define inputs
    inpA = Input(shape=(sequence_length, n_features), name="inputA")
    inpB = Input(shape=(sequence_length, n_features), name="inputB")

    # Shared embedding
    embed_net = build_embedding_net(hidden_dims, input_shape=(sequence_length, n_features))
    embA = embed_net(inpA)
    embB = embed_net(inpB)

    # Differentiable DTW distance
    dist = DiffDTW(gamma=gamma, name="diffdtw")([embA, embB])

    # Final sigmoid
    out = Dense(1, activation="sigmoid", name="output")(dist)

    model = Model(inputs=[inpA, inpB], outputs=out, name="SiameseDTW")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
