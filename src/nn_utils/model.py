"""
Model utilities for Siamese‐DTW neural extension.
"""

from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense
from src.keras_layers.diff_dtw import DiffDTW

def build_siamese_dtw_model(
    sequence_length: int,
    n_features: int,
    hidden_dims: tuple[int, ...] = (7, 5),
    dtw_gamma: float = 1.0,
    post_hidden: tuple[int, ...] = (16, 8),
) -> Model:
    """
    Siamese network with a differentiable DTW layer and post-DTW dense stack,
    ending in a 2-way softmax.
    """
    inpA = Input((sequence_length, n_features), name="inputA")
    inpB = Input((sequence_length, n_features), name="inputB")

    # time-distributed embedding
    xA, xB = inpA, inpB
    for i, dim in enumerate(hidden_dims):
        td = TimeDistributed(Dense(dim, activation="relu"), name=f"td_dense_{i}")
        xA, xB = td(xA), td(xB)

    # DTW layer
    dist = DiffDTW(gamma=dtw_gamma, name="diffdtw")([xA, xB])

    # post-DTW stack
    x = dist
    for i, dim in enumerate(post_hidden):
        x = Dense(dim, activation="relu", name=f"post_dtw_dense_{i}")(x)

    out = Dense(2, activation="softmax", name="output")(x)

    model = Model([inpA, inpB], out, name="SiameseDTW")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_baseline_nn(
    input_dim: int = 18,
    hidden_dims: tuple[int, ...] = (32, 16),
) -> Model:
    """
    Simple feed-forward network on an 18-dim vector, ending in 2-way softmax.
    """
    inp = Input((input_dim,), name="baseline_input")
    x = inp
    for i, dim in enumerate(hidden_dims):
        x = Dense(dim, activation="relu", name=f"baseline_dense_{i}")(x)
    out = Dense(2, activation="softmax", name="baseline_output")(x)

    model = Model(inp, out, name="BaselineNN")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model