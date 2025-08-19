# src/nn_utils/model.py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, TimeDistributed, Dense, Lambda

try:
    from src.keras_layers.diff_dtw import DiffDTW  # your layer
except Exception:
    DiffDTW = None  # type: ignore

def _make_dtw_layer(dtw_gamma: float):
    # Guard on the symbol itself (Pylance-safe)
    if DiffDTW is not None:
        return DiffDTW(gamma=dtw_gamma, name="diffdtw")
    # Fallback: differentiable proxy – mean L1 distance over time/features
    return Lambda(
        lambda pair: tf.expand_dims(
            tf.reduce_mean(tf.abs(pair[0] - pair[1]), axis=[1, 2]),
            axis=-1
        ),
        name="l1_time_mean",
    )

def build_siamese_dtw_model(
    sequence_length: int,
    n_features: int,
    hidden_dims: tuple[int, ...] = (7, 5),
    dtw_gamma: float = 1.0,
    post_hidden: tuple[int, ...] = (16, 8),
) -> Model:
    inpA = Input((sequence_length, n_features), name="inputA")
    inpB = Input((sequence_length, n_features), name="inputB")

    xA, xB = inpA, inpB
    for i, dim in enumerate(hidden_dims):
        td = TimeDistributed(Dense(dim, activation="relu"), name=f"td_dense_{i}")
        xA, xB = td(xA), td(xB)

    dist = _make_dtw_layer(dtw_gamma)([xA, xB])  # (batch, 1)

    x = dist
    for i, dim in enumerate(post_hidden):
        x = Dense(dim, activation="relu", name=f"post_dtw_dense_{i}")(x)
    out = Dense(2, activation="softmax", name="output")(x)

    model = Model([inpA, inpB], out, name="SiameseDTW")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_baseline_nn(input_dim: int = 18, hidden_dims: tuple[int, ...] = (32, 16)) -> Model:
    inp = Input((input_dim,), name="baseline_input")
    x = inp
    for i, dim in enumerate(hidden_dims):
        x = Dense(dim, activation="relu", name=f"baseline_dense_{i}")(x)
    out = Dense(2, activation="softmax", name="baseline_output")(x)
    model = Model(inp, out, name="BaselineNN")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
