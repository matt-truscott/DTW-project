from tensorflow.keras.layers import (Input, TimeDistributed, Dense,
                                     GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from src.keras_layers.diff_dtw import DiffDTW

L, F = 128, 32
ref_input = Input((L, F), name="reference")
qry_input = Input((L, F), name="query")

td = TimeDistributed(Dense(7, activation="relu"))
ref_feat = GlobalAveragePooling1D()(td(ref_input))
qry_feat = GlobalAveragePooling1D()(td(qry_input))

dist = DiffDTW()([ref_feat, qry_feat])
out = Dense(1, activation="sigmoid")(dist)

model = Model([ref_input, qry_input], out, name="diffdtw_baseline")
