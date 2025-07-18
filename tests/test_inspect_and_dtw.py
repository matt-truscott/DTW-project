import os
import numpy as np
import pytest
import scipy.io as sio

from src.displayData import inspect_mat_file
from src.dtwAlgorithm import dp

# helper to get full path under tests/data
HERE = os.path.dirname(__file__)
def data_path(*parts):
    return os.path.join(HERE, "data", *parts)

def test_inspect_global_vector(tmp_path):
    mat_path = data_path("GlobalFeatures", "u0001s0001_sg0001.mat")
    feats = inspect_mat_file(mat_path)

    # should return exactly one key: globalFeatures
    assert "globalFeatures" in feats
    vec = feats["globalFeatures"]
    assert vec.shape == (40,)
    # we generated linspace(0,1,40)
    np.testing.assert_allclose(vec[0], 0.0)
    np.testing.assert_allclose(vec[-1], 1.0)

def test_inspect_local_matrix(tmp_path):
    mat_path = data_path("LocalFunctions", "u0001s0001_sg0001.mat")
    feats = inspect_mat_file(mat_path)

    assert "localFunctions" in feats
    mat = feats["localFunctions"]
    assert mat.ndim == 2
    assert mat.shape[1] == 9  # 9 time‑functions

def test_dtw_identity():
    # simple identity test: two identical sequences
    seq = np.arange(10)
    # build distance matrix
    dist_mat = np.abs(seq.reshape(-1,1) - seq.reshape(1,-1))
    path, cost = dp(dist_mat)

    # alignment should be diagonal
    expected = [(i, i) for i in range(10)]
    assert path == expected

    # cost at (9,9) should be 0
    assert cost[-1, -1] == 0