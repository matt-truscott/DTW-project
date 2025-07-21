import numpy as np
from src.dtw.compute_dtw import compute_pair_dtw


def test_compute_pair_dtw_identity():
    a = np.arange(5).reshape(-1, 1)
    b = np.arange(5).reshape(-1, 1)
    d_raw, n1, n2, n3, plen = compute_pair_dtw(a, b, backend="python")

    assert d_raw == 0.0
    assert plen == 5
    assert n1 == 0.0
    assert n2 == 0.0
    assert n3 == 0.0
