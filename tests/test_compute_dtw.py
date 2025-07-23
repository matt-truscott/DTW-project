import numpy as np
from src.dtw.compute_dtw import compute_pair_dtw


def test_compute_pair_dtw_identity():
    a = np.arange(5).reshape(-1, 1)
    b = np.arange(5).reshape(-1, 1)
    d_raw, d_bound, plen, la, lb = compute_pair_dtw(a, b, backend="python", window=1)

    assert d_raw == 0.0
    assert d_bound == 0.0
    assert plen == 5
    assert la == 5
    assert lb == 5
