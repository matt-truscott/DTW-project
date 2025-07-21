import numpy as np
import tensorflow as tf
from src.keras_layers.diff_dtw import DiffDTW


def soft_dtw_numpy(a, b, gamma=1.0):
    """Reference soft-DTW implementation using NumPy."""
    n, f = a.shape
    m, _ = b.shape
    D = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    R = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            r = np.array([R[i - 1, j - 1], R[i - 1, j], R[i, j - 1]], dtype=np.float32)
            softmin = -gamma * np.log(np.exp(-r[0] / gamma) + np.exp(-r[1] / gamma) + np.exp(-r[2] / gamma))
            R[i, j] = D[i - 1, j - 1] + softmin
    return R[n, m]


def test_forward_distance():
    a = np.array([[0.0], [1.0]], dtype=np.float32)
    b = np.array([[1.0], [0.0]], dtype=np.float32)
    layer = DiffDTW(gamma=1.0)
    tf_dist = layer([a[None, ...], b[None, ...]])
    np_dist = soft_dtw_numpy(a, b, gamma=1.0)
    assert np.allclose(tf_dist.numpy().squeeze(), np_dist, atol=1e-5)


def test_gradients():
    seq_a = tf.random.uniform((2, 3, 2))
    seq_b = tf.random.uniform((2, 3, 2))
    layer = DiffDTW(gamma=1.0)
    func = lambda x, y: layer([x, y])
    (grad_a, grad_b), (num_a, num_b) = tf.test.compute_gradient(func, [seq_a, seq_b])
    tf.debugging.assert_near(grad_a, num_a, rtol=1e-2, atol=1e-2)
    tf.debugging.assert_near(grad_b, num_b, rtol=1e-2, atol=1e-2)
