import tensorflow as tf
from tensorflow.keras.layers import Layer


def _pairwise_distances(seq_a: tf.Tensor, seq_b: tf.Tensor) -> tf.Tensor:
    """Compute squared Euclidean distances between sequence steps."""
    a_exp = tf.expand_dims(seq_a, 2)  # (B, L_a, 1, F)
    b_exp = tf.expand_dims(seq_b, 1)  # (B, 1, L_b, F)
    return tf.reduce_sum(tf.square(a_exp - b_exp), axis=-1)  # (B, L_a, L_b)


def _soft_dtw(seq_a: tf.Tensor, seq_b: tf.Tensor, gamma: float) -> tf.Tensor:
    """Differentiable soft-DTW implementation for batches."""
    D = _pairwise_distances(seq_a, seq_b)
    B = tf.shape(D)[0]
    N = tf.shape(D)[1]
    M = tf.shape(D)[2]
    dtype = D.dtype

    inf = tf.constant(1e6, dtype=dtype)
    R = tf.fill((B, N + 2, M + 2), inf)
    idx0 = tf.stack(
        [tf.range(B, dtype=tf.int32), tf.zeros((B,), tf.int32), tf.zeros((B,), tf.int32)],
        axis=1,
    )
    R = tf.tensor_scatter_nd_update(R, idx0, tf.zeros((B,), dtype=dtype))

    def body_i(i, R):
        def body_j(j, R):
            r0 = R[:, i, j]
            r1 = R[:, i, j + 1]
            r2 = R[:, i + 1, j]
            r = tf.stack([r0, r1, r2], axis=-1)
            softmin = -gamma * tf.reduce_logsumexp(-r / gamma, axis=-1)
            val = D[:, i, j] + softmin
            idx = tf.stack(
                [tf.range(B, dtype=tf.int32), tf.fill([B], i + 1), tf.fill([B], j + 1)],
                axis=1,
            )
            R = tf.tensor_scatter_nd_update(R, idx, val)
            return j + 1, R

        j0 = tf.constant(0)
        _, R = tf.while_loop(lambda j, _: j < M, body_j, [j0, R])
        return i + 1, R

    i0 = tf.constant(0)
    _, R = tf.while_loop(lambda i, _: i < N, body_i, [i0, R])
    return R[:, N, M]


class DiffDTW(Layer):
    """Keras layer computing differentiable DTW distance."""

    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = float(gamma)

    def call(self, inputs):
        seq_a, seq_b = inputs
        dist = _soft_dtw(seq_a, seq_b, self.gamma)
        return tf.expand_dims(dist, axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        return (batch_size, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config
