import numpy as np


def dp(dist_mat: np.ndarray):
    """Compute the optimal DTW path and cost matrix."""
    N, M = dist_mat.shape

    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for j in range(1, M + 1):
        cost_mat[0, j] = np.inf

    traceback_mat = np.zeros((N, M), dtype=int)

    for i in range(N):
        for j in range(M):
            penalties = [cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j]]
            best = np.argmin(penalties)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalties[best]
            traceback_mat[i, j] = best

    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb = traceback_mat[i, j]
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))

    cost_mat = cost_mat[1:, 1:]
    return path[::-1], cost_mat
