from matplotlib.patches import ConnectionPatch  # for drawing arrows between subplots (if needed)
import matplotlib.pyplot as plt                 # standard plotting library
import numpy as np                              # numerical array operations
import scipy.spatial.distance as dist           # for computing distance matrices, if you need to build dist_mat

def dp(dist_mat):
    """
    Compute the optimal Dynamic Time Warping (DTW) alignment path
    through a given cost matrix `dist_mat`, using dynamic programming.

    Parameters
    ----------
    dist_mat : ndarray, shape (N, M)
        Pairwise local distances between sequence A (length N)
        and sequence B (length M).

    Returns
    -------
    path : list of (i, j) tuples
        The best alignment path from (0,0) to (N-1, M-1).
    cost_mat : ndarray, shape (N, M)
        Accumulated cost matrix: cost_mat[i, j] is the minimum
        total cost of aligning prefixes A[:i+1], B[:j+1].
    """
    # Retrieve the lengths of the two sequences
    N, M = dist_mat.shape

    # 1) Create a (N+1)x(M+1) cost matrix with an extra top row/left column
    #    to simplify boundary conditions. Initialize to zeros.
    cost_mat = np.zeros((N + 1, M + 1))

    # 2) Set the first column (except cost_mat[0,0]) to infinity:
    #    these cells should never be picked.
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf

    # 3) Set the first row (except cost_mat[0,0]) to infinity:
    for j in range(1, M + 1):
        cost_mat[0, j] = np.inf

    # 4) Prepare a traceback matrix (N x M) to record which move
    #    was chosen at each step (0=match, 1=insertion, 2=deletion).
    traceback_mat = np.zeros((N, M), dtype=int)

    # 5) Fill in the cost_mat row by row
    for i in range(N):
        for j in range(M):
            # For each cell (i,j) in dist_mat, consider three possible predecessors:
            #   cost_mat[i, j]     -> diagonal (match)
            #   cost_mat[i, j+1]   -> up       (insertion in B)
            #   cost_mat[i+1, j]   -> left     (deletion from A)
            penalties = [
                cost_mat[i,   j],   # match (0)
                cost_mat[i,   j+1], # insertion (1)
                cost_mat[i+1, j]    # deletion  (2)
            ]
            # Pick the move with minimal cumulative cost so far
            best = np.argmin(penalties)

            # Update the cumulative cost at (i+1, j+1)
            cost_mat[i+1, j+1] = dist_mat[i, j] + penalties[best]
            # Record the chosen move for backtracking
            traceback_mat[i, j] = best

    # 6) Backtrack from the bottom-right corner to recover the optimal path
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb = traceback_mat[i, j]
        if tb == 0:
            # match: move diagonally up-left
            i -= 1
            j -= 1
        elif tb == 1:
            # insertion: move up (advance in sequence A only)
            i -= 1
        else:
            # deletion: move left (advance in sequence B only)
            j -= 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)
