import numpy as np

def greedy_pivots(K: np.ndarray, m:int):
    """

    :param K:
    :param m:
    :return:
    """
    n = K.shape[0]
    diags = np.diag(K).copy()
    G = np.zeros((m, n), dtype=np.float64)
    pivots = []

    for i in range(m):
        idx = int(np.argmax(diags))
        if diags[idx] <= 1e-14:
            break
        pivots.append(idx)

        row = K[idx, :]
        correction = G[:i, idx].T @ G[:i, :]
        G[i, :] = (row - correction) / np.sqrt(diags[idx])

        diags = np.clip(diags - G[i, :] ** 2, 0.0, None)
        diags[idx] = 0.0

    return pivots
