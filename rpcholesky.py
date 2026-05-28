import numpy as np

def rpcholesky_pivots(K: np.ndarray, m: int, seed: int=0):
    """
    RPCholesky to compute pivtos from kernel matrix K as given by Epperly et al.:
    https://github.com/eepperly/Randomly-Pivoted-Cholesky.
    The low-rank factor is build rowwise
    :param K: kernel matrix
    :param m: target rank
    :param seed: .
    :return: list of pivots. Optionally: return G if you want the low-rank factor K \approx G^T G. So, G = L.T
    """
    rng = np.random.default_rng(seed)
    diags = np.diag(K).copy()
    G = np.zeros((m, K.shape[0]), dtype=np.float64)
    pivots = []

    for i in range(m):
        eligible = np.where(diags > 1e-14)[0]
        if eligible.size == 0:
            break
        p = diags[eligible] / float(diags[eligible].sum())
        idx = int(rng.choice(eligible, p=p))
        pivots.append(idx)

        row = K[idx, :]
        correction = G[:i, idx].T @ G[:i, :]
        G[i, :] = (row - correction) / np.sqrt(diags[idx])

        diags = np.clip(diags - G[i, :] ** 2, 0.0, None)
        diags[idx] = 0.0

    return pivots #, G[:len(pivots), :]