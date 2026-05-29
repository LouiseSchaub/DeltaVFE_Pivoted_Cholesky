import numpy as np


def woodbury_apply_matrix(L: np.ndarray, B: np.ndarray, sigma2: float, v: np.ndarray) -> np.ndarray:
    """
    Batched Woodbury apply Z = (LL^T + sigma2 I)^{-1}V

    :param L ((n, r)-np.ndarray): approximate Cholesky factro
    :param B ((r, r)-np.ndarray): inverse B = (I + sigma2^{-1} L^T L)^{-1}
    :param sigma2: squared noise variance
    :param v ((n, k)-np.ndarray): vector to apply the inverse to
    :return: Z ((n, k)-np.ndarray): inverse applied to vector v
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)

    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if L.ndim != 2:
        raise ValueError("L must be 2D.")
    if B.shape != (L.shape[1], L.shape[1]):
        raise ValueError("B must have shape (r, r).")
    if v.shape[0] != L.shape[0]:
        raise ValueError("V has incompatible shape.")

    if L.shape[1] == 0:
        return v / sigma2

    LtV = L.T @ v
    return (v / sigma2) - (L @ (B @ LtV)) / (sigma2 * sigma2)


def woodbury_apply_vector(L: np.ndarray, B: np.ndarray, sigma2: float, v: np.ndarray) -> np.ndarray:
    """
    Vector convenience wrapper for the batched Woodbury apply

    :param L (np.ndarray): Cholesky factor.
    :param B (np.ndarray): Woodbury-inverse
    :param sigma2 (float): squared noise variance
    :param v (np.ndarray): vector that the Woodbury inverse should be applied to.
    :return: \Sigma_i^{-1}v
    """
    z = woodbury_apply_matrix(L, B, sigma2, np.asarray(v, dtype=np.float64).reshape(-1, 1))
    return z[:, 0]


def update_B(B: np.ndarray, L: np.ndarray, u: np.ndarray, sigma2: float, eps: float = 1e-14) -> np.ndarray:
    """
    Lemma 5.3 block update for B.
        B_new = (I + sigma2^{-1} [L u]^T [L u])^{-1}

    :param B (np.ndarray): current B to be updated.
    :param L (np.ndarray): Cholesky factor.
    :param u (np.ndarray): rank-1 update
    :param sigma2 (float): squared noise variance
    :param eps (float): numerical safeguqad
    :return: updated B
    """

    u = np.asarray(u, dtype=np.float64).reshape(-1)
    r = L.shape[1]

    if r == 0:
        xi = 1.0 + float(u @ u) / sigma2
        return np.array([[1.0 / max(xi, eps)]], dtype=np.float64)

    c = (L.T @ u) / sigma2
    xi = 1.0 + float(u @ u) / sigma2
    Bc = B @ c
    denom = xi - float(c @ Bc)
    gamma = 1.0 / max(denom, eps)

    B_new = np.empty((r + 1, r + 1), dtype=np.float64)
    B_new[:r, :r] = B + gamma * np.outer(Bc, Bc)
    B_new[:r, r] = -gamma * Bc
    B_new[r, :r] = -gamma * Bc
    B_new[r, r] = gamma
    return B_new


def delta_vfe_pivots_batched(K: np.ndarray, y: np.ndarray, m: int, sigma2: float, s: int = 64, seed: int = 0,
                                   stop_if_nonpositive_gain: bool = False, eps: float = 1e-14) -> list[int]:
    """
    This is the realization of Algorithm 5.2 and Remark 5.4 in the paper, i.e.
        - sample candidate batch S proportional to residual diagonal (compare RPCholesky)
        - compute v = Sigma^{-1} y once per iteration
        - form all residual columns for S at once
        - normalize all candidate directions at once
        - apply Woodbury to all candidates at once
        - score all candidates at once
        - append the best candidate and update B, d, pivots


    :param K (np.ndarray): full nxn kernel matrix to approximate
    :param y (np.ndarray): output data vector.
    :param m (int): maximum target rank.
    :param sigma2 (float): squared noise variance.
    :param s (int): batch factor.
    :param seed (int): RNG seed.
    :param stop_if_nonpositive_gain (bool): optional safeguard. Leave False for behaviour that is faithful to the paper.
    :param eps: numerical safeguqad.
    :return: list of pivots.
    """
    K = np.asarray(K, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError("K must be square.")
    if y.shape[0] != n:
        raise ValueError("y has incompatible shape.")
    if m < 0:
        raise ValueError("m must be nonnegative.")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if s <= 0:
        raise ValueError("s must be positive.")

    rng = np.random.default_rng(seed)

    # Algorithm 5.2 state
    L = np.zeros((n, 0), dtype=np.float64)
    B = np.zeros((0, 0), dtype=np.float64)
    d = np.diag(K).astype(np.float64).copy()
    pivot_mask = np.zeros(n, dtype=bool)
    pivots: list[int] = []

    for _ in range(m):
        C = np.flatnonzero(~pivot_mask)
        if C.size == 0:
            break

        eligible = C[d[C] > eps]
        if eligible.size == 0:
            break

        weights = d[eligible]
        wsum = float(weights.sum())
        if wsum <= 0.0:
            break

        batch_size = min(s, eligible.size)
        S = rng.choice(eligible, size=batch_size, replace=False, p=weights / wsum)

        # Algorithm 5.2 line 8: v = Sigma^{-1} y
        v = woodbury_apply_vector(L, B, sigma2, y)

        # Remark 5.4: form all residual vectors jointly
        # T_S = K[:, S] - L L[S, :]^T
        K_batch = K[:, S]  # shape (n, batch_size)
        if L.shape[1] == 0:
            T_batch = K_batch.copy()
        else:
            T_batch = K_batch - (L @ L[S, :].T)

        dS = d[S].astype(np.float64)
        valid = dS > eps
        if not np.any(valid):
            break

        # Keep only numerically valid candidates
        S = S[valid]
        T_batch = T_batch[:, valid]
        dS = dS[valid]

        sqrt_dS = np.sqrt(dS)
        U_batch = T_batch / sqrt_dS.reshape(1, -1)

        # Batched Woodbury apply: Z_batch[:,j] = Sigma^{-1} u_j
        Z_batch = woodbury_apply_matrix(L, B, sigma2, U_batch)

        # a_p = u_p^T z_p for all candidates
        a = np.einsum("ij,ij->j", U_batch, Z_batch)

        # b_p = u_p^T v for all candidates
        b = U_batch.T @ v

        # ||u_p||^2 for all candidates
        unorm2 = np.einsum("ij,ij->j", U_batch, U_batch)

        onepa = np.maximum(1.0 + a, eps)
        delta = -0.5 * (np.log(onepa) - (b * b) / onepa - unorm2 / sigma2)

        best_idx = int(np.argmax(delta))
        best_delta = float(delta[best_idx])

        if stop_if_nonpositive_gain and best_delta <= 0.0:
            break

        best_p = int(S[best_idx])
        u = U_batch[:, best_idx]

        # Update B and L
        B = update_B(B, L, u, sigma2, eps=eps)
        L = np.column_stack([L, u])

        # Update residual diagonal: d <- d - u^2
        d -= u * u
        np.maximum(d, 0.0, out=d)
        d[best_p] = 0.0

        pivot_mask[best_p] = True
        pivots.append(best_p)

    return pivots


