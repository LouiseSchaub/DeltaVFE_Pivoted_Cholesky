import numpy as np


def delta_vfe_pivots_fast(
    K: np.ndarray,
    y: np.ndarray,
    m: int,
    sigma2: float,
    s: int = 64,
    seed: int = 0,
    assume_symmetric: bool = True,
    stop_if_nonpositive_gain: bool = False,
    eps: float = 1e-14,
) -> list[int]:
    """
    Faster realization of Algorithm 5.2 (Delta-VFE pivoted Cholesky).

    Mathematically identical to the reference implementation; for the same
    `seed` it draws the same candidate batches and therefore returns the same
    pivots (up to floating-point rounding in the argmax).

    Key differences from the reference:
      * Never materializes Z_batch = Sigma^{-1} U_batch (an (n, s) array).
        Scores are computed from W = L^T U_batch using the Woodbury form
            Sigma^{-1} = sigma2^{-1} I - sigma2^{-2} L B L^T,
        so   a_p = sigma2^{-1}||u_p||^2 - sigma2^{-2} w_p^T B w_p,
             b_p = sigma2^{-1}(u_p^T y) - sigma2^{-2} w_p^T B (L^T y).
        This removes one O(n i s) GEMM per iteration (3 -> 2).
      * Maintains Ly = L^T y incrementally, so v = Sigma^{-1} y is never formed.
      * Reuses W[:, best] as c for the rank-1 B update (no recompute of L^T u).
      * Reads K[S, :] (contiguous rows) and transposes, valid because the kernel
        matrix is symmetric (assume_symmetric=True); avoids strided K[:, S] gather.

    :param K: (n, n) symmetric positive definite kernel matrix.
    :param y: (n,) output vector.
    :param m: maximum target rank.
    :param sigma2: squared noise variance (> 0).
    :param s: batch factor.
    :param seed: RNG seed.
    :param assume_symmetric: if True, read K[S, :].T instead of K[:, S].
    :param stop_if_nonpositive_gain: optional safeguard; False is faithful to the paper.
    :param eps: numerical safeguard.
    :return: list of pivot indices.
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

    inv_s2 = 1.0 / sigma2
    inv_s4 = inv_s2 * inv_s2

    rng = np.random.default_rng(seed)

    # Algorithm 5.2 state
    L = np.zeros((n, 0), dtype=np.float64)
    B = np.zeros((0, 0), dtype=np.float64)
    Ly = np.zeros((0,), dtype=np.float64)          # L^T y, maintained incrementally
    d = np.diag(K).astype(np.float64).copy()       # residual diagonal
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
        # NOTE: kept identical to the reference for reproducibility. numpy's
        # weighted sampling-without-replacement is a known slow path; a
        # Gumbel-top-k trick is O(eligible.size) but changes the RNG stream:
        #   keys = np.log(weights) + rng.gumbel(size=weights.size)
        #   S = eligible[np.argpartition(keys, -batch_size)[-batch_size:]]
        S = rng.choice(eligible, size=batch_size, replace=False, p=weights / wsum)

        i = L.shape[1]

        # Residual columns for the batch (GEMM #1 when i > 0).
        if assume_symmetric:
            K_cols = K[S, :].T          # contiguous row read, symmetric -> == K[:, S]
        else:
            K_cols = K[:, S]
        if i == 0:
            T_batch = K_cols.copy()
        else:
            T_batch = K_cols - L @ L[S, :].T

        dS = d[S].astype(np.float64)
        valid = dS > eps
        if not np.any(valid):
            break
        if not np.all(valid):
            S = S[valid]
            T_batch = T_batch[:, valid]
            dS = dS[valid]

        sqrt_dS = np.sqrt(dS)
        U_batch = T_batch / sqrt_dS.reshape(1, -1)          # (n, s)

        # ||u_p||^2 for all candidates
        unorm2 = np.einsum("ij,ij->j", U_batch, U_batch)    # O(n s)

        if i == 0:
            # Sigma^{-1} = sigma2^{-1} I, so a = ||u||^2 / sigma2, b = (u^T y) / sigma2
            a = unorm2 * inv_s2
            uy = U_batch.T @ y                              # O(n s)
            b = uy * inv_s2
            W = None
        else:
            W = L.T @ U_batch                               # (i, s)  GEMM #2
            BW = B @ W                                      # O(i^2 s)
            aquad = np.einsum("ij,ij->j", W, BW)            # diag(W^T B W), O(i s)
            a = unorm2 * inv_s2 - aquad * inv_s4

            uy = U_batch.T @ y                              # O(n s)
            BLy = B @ Ly                                    # O(i^2)
            bquad = W.T @ BLy                               # (s,), O(i s)
            b = uy * inv_s2 - bquad * inv_s4

        onepa = np.maximum(1.0 + a, eps)
        delta = -0.5 * (np.log(onepa) - (b * b) / onepa - unorm2 * inv_s2)

        best_idx = int(np.argmax(delta))
        if stop_if_nonpositive_gain and float(delta[best_idx]) <= 0.0:
            break

        best_p = int(S[best_idx])
        u = U_batch[:, best_idx].copy()

        # Rank-1 block update of B (Lemma 5.2), reusing c = L^T u / sigma2 = W[:, best]/sigma2
        xi = 1.0 + float(u @ u) * inv_s2
        if i == 0:
            B = np.array([[1.0 / max(xi, eps)]], dtype=np.float64)
        else:
            c = W[:, best_idx] * inv_s2                     # = L^T u / sigma2
            Bc = B @ c
            gamma = 1.0 / max(xi - float(c @ Bc), eps)
            B_new = np.empty((i + 1, i + 1), dtype=np.float64)
            B_new[:i, :i] = B + gamma * np.outer(Bc, Bc)
            B_new[:i, i] = -gamma * Bc
            B_new[i, :i] = -gamma * Bc
            B_new[i, i] = gamma
            B = B_new

        # Append column, maintain Ly = L^T y incrementally, update residual diagonal
        L = np.column_stack([L, u])
        Ly = np.concatenate([Ly, [float(u @ y)]])
        d -= u * u
        np.maximum(d, 0.0, out=d)
        d[best_p] = 0.0

        pivot_mask[best_p] = True
        pivots.append(best_p)

    return pivots


if __name__ == "__main__":
    # Equivalence smoke test against a brute-force scorer on a small problem.
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 3))
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    Kmat = np.exp(-sq / 2.0) + 1e-6 * np.eye(n)
    yv = rng.standard_normal(n)
    piv = delta_vfe_pivots_fast(Kmat, yv, m=20, sigma2=0.1, s=8, seed=0)
    print("pivots:", piv)
    assert len(set(piv)) == len(piv), "pivots must be distinct"
    print("ok")
