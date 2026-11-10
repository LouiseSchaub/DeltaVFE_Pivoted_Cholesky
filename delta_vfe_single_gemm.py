import numpy as np


def delta_vfe_pivots_single_gemm(
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
    Single-GEMM realization of Algorithm 5.2 (Delta-VFE pivoted Cholesky).

    Reduces the per-iteration cost to ONE big O(n i s) GEMM (P = L^T K[:, S]),
    by scoring all candidates in the (i, s) space instead of forming the full
    (n, s) residual block. Everything else is O(i^2 s), O(n s) reads, or O(n i)
    for the single selected column.

    Quantities (residual t_p = K[:, p] - L g_p with g_p = L[p, :]^T, u_p = t_p / sqrt(d_p)):

        w_p   = L^T u_p          = (P[:, p] - M g_p) / sqrt(d_p)        M = L^T L
        ||t_p||^2                = ||K[:, p]||^2 - 2 P[:, p]^T g_p + g_p^T M g_p
        t_p^T y                  = (K^T y)[p] - g_p^T (L^T y)

        a_p   = sigma2^{-1} ||u_p||^2 - sigma2^{-2} w_p^T B w_p
        b_p   = sigma2^{-1} (u_p^T y) - sigma2^{-2} w_p^T B (L^T y)

    The single GEMM is K[S, :] @ L (using K symmetric so K[:, S] = K[S, :]^T),
    which also yields ||K[:, p]||^2 and (K^T y)[p] from the same contiguous read.

    NUMERICAL NOTE: ||t_p||^2 and w_p are formed by cancellation rather than by
    subtracting vectors of length n. Near convergence (small residual) or for
    ill-conditioned M = L^T L this can lose precision, so pivots may diverge from
    the reference at high rank even though the formulas are algebraically exact.
    The two-GEMM variant (which forms t_p explicitly) is numerically safer; prefer
    this version when n is large enough that the single remaining GEMM dominates.

    Returns the same pivots as the reference for the same seed up to floating point.
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
    if not assume_symmetric:
        # The single-GEMM trick relies on K[:, S] == K[S, :]^T. For a genuinely
        # non-symmetric A, use the two-GEMM variant instead.
        raise NotImplementedError("single-GEMM variant assumes a symmetric matrix.")

    inv_s2 = 1.0 / sigma2
    inv_s4 = inv_s2 * inv_s2

    rng = np.random.default_rng(seed)

    # State
    L = np.zeros((n, 0), dtype=np.float64)
    B = np.zeros((0, 0), dtype=np.float64)
    M = np.zeros((0, 0), dtype=np.float64)         # M = L^T L
    Ly = np.zeros((0,), dtype=np.float64)          # L^T y
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
        # Kept identical to the reference for reproducibility.
        S = rng.choice(eligible, size=batch_size, replace=False, p=weights / wsum)

        dS = d[S].astype(np.float64)
        valid = dS > eps
        if not np.any(valid):
            break
        if not np.all(valid):
            S = S[valid]
            dS = dS[valid]

        sqrt_dS = np.sqrt(dS)
        i = L.shape[1]

        # Single contiguous read of s rows; symmetry gives K[:, S] = K_rows^T.
        K_rows = K[S, :]                                   # (s, n)
        Kcol_sqnorm = np.einsum("ij,ij->i", K_rows, K_rows)  # ||K[:, p]||^2  (s,)
        Ky_S = K_rows @ y                                   # (K^T y)[p]      (s,)

        if i == 0:
            t_sqnorm = Kcol_sqnorm
            ty = Ky_S
            W = np.zeros((0, S.size), dtype=np.float64)
            a = t_sqnorm * inv_s2 / dS                      # ||u||^2 / sigma2
            unorm2 = t_sqnorm / dS
            b = (ty / sqrt_dS) * inv_s2
        else:
            P = (K_rows @ L).T                              # L^T K[:, S]  (i, s)  GEMM
            G_S = L[S, :].T                                 # g_p columns  (i, s)
            MG = M @ G_S                                    # O(i^2 s)

            W = (P - MG) / sqrt_dS.reshape(1, -1)           # w_p = L^T u_p  (i, s)

            PG = np.einsum("ij,ij->j", P, G_S)              # P[:, p]^T g_p   (s,)
            GMG = np.einsum("ij,ij->j", G_S, MG)            # g_p^T M g_p     (s,)
            t_sqnorm = Kcol_sqnorm - 2.0 * PG + GMG
            np.maximum(t_sqnorm, 0.0, out=t_sqnorm)         # guard cancellation
            unorm2 = t_sqnorm / dS

            ty = Ky_S - G_S.T @ Ly                          # t_p^T y         (s,)
            uy = ty / sqrt_dS

            BW = B @ W                                      # O(i^2 s)
            aquad = np.einsum("ij,ij->j", W, BW)            # w_p^T B w_p     (s,)
            a = unorm2 * inv_s2 - aquad * inv_s4

            BLy = B @ Ly                                    # O(i^2)
            bquad = W.T @ BLy                               # w_p^T B (L^T y) (s,)
            b = uy * inv_s2 - bquad * inv_s4

        onepa = np.maximum(1.0 + a, eps)
        delta = -0.5 * (np.log(onepa) - (b * b) / onepa - unorm2 * inv_s2)

        best_idx = int(np.argmax(delta))
        if stop_if_nonpositive_gain and float(delta[best_idx]) <= 0.0:
            break

        best_p = int(S[best_idx])
        unorm2_best = float(unorm2[best_idx])

        # Recompute the single selected column (one O(n i) column, like RPCholesky).
        if i == 0:
            t = K_rows[best_idx, :].copy()
            w_best = np.zeros((0,), dtype=np.float64)
        else:
            g_best = L[best_p, :]                           # (i,)
            t = K_rows[best_idx, :] - L @ g_best            # K[:, best_p] - L g_best
            w_best = W[:, best_idx]                         # L^T u_best
        u = t / sqrt_dS[best_idx]

        # Rank-1 block update of B (Lemma 5.2), reusing c = L^T u / sigma2 = w_best / sigma2
        xi = 1.0 + unorm2_best * inv_s2
        if i == 0:
            B = np.array([[1.0 / max(xi, eps)]], dtype=np.float64)
            M = np.array([[unorm2_best]], dtype=np.float64)
            uy_best = float(t @ y) / sqrt_dS[best_idx]
        else:
            c = w_best * inv_s2
            Bc = B @ c
            gamma = 1.0 / max(xi - float(c @ Bc), eps)
            B_new = np.empty((i + 1, i + 1), dtype=np.float64)
            B_new[:i, :i] = B + gamma * np.outer(Bc, Bc)
            B_new[:i, i] = -gamma * Bc
            B_new[i, :i] = -gamma * Bc
            B_new[i, i] = gamma
            B = B_new

            # M_new = [[M, w_best], [w_best^T, ||u||^2]]
            M_new = np.empty((i + 1, i + 1), dtype=np.float64)
            M_new[:i, :i] = M
            M_new[:i, i] = w_best
            M_new[i, :i] = w_best
            M_new[i, i] = unorm2_best
            M = M_new

            uy_best = float(ty[best_idx]) / sqrt_dS[best_idx]

        # Append column, maintain L^T y, update residual diagonal
        L = np.column_stack([L, u])
        Ly = np.concatenate([Ly, [uy_best]])
        d -= u * u
        np.maximum(d, 0.0, out=d)
        d[best_p] = 0.0

        pivot_mask[best_p] = True
        pivots.append(best_p)

    return pivots


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 200
    X = rng.standard_normal((n, 3))
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    Kmat = np.exp(-sq / 2.0) + 1e-6 * np.eye(n)
    yv = rng.standard_normal(n)
    piv = delta_vfe_pivots_single_gemm(Kmat, yv, m=20, sigma2=0.1, s=8, seed=0)
    print("pivots:", piv)
    assert len(set(piv)) == len(piv)
    print("ok")
