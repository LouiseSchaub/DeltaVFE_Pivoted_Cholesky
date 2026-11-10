"""
Benchmark harness for Delta-VFE pivoted Cholesky implementations.

Times three methods across a sweep of target ranks and reports, for each rank:
  * wall-clock time per method (median over repeats),
  * runtime ratio of each Delta-VFE variant to RPCholesky,
  * pivot agreement of each Delta-VFE variant with the reference scorer
    (prefix length over which the selected pivots are identical).

Drop in your own (K, y) for Abalone / QM7. K must be the full kernel matrix
(symmetric PSD); y the target vector. Hyperparameters (kernel lengthscale,
sigma2) should already be baked into K and sigma2 as in the paper's setup.

Usage:
    python benchmark_delta_vfe.py
or import `benchmark(K, y, sigma2, ranks=..., s=..., repeats=...)`.
"""

import time
import numpy as np

from delta_vfe_fast import delta_vfe_pivots_fast
from delta_vfe_single_gemm import delta_vfe_pivots_single_gemm


# --------------------------------------------------------------------------
# Baselines: RPCholesky and the faithful (slow) reference scorer.
# --------------------------------------------------------------------------

def rpcholesky_pivots(K, m, seed=0, eps=1e-14):
    """RPCholesky (Epperly et al.), row-wise factor build."""
    rng = np.random.default_rng(seed)
    diags = np.diag(K).copy()
    G = np.zeros((m, K.shape[0]), dtype=np.float64)
    pivots = []
    for i in range(m):
        eligible = np.where(diags > eps)[0]
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
    return pivots


def _woodbury_apply_matrix(L, B, sigma2, v):
    v = np.asarray(v, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if L.shape[1] == 0:
        return v / sigma2
    return (v / sigma2) - (L @ (B @ (L.T @ v))) / (sigma2 * sigma2)


def _update_B(B, L, u, sigma2, eps=1e-14):
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    r = L.shape[1]
    if r == 0:
        xi = 1.0 + float(u @ u) / sigma2
        return np.array([[1.0 / max(xi, eps)]])
    c = (L.T @ u) / sigma2
    xi = 1.0 + float(u @ u) / sigma2
    Bc = B @ c
    g = 1.0 / max(xi - float(c @ Bc), eps)
    Bn = np.empty((r + 1, r + 1))
    Bn[:r, :r] = B + g * np.outer(Bc, Bc)
    Bn[:r, r] = -g * Bc
    Bn[r, :r] = -g * Bc
    Bn[r, r] = g
    return Bn


def delta_vfe_reference(K, y, m, sigma2, s=64, seed=0, eps=1e-14):
    """Faithful reference scorer (materializes Z_batch); ground truth for pivots."""
    K = np.asarray(K, float)
    y = np.asarray(y, float).reshape(-1)
    n = K.shape[0]
    rng = np.random.default_rng(seed)
    L = np.zeros((n, 0))
    B = np.zeros((0, 0))
    d = np.diag(K).copy()
    mask = np.zeros(n, bool)
    piv = []
    for _ in range(m):
        C = np.flatnonzero(~mask)
        if C.size == 0:
            break
        el = C[d[C] > eps]
        if el.size == 0:
            break
        w = d[el]
        ws = float(w.sum())
        if ws <= 0:
            break
        S = rng.choice(el, size=min(s, el.size), replace=False, p=w / ws)
        v = _woodbury_apply_matrix(L, B, sigma2, y)[:, 0]
        Kb = K[:, S]
        Tb = Kb.copy() if L.shape[1] == 0 else Kb - (L @ L[S, :].T)
        dS = d[S]
        val = dS > eps
        if not np.any(val):
            break
        S, Tb, dS = S[val], Tb[:, val], dS[val]
        U = Tb / np.sqrt(dS).reshape(1, -1)
        Z = _woodbury_apply_matrix(L, B, sigma2, U)
        a = np.einsum("ij,ij->j", U, Z)
        b = U.T @ v
        un = np.einsum("ij,ij->j", U, U)
        opa = np.maximum(1.0 + a, eps)
        dl = -0.5 * (np.log(opa) - (b * b) / opa - un / sigma2)
        bi = int(np.argmax(dl))
        bp = int(S[bi])
        u = U[:, bi]
        B = _update_B(B, L, u, sigma2, eps)
        L = np.column_stack([L, u])
        d -= u * u
        np.maximum(d, 0, out=d)
        d[bp] = 0.0
        mask[bp] = True
        piv.append(bp)
    return piv


# --------------------------------------------------------------------------
# Timing utilities
# --------------------------------------------------------------------------

def _time_call(fn, repeats):
    """Median wall-clock over `repeats` calls (after one warmup)."""
    fn()  # warmup (touch BLAS, allocate)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _prefix_agree(a, b):
    k = 0
    for x, z in zip(a, b):
        if x == z:
            k += 1
        else:
            break
    return k


def benchmark(K, y, sigma2, ranks=(16, 32, 64, 128, 256, 512),
              s=16, seed=0, repeats=3, check_pivots=True):
    """
    Returns a list of dict rows, one per rank, and prints a table.

    Each row has: rank, t_rp, t_two, t_single, ratio_two, ratio_single,
    agree_two, agree_single (prefix agreement with the reference, or None).
    """
    K = np.asarray(K, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = K.shape[0]

    header = (f"{'rank':>6} | {'RPChol(s)':>10} {'2GEMM(s)':>10} {'1GEMM(s)':>10}"
              f" | {'x2GEMM':>7} {'x1GEMM':>7} | {'agree2':>10} {'agree1':>10}")
    print(f"n={n}, s={s}, sigma2={sigma2:g}, repeats={repeats}")
    print(header)
    print("-" * len(header))

    rows = []
    for r in ranks:
        if r > n:
            continue
        t_rp = _time_call(lambda: rpcholesky_pivots(K, r, seed=seed), repeats)
        t_two = _time_call(lambda: delta_vfe_pivots_fast(K, y, r, sigma2, s=s, seed=seed), repeats)
        t_sg = _time_call(lambda: delta_vfe_pivots_single_gemm(K, y, r, sigma2, s=s, seed=seed), repeats)

        agree_two = agree_sg = None
        if check_pivots:
            ref = delta_vfe_reference(K, y, r, sigma2, s=s, seed=seed)
            p_two = delta_vfe_pivots_fast(K, y, r, sigma2, s=s, seed=seed)
            p_sg = delta_vfe_pivots_single_gemm(K, y, r, sigma2, s=s, seed=seed)
            agree_two = f"{_prefix_agree(ref, p_two)}/{len(ref)}"
            agree_sg = f"{_prefix_agree(ref, p_sg)}/{len(ref)}"

        row = dict(rank=r, t_rp=t_rp, t_two=t_two, t_single=t_sg,
                   ratio_two=t_two / t_rp, ratio_single=t_sg / t_rp,
                   agree_two=agree_two, agree_single=agree_sg)
        rows.append(row)
        print(f"{r:>6} | {t_rp:>10.4f} {t_two:>10.4f} {t_sg:>10.4f}"
              f" | {row['ratio_two']:>7.2f} {row['ratio_single']:>7.2f}"
              f" | {str(agree_two):>10} {str(agree_sg):>10}")
    return rows


# --------------------------------------------------------------------------
# Demo on a synthetic Gaussian-kernel problem. Replace with Abalone / QM7.
# --------------------------------------------------------------------------

def _demo_problem(n=2000, dim=8, ell=1.0, sigma2=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim))
    sq = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    K = np.exp(-sq / (2.0 * ell * ell)) + 1e-6 * np.eye(n)
    y = rng.standard_normal(n)
    return K, y, sigma2


if __name__ == "__main__":
    K, y, sigma2 = _demo_problem(n=2000)
    benchmark(K, y, sigma2, ranks=(16, 32, 64, 128, 256, 512),
              s=16, seed=0, repeats=3, check_pivots=True)
