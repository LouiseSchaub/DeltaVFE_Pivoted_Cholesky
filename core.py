"""
This file is used to recreate the results represented in the paper "Variational Free Energy Pivot Selection for
Pivoted Cholesky". The core functions are included here and are reused over methods and datasets.
"""

import numpy as np
import os
import pandas as pd
import time

from tqdm.auto import trange
from sklearn.preprocessing import StandardScaler

from helpers import compute_B, make_size_array, summarize, dataset_config
from metrics import rmse_mae, vfe_bound, gpr_predict_mean, gp_neg_log_marginal_likelihood
from delta_vfe import woodbury_apply_matrix, delta_vfe_pivots_batched
from greedy_cholesky import greedy_pivots
from rpcholesky import rpcholesky_pivots

def evaluate_at_ranks(K_full: np.ndarray, Kmm_full: np.ndarray, Knm_full: np.ndarray, Ktm_full: np.ndarray,
                      sigma2:float, y_train: np.ndarray, y_test: np.ndarray, ranks, jitter: float=1e-8):
    """
    Evaluate metrics at a sequence of ranks for a pixed pivot set of size m. For each rank r <= m, the rank-r Nystrom
    approximation is rebuilt from scratch via a Cholesky factorization of K_ZZ[:r, :r] and a Woodbury solve.
    All quantities are recomputed at each rank to ensure comparability of the pivot sequence.

    :param K_full: full nxn kernel matrix
    :param Kmm_full: kernel matrix using m pivots
    :param Knm_full:
    :param Ktm_full:
    :param sigma2: squared noise variance
    :param y_train: output data vector from the training dataset.
    :param y_test: output data vector from the test dataset.
    :param ranks: list of ranks
    :param jitter: small additive factor used for numerical stability.
    :return:
    """

    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

    m = Kmm_full.shape[0]
    ranks = np.array(sorted(set(int(r) for r in ranks if 1 <= int(r) <= m)), dtype=int)
    out = {}
    trK = float(np.trace(K_full))

    for r in ranks:
        Kmm = Kmm_full[:r, :r].copy()
        Knm = Knm_full[:, :r]
        Ktm = Ktm_full[:, :r]

        Kmm[np.diag_indices_from(Kmm)] += jitter
        W = np.linalg.cholesky(Kmm)

        V_train = np.linalg.solve(W, Knm.T)
        L = V_train.T

        B = compute_B(L, sigma2)
        elbo_r = vfe_bound(K_full, L, B, sigma2, y_train)

        alpha_vec = woodbury_apply_matrix(L, B, sigma2, y_train)

        V_test = np.linalg.solve(W, Ktm.T)
        Phi_test = V_test.T

        beta = V_train @ alpha_vec
        mu = Phi_test @ beta

        rmse_r, mae_r = rmse_mae(mu, y_test)
        trLL = float(np.sum(L*L))
        trace_err_r = trK - trLL
        out[int(r)] = (float(rmse_r), float(mae_r), float(elbo_r), float(trace_err_r))

    return out

def run_method(method_name: str, pivot_fn, K_full: np.ndarray, X_train: np.ndarray, X_test: np.ndarray,
               kernel_precompute_fn, ell: float, sigma2: float, y_train_std: np.ndarray, y_test_std: np.ndarray,
               ranks, m: int, block_size: int=256, seeds=None):
    """

    :param method_name:
    :param pivot_fn:
    :param K_full:
    :param X_train:
    :param X_test:
    :param kernel_precompute_fn:
    :param ell:
    :param sigma2:
    :param y_train_std:
    :param y_test_std:
    :param ranks: rank sequence (to be evaluated at)
    :param m: max. target rank
    :param block_size:
    :param seeds: seed sequence
    :return: DataFrame
    """

    rows = []
    if seeds is None:
        seeds = np.random.default_rng(0).integers(0, 2 ** 31 - 1, size=15, dtype=np.int64)
    else:
        seeds = np.asarray(seeds, dtype=np.int64)

    runs = len(seeds)
    for run_id in trange(runs, desc=f"Runs ({method_name})", position=0):
        seed = int(seeds[run_id])

        t0 = time.time()
        piv = pivot_fn(seed)
        if len(piv) < m:
            m_eff = len(piv)
            use_ranks = ranks[ranks <= m_eff]
        else:
            piv = piv[:m]
            m_eff = m
            use_ranks = ranks

        if m_eff == 0:
            print(f"WARNING: pivot_fn returned empty list for seed {seed}, skipping run")
            continue
            
        Kmm_full, Knm_full, Ktm_full = kernel_precompute_fn(X_train, X_test, piv, ell, block_size=block_size)

        stats = evaluate_at_ranks(K_full, Kmm_full, Knm_full, Ktm_full, sigma2, y_train_std, y_test_std,
                                  ranks=use_ranks, jitter=1e-8)
        t1 = time.time()

        for r in use_ranks:
            rmse_r, mae_r, elbo_r, trace_err_r = stats[int(r)]
            rows.append({
                "Method": method_name,
                "RunID": run_id,
                "Seed": seed,
                "Rank": r,
                "Kernelwidth": ell,
                "Noise2": sigma2,
                "RMSE": rmse_r,
                "MAE": mae_r,
                "ELBO": elbo_r,
                "TraceErr": trace_err_r,
                "RunWallTimeSec": float(t1-t0),
                "m_eff": m_eff
            })

    return pd.DataFrame(rows)

def _baseline_cache_path(out_dir: str, fold_label, ell: float, sigma2: float):
    return f"{out_dir}/Full/baseline{fold_label}_kernelwidth={ell}_noise2={sigma2}.csv"

def _compute_or_load_baseline(K: np.ndarray, X_train: np.ndarray, X_test: np.ndarray, y_train_raw: np.ndarray,
                              y_test_raw: np.ndarray, y_train_std: np.ndarray, y_test_std: np.ndarray, scalery,
                              kernel_fn, ell: float, sigma2: float, out_dir: str, fold_label, block_size: int,
                              force_recompute:bool):
    cache_path = _baseline_cache_path(out_dir, fold_label, ell, sigma2)

    if not force_recompute and os.path.exists(cache_path):
        row = pd.read_csv(cache_path).iloc[0].to_dict()
        print(f"Baseline loaded from cache: {cache_path}")
        return row

    print("Computing full-GP baseline....")
    K_s_full = kernel_fn(X_test, X_train, ell, block_size=block_size)
    mu_std = gpr_predict_mean(K, K_s_full, y_train_std, sigma2)
    rmse_std, mae_std = rmse_mae(mu_std, y_test_std)

    mu_orig = scalery.inverse_transform(mu_std.reshape(-1, 1)).reshape(-1)
    rmse_orig, mae_orig = rmse_mae(mu_orig, y_test_raw)

    logml = -gp_neg_log_marginal_likelihood(K, y_train_std, sigma2)
    trK = float(np.trace(K))

    row = {
        "Kernelwidth": float(ell),
        "Noise2": float(sigma2),
        "TrK": trK,
        "n_train": int(X_train.shape[0]),
        "RMSE_std": float(rmse_std), "MAE_std": float(mae_std),
        "RMSE_orig": float(rmse_orig), "MAE_orig": float(mae_orig),
        "Logml": float(logml),
    }

    os.makedirs(f"{out_dir}/Full", exist_ok=True)
    pd.DataFrame([row]).to_csv(cache_path, index=False)
    print(f"Wrote Baseline: {cache_path}")

    return row


def run_full_experiment(*, dataset: str, fold=None, ell: float, sigma2: float, m: int= 2048, s_values,
                        runs: int = 15, master_seed: int = 42, block_size: int = 256, run_greedy: bool = True,
                        run_rpcholesky: bool =True, run_deltavfe: bool = True, skip_baseline: bool = False,
                        force_recompute_baseline: bool = False):
    """

    :param dataset:
    :param fold:
    :param ell:
    :param sigma2:
    :param kernel_precompute_fn:
    :param m:
    :param s_values:
    :param runs:
    :param master_seed:
    :param block_size:
    :param run_greedy:
    :param run_rpcholesky:
    :param run_deltavfe:
    :param skip_baseline:
    :param force_recompute_baseline:
    :return:
    """

    # Load data and dataset-specific config
    X_train, X_test, y_train_raw, y_test_raw, kernel_fn, kernel_precompute_fn, fold_label, out_dir = dataset_config(dataset, fold=fold, block_size=block_size)

    print(f"\n{'=' * 42}")
    print(f"Experiment: {dataset}{fold_label}")
    print(f"  ell={ell}, sigma2={sigma2}, m={m}, runs={runs}")
    print(f"{'=' * 42}")

    # Create output subdirs
    for sub in ("Full", "Greedy", "RPCholesky", "DeltaVFE"):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)

    # Standardise y
    scalery = StandardScaler()
    y_train_std = scalery.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test_std = scalery.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    # Build full K
    print("Building full K (train-train) ...")
    K = kernel_fn(X_train, X_train, ell, block_size=block_size)

    # Baseline
    baseline = None
    if not skip_baseline:
        baseline = _compute_or_load_baseline(K, X_train, X_test, y_train_raw, y_test_raw, y_train_std, y_test_std,
                                             scalery, kernel_fn, ell, sigma2, out_dir, fold_label, block_size,
                                             force_recompute_baseline)
        print(f"RMSE_std={baseline["RMSE_std"]:.6f}, MAE_std={baseline["MAE_std"]:.6f}, logml={baseline["Logml"]:.6f}")

    # Create rank grid
    ranks = make_size_array(m)

    # Shared ssed list
    shared_seeds = np.random.default_rng(master_seed).integers(0, 2 ** 31 - 1, size=runs, dtype=np.int64)
    seed_path = f"{out_dir}/Full/seeds{fold_label}_master={master_seed}.csv"
    pd.DataFrame({"RunID": np.arange(runs), "Seed": shared_seeds}).to_csv(seed_path, index=False)
    print(f"Wrote seeds: {seed_path}")

    common_kwargs = dict(K_full=K, X_train=X_train, X_test=X_test, kernel_precompute_fn=kernel_precompute_fn, ell=ell,
                         sigma2=sigma2, y_train_std=y_train_std, y_test_std=y_test_std, ranks=ranks, m=m,
                         block_size=block_size, seeds=shared_seeds)

    def _csv_path(subdir, base, s=None):
        s_part = f"_s={s}" if s is not None else ""
        return (
            f"{out_dir}/{subdir}/"
            f"{base}{fold_label}_kernelwidth={ell}_noise2={sigma2}_m={m}{s_part}.csv"
        )

    # === RUN METHODS ===
    if run_greedy:
        df_greedy = run_method(method_name="Greedy", pivot_fn=lambda seed: greedy_pivots(K,m), **common_kwargs)
        path = _csv_path("Greedy", "greedy")
        df_greedy.to_csv(path, index=False)
        print(f"Wrote: {path}")

    if run_rpcholesky:
        df_rp = run_method(method_name="RPCholesky", pivot_fn=lambda seed: rpcholesky_pivots(K, m, seed=seed),
                           **common_kwargs)
        path = _csv_path("RPCholesky", "rpcholesky")
        df_rp.to_csv(path, index=False)
        print(f"Wrote: {path}")

    if run_deltavfe:
        all_vfe = []
        for s in s_values:
            print(f"\n--- DeltaVFE s={s} ---")
            df_vfe = run_method(method_name=f"DeltaVFE_s={s}",
                                pivot_fn=lambda seed, s=s: delta_vfe_pivots_batched(K, y_train_std, m, sigma2, s=s,
                                                                                    seed=seed,
                                                                                    stop_if_nonpositive_gain=True,),
                **common_kwargs)
            path = _csv_path("DeltaVFE", "deltavfe", s=s)
            df_vfe.to_csv(path, index=False)
            print(f"Wrote: {path}")

            rm, ma, el, te = summarize(df_vfe, m)
            print(f"  s={s} at rank m={m}: "
                  f"RMSE={rm:.4f}  MAE={ma:.4f}  ELBO={el:.4f}  TraceErr={te:.4f}")
            all_vfe.append(df_vfe)

            if all_vfe:
                df_vfe_all = pd.concat(all_vfe, ignore_index=True)
                path = (
                    f"{out_dir}/DeltaVFE/"
                    f"deltavfe_ALLS{fold_label}_kernelwidth={ell}_noise2={sigma2}_m={m}.csv"
                )
                df_vfe_all.to_csv(path, index=False)
                print(f"Wrote combined: {path}")

    return {"baseline": baseline, "seeds": shared_seeds}
