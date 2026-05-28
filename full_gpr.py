import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler

from helpers import dataset_config
from hyperparameters import load_hypers
from metrics import gpr_predict_mean, rmse_mae, gp_neg_log_marginal_likelihood


def full_gpr(dataset:str, fold=None, block_size: int = 256):
    """

    :param dataset:
    :param fold:
    :param block_size:
    :return:
    """
    assert dataset in ["Abalone", "QM7"], "Dataset must be either Abalone or QM7"
    if dataset == "Abalone":
        ell, sigma2 = load_hypers()
    else:
        ell, sigma2 = load_hypers(fold=fold)
    print("Loaded hypers:", "ell=", ell, "noise2=", sigma2)

    X_train, X_test, y_train_raw, y_test_raw, kernel_fn, fold_label, out_dir = dataset_config(dataset, fold)

    # Standardize y
    scalery = StandardScaler()
    y_train_std = scalery.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test_std = scalery.transfrom(y_test_raw.reshape(-1, 1)).reshape(-1)

    # BUild Kernel matrices
    K = kernel_fn(X_train, X_train, ell, block_size=block_size)
    K_s_full = kernel_fn(X_test, X_train, ell, block_size=block_size)

    trK = float(np.trace(K))

    os.makedirs(f"{out_dir}/Full", exist_ok=True)
    trace_row = {"Kernelwidth": float(ell), "Noise2": float(sigma2), "TrK": trK, "n_train":int(X_train.shape[0])}

    if fold is not None:
        trace_row["Fold"] = int(fold)
    out_trace = f"{out_dir}/Full/trace_kernelwidth={ell}_noise2={sigma2}_fold={fold}.csv"
    pd.DataFrame([trace_row]).to_csv(out_trace, index=False)
    print("Wrote:", out_trace)

    # Full GP prediction
    mu_std = gpr_predict_mean(K, K_s_full, y_train_std, sigma2)
    rmse_std, mae_std = rmse_mae(mu_std, y_test_std)
    print(f"Full GP (std y):  RMSE={rmse_std:.4f}, MAE={mae_std:.4f}")

    # Original-scale RMSE/MAE
    mu_orig = scalery.inverse_transform(mu_std.reshape(-1, 1)).reshape(-1)
    rmse_orig, mae_orig = rmse_mae(mu_orig, y_test_raw)
    print(f"Full GP (orig y): RMSE={rmse_orig:.4f}, MAE={mae_orig:.4f}")

    # Log marginal likelihood
    logml = -gp_neg_log_marginal_likelihood(K, y_train_std, sigma2)
    print(f"Full GP log marginal likelihood: {logml:.6f}")

    return {"rmse_std": rmse_std, "mae_std": mae_std, "rmse_orig": rmse_orig, "mae_orig": mae_orig, "logml": logml, "trK": trK}