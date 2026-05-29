import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kernels import laplacian_kernel
from metrics import gp_neg_log_marginal_likelihood, gpr_predict_mean, rmse_mae


def load_hypers(fold:int=None):
    """
    Loading pre-computed hyperparameters.
    :param fold: QM7-fold. If no fold is given, then the Abalone hypers are loaded
    :return: hyperparameters kernel width, noise variance (squared)
    """
    if fold is None:
        df = pd.read_csv(f"Data/Abalone/Full/full_gaussian_hypers.dat")
    else:
        df = pd.read_csv(f"Data/QM7/Full/full_laplacian_fold={fold}_hypers.dat")
    ell = float(df["Kernelwidth"].iloc[0])
    sigma2 = float(df["Noise2"].iloc[0])
    return ell, sigma2


def optimize_laplacian_gp_hypers_no_variance(X_tr, y_tr, X_val=None, y_val=None, ell0=5.0, noise0=1e-2,
                                             bounds_ell=(1e-3, 1e4), bounds_noise=(1e-8, 1e2), maxiter=80, jitter=1e-8,
                                             objective="nll", kernel_block=256):
    """
    Hyperparameter optimization using L-BFGS-B.
    :param X_tr ((n,n)-ndarray): input data on training dataset.
    :param y_tr ((n,)-ndarray): output data vector on training dataset.
    :param X_val ((n_val, n_val)-ndarray): input data on validation dataset.
    :param y_val ((n_val)-ndarray): output data vector on validation dataset.
    :param ell0: initial value for kernel width
    :param noise0: initial value for noise variance
    :param bounds_ell: lower and upper bounds for kernel width.
    :param bounds_noise: lower and upper bounds for noise variance.
    :param maxiter: maximum number of iterations.
    :param jitter: small additive factor used for numerical stability.
    :param objective (str): specified objective function.
    :param kernel_block: compute block size
    :return: optimized hyperparameters and corresponding objective value.
    """
    x0 = np.array([np.log(ell0), np.log(noise0)], dtype=np.float64)
    bnds = [(np.log(bounds_ell[0]), np.log(bounds_ell[1])), (np.log(bounds_noise[0]), np.log(bounds_noise[1]))]

    def unpack(x):
        return float(np.exp(x[0])), float(np.exp(x[1]))

    def obj(x):
        ell, noise2 = unpack(x)
        try:
            K_tr = laplacian_kernel(X_tr, X_tr, ell, block_size=kernel_block)
        except Exception:
            return 1e50

        if objective == "nll":
            try:
                return gp_neg_log_marginal_likelihood(K_tr, y_tr, noise2, jitter=jitter)
            except np.linalg.LinAlgError:
                return 1e50

        if X_val is None or y_val is None:
            raise ValueError("objective='val_rmse' requires X_val and y_val.")

        try:
            K_val_tr = laplacian_kernel(X_val, X_tr, ell, block_size=kernel_block)
            mu_val = gpr_predict_mean(K_tr, K_val_tr, y_tr, noise2, jitter=jitter)
        except np.linalg.LinAlgError:
            return 1e50

        rmse, _ = rmse_mae(mu_val, y_val)
        return rmse

    res = minimize(obj, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": int(maxiter)})
    ell_opt, noise2_opt = unpack(res.x)
    return ell_opt, noise2_opt, res


