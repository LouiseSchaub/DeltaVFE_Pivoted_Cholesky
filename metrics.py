import numpy as np

def gpr_predict_mean(K_train, K_test_train, y_train, noise2, jitter=1e-8):
    """
    Computes predictive mean in Gaussian process regression.

    :param K_train: kernel matrix on the training data points.
    :param K_test_train: kernel matrix evaluated on train and test data points.
    :param y_train: output data points of the training data.
    :param noise2: squared noise variance
    :param jitter: small factor used for numerical stability.
    :return: predictive mean
    """
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    A = K_train.copy()
    A[np.diag_indices_from(A)] += (noise2 + jitter)

    L = np.linalg.cholesky(A)
    tmp = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, tmp)

    return (K_test_train @ alpha).reshape(-1)


def rmse_mae(y_pred, y_true):
    """
    Computes root mean squared error (RMSE) and mean absolute error (MAE)

    :param y_pred: predicted data vector.
    :param y_true: true data vector.
    :return: rmse, mae values
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def gp_neg_log_marginal_likelihood(K: np.ndarray, y: np.ndarray, noise2: float, jitter: float=1e-8):
    """


    :param K: Kernel matrix.
    :param y: Corresponding output data vector.
    :param noise2: squared noise variance.
    :param jitter: small additive factor for numerical stability.
    :return: -LML
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    A = K.copy()
    A[np.diag_indices_from(A)] += (noise2 + jitter)

    L = np.linalg.cholesky(A)
    tmp = np.linalg.solve(L, y)
    alpha = np.linalg.solve(L.T, tmp)

    quad = float(y.T @ alpha)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    n = y.shape[0]
    return 0.5 * (quad + logdet + n * np.log(2.0 * np.pi))


def _woodbury_apply(L: np.ndarray, B: np.ndarray, sigma2: float, v: np.ndarray):
    """
    Apply (sigma2 I + L L^T)^(-1) to v using the Woodbury identity.

    :param L (np.ndarray): approximate Cholesky factor
    :param B (np.ndarray): B = (I + sigma2^{-1} L^T L)^{-1}
    :param sigma2 (float): squared noise variance.
    :param v (np.ndarray): vector to apply the inverse to.
    :return: result of application.
    """
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    if L.shape[1] == 0:
        return v / sigma2

    Lt_v = L.T @ v
    return (v / sigma2) - (L @ (B @ Lt_v)) / (sigma2 * sigma2)


def vfe_bound(K, L, B, sigma2, y):
    """
    Compute the VFE lower bound for covariance approximation K \approx L L^T.

    :param K (np.ndarray): kernel matrix.
    :param L (np.ndarray): Cholesky factor.
    :param B (np.ndarray): B = (I + sigma2^{-1} L^T L).
    :param sigma2 (float): squared noise variance.
    :param y (np.ndarray): Corresponding output data vector.
    :return: lower bound of VFE approximation.
    """
    K = np.asarray(K, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n = K.shape[0]
    if K.ndim != 2 or K.shape[1] != n:
        raise ValueError("K must be square.")
    if y.shape[0] != n:
        raise ValueError("y must have length n.")
    if L.shape[0] != n:
        raise ValueError("L must have shape (n, r).")
    if B.shape != (L.shape[1], L.shape[1]):
        raise ValueError("B must have shape (r, r).")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    r = L.shape[1]

    if r == 0:
        logdet = n * np.log(sigma2)
    else:
        C = np.eye(r, dtype=np.float64) + (L.T @ L) / sigma2
        sign, logdetC = np.linalg.slogdet(C)
        if sign <= 0:
            raise RuntimeError("C is not positive definite.")
        logdet = n * np.log(sigma2) + logdetC

    alpha = _woodbury_apply(L, B, sigma2, y)
    quad = float(y @ alpha)

    trK = float(np.trace(K))
    trLL = float(np.einsum("ij,ij->", L, L))
    trace_pen = (trK - trLL) / sigma2

    const = n * np.log(2.0 * np.pi)
    return -0.5 * (const + logdet + quad + trace_pen)


