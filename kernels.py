import numpy as np

def laplacian_kernel(X, Z, ell, block_size=256):
    """

    :param X:
    :param Z:
    :param ell:
    :param block_size:
    :return:
    """
    """
    k(x,z) = exp(-||x-z||_1 / (ell^2))
    Computed in blocks to control memory.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ell = float(ell)
    if ell <= 0:
        raise ValueError("ell must be > 0")

    n, _ = X.shape
    m = Z.shape[0]
    K = np.empty((n, m), dtype=np.float64)

    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        Xi = X[i0:i1]
        l1 = np.sum(np.abs(Xi[:, None, :] - Z[None, :, :]), axis=2)
        K[i0:i1, :] = np.exp(-l1 / ell)
    return K

def gaussian_kernel(X, Z, ell, block_size=256):
    """

    :param X:
    :param Z:
    :param ell:
    :param block_size:
    :return:
    """
    """
    k(x,z) = exp(-||x-z||^2 / (2*ell^2))
    Computed in blocks to control memory.
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    ell = float(ell)
    if ell <= 0:
        raise ValueError("ell must be > 0")

    n, _ = X.shape
    m = Z.shape[0]
    K = np.empty((n, m), dtype=np.float64)

    ell2 = 2.0 * (ell * ell)

    for i0 in range(0, n, block_size):
        i1 = min(n, i0 + block_size)
        Xi = X[i0:i1]  # (b,d)

        # squared Euclidean distances in a stable vectorized form:
        # ||x-z||^2 = ||x||^2 + ||z||^2 - 2 x·z
        x2 = np.sum(Xi * Xi, axis=1, keepdims=True)          # (b,1)
        z2 = np.sum(Z * Z, axis=1, keepdims=True).T          # (1,m)
        dist2 = x2 + z2 - 2.0 * (Xi @ Z.T)                   # (b,m)
        dist2 = np.maximum(dist2, 0.0)                       # numerical guard

        K[i0:i1, :] = np.exp(-dist2 / ell2)

    return K


def precompute_inducing_kernels_laplacian(X_train, X_test, pivots, ell, block_size=256):
    """

    :param X_train:
    :param X_test:
    :param pivots:
    :param ell:
    :param block_size:
    :return:
    """
    pivots = list(pivots)
    Z = X_train[pivots, :]
    Kmm_full = laplacian_kernel(Z, Z, ell, block_size=block_size)
    Knm_full = laplacian_kernel(X_train, Z, ell, block_size=block_size)
    Ktm_full = laplacian_kernel(X_test, Z, ell, block_size=block_size)
    return Kmm_full, Knm_full, Ktm_full


def precompute_inducing_kernels_gaussian(X_train, X_test, pivots, ell, block_size=256):
    """

    :param X_train:
    :param X_test:
    :param pivots:
    :param ell:
    :param block_size:
    :return:
    """
    pivots = list(pivots)
    Z = X_train[pivots, :]
    Kmm_full = gaussian_kernel(Z, Z, ell, block_size=block_size)
    Knm_full = gaussian_kernel(X_train, Z, ell, block_size=block_size)
    Ktm_full = gaussian_kernel(X_test, Z, ell, block_size=block_size)
    return Kmm_full, Knm_full, Ktm_full