import numpy as np

from data_handling import prepare_abalone, prepare_qm7
from kernels import gaussian_kernel, laplacian_kernel, precompute_inducing_kernels_laplacian, precompute_inducing_kernels_gaussian


def make_size_array(limit: int):
    """
    Generates an array made of powers of 2.

    :param limit: upper limit (not included) for the power of 2 array.
    :return: array with powers of 2 with specified limit.
    """
    max_limit = 2**14
    effective_limit = min(limit, max_limit)
    sizes = [2**i for i in range(int(np.log2(effective_limit)) + 1) if 2**i < effective_limit]
    sizes.append(effective_limit)
    return np.array(sizes, dtype=int)


def summarize(df, rank_value):
    """
    Summarizes the mean of RMSE, MAE, ELBO and TraceErr of a dataframe

    :param df: Dataframe to be summarized
    :param rank_value: specified rank to summarize at
    :return:
    """
    dfm = df[df["Rank"] == rank_value]
    if dfm.empty:
        return np.nan, np.nan, np.nan, np.nan
    return (float(dfm["RMSE"].mean()), float(dfm["MAE"].mean()), float(dfm["ELBO"].mean()),
            float(dfm["TraceErr"].mean()))


def build_L_from_pivots_paper(K: np.ndarray, pivots: list[int], eps: float = 1e-14) -> np.ndarray:
    """
    Use a sequence of pivots and reconstruct the (approximate) Cholesky factor from a Nystrom approximation.
    :param K (np.ndarray): full nxn kernel matrix.
    :param pivots: iterable list of pivots.
    :param eps: numerical safeguard.
    :return: reconstructed Cholesky factor.
    """

    K = np.asarray(K, dtype=np.float64)
    n = K.shape[0]

    L = np.zeros((n, 0), dtype=np.float64)
    d = np.diag(K).astype(np.float64).copy()

    for p in pivots:
        dp = float(d[p])
        if dp <= eps:
            raise RuntimeError(f"Pivot {p} selected with nonpositive residual diagonal.")

        if L.shape[1] == 0:
            tp = K[:, p].copy()
        else:
            tp = K[:, p] - (L @ L[p, :])

        up = tp / np.sqrt(dp)
        L = np.column_stack([L, up])

        d -= up * up
        np.maximum(d, 0.0, out=d)
        d[p] = 0.0

    return L


# Used for the comparison of pivot selection
def compute_B(L, sigma2):
    """
    Compute B = (I + sigma2^{-1} L^T L)^{-1} from scratch.

    :param L (np.ndarray): Cholesky factor.
    :param sigma2 (float): squared noise vairance
    :return: Direct computation of B
    """
    L = np.asarray(L, dtype=np.float64)
    r = L.shape[1]
    if r == 0:
        return np.zeros((0, 0), dtype=np.float64)
    C = np.eye(r, dtype=np.float64) + (L.T @ L) / sigma2
    return np.linalg.solve(C, np.eye(r, dtype=np.float64))


def dataset_config(dataset: str, fold=None):
    """
    Prepare the dataset and return the corresponding dataset split.

    :param dataset (str): Either "Abalone" or "QM7"
    :param fold: use None for Abalone. Otherwise, the specific fold integer for "QM7"
    :return:
    """
    assert dataset in ["Abalone", "QM7"], "Dataset must be either Abalone or QM7."

    if dataset == "Abalone":
        X_train, X_test, y_train_raw, y_test_raw = prepare_abalone()
        return (
            X_train, X_test, y_train_raw, y_test_raw, gaussian_kernel, precompute_inducing_kernels_gaussian, "",
            "Data/Abalone")
    elif dataset == "QM7":
        assert fold is not None, "QM7 requires a fold argument"
        X_train, X_test, y_train_raw, y_test_raw = prepare_qm7(fold=fold)
        return (X_train, X_test, y_train_raw, y_test_raw, laplacian_kernel, precompute_inducing_kernels_laplacian,
                f"_fold={fold}", "Data/QM7")