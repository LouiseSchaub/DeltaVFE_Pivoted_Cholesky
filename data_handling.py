import numpy as np
import os
import pandas as pd
import scipy
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def sort_coulomb_matrices_row_norm(X_cm):
    """
    Sort Coulomb matrix.

    :param X_cm: unsorted Coulomb matrix
    :return: Sorted Coulomb matrix.
    """
    X_cm = np.asarray(X_cm)
    n, a, b = X_cm.shape
    assert a == b, "Coulomb matrices must be square."

    Xs = np.empty_like(X_cm)
    for i in range(n):
        M = X_cm[i]
        norms = np.linalg.norm(M, axis=1)
        perm = np.argsort(-norms)
        Xs[i] = M[np.ix_(perm, perm)]
    return Xs


def _clean_index_array(arr, n):
    arr = np.asarray(arr).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr.astype(np.int64)

    arr = arr.astype(np.int64, copy=False)
    a_min, a_max = int(arr.min()), int(arr.max())

    # If looks 1-based, drop zeros then shift to 0-based
    if a_min >= 1 and a_max <= n:
        arr = arr[arr > 0]
        arr0 = arr - 1
        arr0 = arr0[(arr0 >= 0) & (arr0 < n)]
        return np.unique(arr0)

    # Otherwise assume 0-based
    arr = arr[(arr >= 0) & (arr < n)]
    return np.unique(arr)


def get_qm7_split_from_P(P, n, fold=0):
    """
    Handles common QM7 P formats:
      - (5,2,k) train/test lists padded with zeros
      - (2,5,k) same
      - (5,n) permutation per fold -> first 80% train
      - (5,k) test list per fold -> train = complement
      - object cell arrays with (5,2) or (2,5)
    """
    if P is None:
        return None

    P = np.asarray(P)

    if P.dtype == object:
        if P.ndim == 2 and P.shape[0] == 5 and P.shape[1] == 2:
            tr = _clean_index_array(P[fold, 0], n)
            te = _clean_index_array(P[fold, 1], n)
            if tr.size and te.size:
                return tr, te
        if P.ndim == 2 and P.shape[0] == 2 and P.shape[1] == 5:
            tr = _clean_index_array(P[0, fold], n)
            te = _clean_index_array(P[1, fold], n)
            if tr.size and te.size:
                return tr, te
        return None

    P_s = np.squeeze(P)

    if P_s.ndim == 3 and P_s.shape[0] == 5 and P_s.shape[1] == 2:
        tr = _clean_index_array(P_s[fold, 0, :], n)
        te = _clean_index_array(P_s[fold, 1, :], n)
        if tr.size and te.size:
            return tr, te

    if P_s.ndim == 3 and P_s.shape[0] == 2 and P_s.shape[1] == 5:
        tr = _clean_index_array(P_s[0, fold, :], n)
        te = _clean_index_array(P_s[1, fold, :], n)
        if tr.size and te.size:
            return tr, te

    # permutation per fold
    if P_s.ndim == 2 and P_s.shape[0] == 5 and P_s.shape[1] == n:
        perm = _clean_index_array(P_s[fold, :], n)
        if perm.size == n:
            n_train = int(0.8 * n)
            return perm[:n_train], perm[n_train:]

    # test-list per fold
    if P_s.ndim == 2 and P_s.shape[0] == 5:
        te = _clean_index_array(P_s[fold, :], n)
        if te.size:
            mask = np.ones(n, dtype=bool)
            mask[te] = False
            tr = np.where(mask)[0]
            return tr, te

    if P_s.ndim == 2 and P_s.shape[1] == 5:
        te = _clean_index_array(P_s[:, fold], n)
        if te.size:
            mask = np.ones(n, dtype=bool)
            mask[te] = False
            tr = np.where(mask)[0]
            return tr, te

    return None


def prepare_qm7(fold:int=0):
    """
    Dataset preparation for QM7.

    :param fold: one of five available folds that are predetermined for QM7
    :return: dataset split into train and text, i.e. X_train, X_test, y_train, y_test.
    """
    os.makedirs("Plots", exist_ok=True)
    os.makedirs("Data/QM7", exist_ok=True)

    url = "http://quantum-machine.org/data/qm7.mat"
    if not os.path.exists("qm7.mat"):
        urllib.request.urlretrieve(url, "qm7.mat")

    data = scipy.io.loadmat("qm7.mat")

    X_cm = data["X"]  # (n, 23, 23)
    y_raw = data["T"].reshape(-1).astype(np.float64)
    P = data.get("P", None)

    n, a, _ = X_cm.shape
    print(f"QM7 raw: X={X_cm.shape}, y={y_raw.shape}")
    print("y stats:", "min", float(y_raw.min()), "max", float(y_raw.max()),
          "mean", float(y_raw.mean()), "std", float(y_raw.std()))

    print("Sorting Coulomb matrices (row-norm) ...")
    X_cm_sorted = sort_coulomb_matrices_row_norm(X_cm)

    tri_u = np.triu_indices(a)
    X_raw = X_cm_sorted[:, tri_u[0], tri_u[1]].astype(np.float64)
    print(f"QM7 vectorized: X={X_raw.shape}")

    split = get_qm7_split_from_P(P, n, fold=fold)
    if split is None:
        print("WARNING: Could not parse P; falling back to random 80/20 split.")
        idx_all = np.arange(n)
        train_idx, test_idx = train_test_split(idx_all, test_size=0.2, random_state=0)
        train_idx = np.sort(train_idx)
        test_idx = np.sort(test_idx)
    else:
        train_idx, test_idx = split
        inter = np.intersect1d(train_idx, test_idx)
        print(f"Using P fold={fold}: train={train_idx.size}, test={test_idx.size}, intersection={inter.size}")

    X_train_raw, y_train_raw = X_raw[train_idx], y_raw[train_idx]
    X_test_raw, y_test_raw = X_raw[test_idx], y_raw[test_idx]

    scalerX = StandardScaler()
    X_train = scalerX.fit_transform(X_train_raw)
    X_test = scalerX.transform(X_test_raw)

    scalery = StandardScaler()
    y_train_std = scalery.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test_std = scalery.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    return X_train, X_test, y_train_std, y_test_std


def prepare_abalone():
    """
    Dataset preparation for the Abalone dataset.
    
    :return: dataset split into train and text, i.e. X_train, X_test, y_train, y_test.
    """
    os.makedirs("Plots", exist_ok=True)
    os.makedirs("Data/Abalone", exist_ok=True)

    abalone = fetch_ucirepo(id=1)
    X_df = abalone.data.features
    y_df = abalone.data.targets

    y = y_df.values.reshape(-1).astype(np.float64)

    # One-hot encode Sex
    X_df_enc = pd.get_dummies(X_df, columns=["Sex"], drop_first=True)
    X = X_df_enc.values.astype(np.float64)

    # Outer split: train/test
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"Abalone raw: train n={X_train_raw.shape[0]}, test n={X_test_raw.shape[0]}, d={X_train_raw.shape[1]}")

    # Inner split for hyperopt (RAW) + scalers fit ONLY on inner-train

    X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=1)

    scalerX_inner = StandardScaler()
    X_tr = scalerX_inner.fit_transform(X_tr_raw)
    X_val = scalerX_inner.transform(X_val_raw)

    scalery_inner = StandardScaler()
    y_tr = scalery_inner.fit_transform(y_tr_raw.reshape(-1, 1)).reshape(-1)
    y_val = scalery_inner.transform(y_val_raw.reshape(-1, 1)).reshape(-1)

    scalerX = StandardScaler()
    X_train = scalerX.fit_transform(X_train_raw)
    X_test = scalerX.transform(X_test_raw)

    scalery = StandardScaler()
    y_train_std = scalery.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test_std = scalery.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    return X_train, X_test, y_train_std, y_test_std
