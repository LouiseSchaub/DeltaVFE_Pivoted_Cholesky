"""
Add the column KernelColumns (distinct kernel columns computed up to each rank) to Delta-VFE result files written
before core.run_method recorded it.

Only the pivot selection is repeated, with the seeds stored in each file (same seed -> same candidate draws and
pivots); the other columns are not touched. Greedy and RPCholesky compute one new column per pivot, so their count is
the rank and needs no file change.

Use from a script:

    from add_kernel_columns import add_kernel_columns
    add_kernel_columns("QM7", s=[16])            # only s = 16
    add_kernel_columns("QM7")                    # all Delta-VFE files of QM7 without the column

Running this file (e.g. the run button of an IDE) updates the s shown by plot_results.py (plot_utils.DATASETS) for
every dataset that has results; edit JOBS below to choose other datasets or s.
"""
import os
import re

import pandas as pd
from sklearn.preprocessing import StandardScaler

from core import deltavfe_with_columns
from helpers import dataset_config
from plot_utils import DATASETS, method_files

# (dataset, list of s or None for all s); default: the s used by plot_results.py
JOBS = [(name, [info["s"]]) for name, info in DATASETS.items()]


def add_kernel_columns(dataset: str, s=None, block_size: int = 64) -> list[str]:
    """
    Add KernelColumns to the Delta-VFE result files of a dataset that do not have it yet.

    :param dataset (str): "Abalone", "QM7", "KIN40K" or "QM9".
    :param s: batch factor(s) to update (int or list of int); None updates all available s.
    :param block_size (int): rows per kernel block when building K (smaller needs less memory; same values).
    :return: list of updated files.
    """
    s_list = [s] if isinstance(s, int) else (s or ["*"])
    files = [f for v in s_list for f in method_files(dataset, "DeltaVFE", s=v)]
    files = [f for f in files if "KernelColumns" not in pd.read_csv(f, nrows=0).columns]
    if not files:
        print(f"{dataset}: nothing to do.")
        return []

    X_train, _, y_train_raw, _, kernel_fn, _, _, _ = dataset_config(dataset, block_size=block_size)
    y_train_std = StandardScaler().fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    K, K_ell = None, None
    updated = []
    for f in files:
        df = pd.read_csv(f)
        ell, sigma2 = float(df["Kernelwidth"].iloc[0]), float(df["Noise2"].iloc[0])
        s_f = int(re.search(r"_s=(\d+)", os.path.basename(f)).group(1))
        m = int(re.search(r"_m=(\d+)", os.path.basename(f)).group(1))
        if K is None or K_ell != ell:
            K, K_ell = kernel_fn(X_train, X_train, ell, block_size=block_size), ell
        counts = {}
        for run_id, seed in df.groupby("RunID")["Seed"].first().items():
            piv, cols = deltavfe_with_columns(K, y_train_std, m, sigma2, s_f, int(seed))
            m_eff = int(df.loc[df["RunID"] == run_id, "m_eff"].iloc[0])
            if len(piv[:m]) != m_eff:
                raise RuntimeError(f"{f}, run {run_id}: selection gives {len(piv)} pivots, file has m_eff={m_eff} "
                                   "(different data or hyperparameters?)")
            counts[run_id] = cols
            print(f"  {dataset} s={s_f} run {run_id}: {cols[-1]} distinct columns for {len(piv)} pivots", flush=True)
        df["KernelColumns"] = [int(counts[rid][int(r) - 1]) for rid, r in zip(df["RunID"], df["Rank"])]
        df.to_csv(f + ".tmp", index=False)
        os.replace(f + ".tmp", f)
        print(f"Updated: {f}")
        updated.append(f)
    return updated


if __name__ == "__main__":
    for name, s_values in JOBS:
        if method_files(name, "DeltaVFE"):
            add_kernel_columns(name, s=s_values)
        else:
            print(f"{name}: no Delta-VFE results, skipped.")