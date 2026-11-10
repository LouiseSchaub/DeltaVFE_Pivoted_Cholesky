"""
Prediction error against the rank, one figure per dataset. The error measure is set in plot_utils.DATASETS: RMSE
(Abalone), MAE in kcal/mol (QM7), SMSE (KIN40K), MAE in eV (QM9); the dotted line is the full GP. If the results contain
MSLL (written by the current core.py), it is shown in a second panel.

Usage: python plot_prediction.py [DATASET ...]
       Without arguments a figure is made for every dataset with results (Abalone, QM7, KIN40K, QM9).
Output: Plots/<dataset>_prediction_error.pdf
"""
import os
import sys

from matplotlib import gridspec
import matplotlib.pyplot as plt

from plot_utils import (DATASETS, FIGSIZE_ROW, GRID_ROW, PLOTS_DIR, baseline_value, datasets_from_argv, error_metric,
                        load_results, plot_metric, save_show, shared_legend)


def plot_prediction(df, base, dataset, outfile=None):
    """
    :param df: results of one dataset from plot_utils.load_results.
    :param base: full-GP baseline row from plot_utils.load_results.
    :param dataset: dataset name.
    :param outfile: file name of the figure.
    """
    metric = error_metric(df, dataset)
    panels = [metric] + (["MSLL"] if "MSLL" in df.columns and df["MSLL"].notna().any() else [])
    fig = plt.figure(figsize=FIGSIZE_ROW)
    gs = gridspec.GridSpec(2, len(panels), **GRID_ROW)
    axes = []
    for j, m in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        plot_metric(ax, df, m, dataset, baseline=baseline_value(base, m) if m == metric else None)
        axes.append(ax)
    fig.suptitle(dataset)
    shared_legend(fig.add_subplot(gs[1, :]), axes, ncol=4)
    save_show(fig, outfile)
    return fig, axes


if __name__ == "__main__":
    for dataset in datasets_from_argv(sys.argv[1:], default=list(DATASETS)):
        try:
            df, base = load_results(dataset)
        except FileNotFoundError as e:
            print(f"Skipping {dataset}: {e}")
            continue
        plot_prediction(df, base, dataset, outfile=os.path.join(PLOTS_DIR, f"{dataset.lower()}_prediction_error.pdf"))