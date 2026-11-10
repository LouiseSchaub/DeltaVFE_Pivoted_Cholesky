"""
Results of one dataset in a four-panel figure:
    - VFE gap per training point (F(K) - L(K~)) / n in nats against the rank          (top left)
    - relative trace-norm error |tr(K) - tr(K~)| / tr(K) against the rank             (top right)
    - prediction error against the rank (bottom left): RMSE (Abalone), MAE in kcal/mol (QM7),
      SMSE (KIN40K), MAE in eV (QM9), set in plot_utils.DATASETS
    - VFE gap against the number of kernel columns evaluated during the selection     (bottom right):
      one column per pivot for Greedy and RPCholesky, s columns per pivot for Delta-VFE

Usage: python plot_results.py [DATASET ...]
       Without arguments a figure is made for every dataset with results (Abalone, QM7, KIN40K, QM9).
Output: Plots/<Dataset>/<dataset>_results.pdf, e.g. Plots/QM7/qm7_results.pdf
"""
import os
import sys

from matplotlib import gridspec
import matplotlib.pyplot as plt

from plot_utils import (DATASETS, FIGSIZE_TWO_ROWS, GRID_TWO_ROWS, METHOD_LABELS, PLOTS_DIR, datasets_from_argv, error_metric, load_results, plot_metric, plot_metric_vs_cost, save_show,
                        shared_legend)

VFE_METRIC = "VFEGapPerPoint"     # alternatives: "VFEGap" (nats), "VFEGapRel" (relative to |F(K)|)
TRACE_METRIC = "TraceErrRel"
COST_METRIC = VFE_METRIC          # metric shown against the number of kernel columns (bottom right)
ADD_KERNEL_COLUMNS = True         # add missing kernel-column counts to Delta-VFE result files before plotting
                                  # (add_kernel_columns.py; repeats only the pivot selection, can take minutes)


def load_with_kernel_columns(dataset):
    """
    load_results, and if the plotted Delta-VFE results have no KernelColumns yet (files written before the column
    was recorded), add it with add_kernel_columns for those s and load again.
    """
    df, base = load_results(dataset)
    dvfe = df[df["Method"] == "DeltaVFE"]
    if ADD_KERNEL_COLUMNS and not dvfe.empty and dvfe["KernelColumns"].isna().any():
        from add_kernel_columns import add_kernel_columns
        s_missing = sorted(set(dvfe.loc[dvfe["KernelColumns"].isna(), "s"].dropna().astype(int)))
        print(f"{dataset}: adding kernel-column counts for s={s_missing} ...")
        add_kernel_columns(dataset, s=s_missing)
        df, base = load_results(dataset)
    return df, base


def plot_results(df, base, dataset, outfile=None):
    """
    :param df: results of one dataset from plot_utils.load_results.
    :param base: full-GP baseline row from plot_utils.load_results (no longer plotted, kept for existing callers).
    :param dataset: dataset name.
    :param outfile: file name of the figure.
    """
    err = error_metric(df, dataset)
    fig = plt.figure(figsize=FIGSIZE_TWO_ROWS)
    gs = gridspec.GridSpec(3, 2, **GRID_TWO_ROWS)
    ax_vfe, ax_trace = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_err, ax_cost = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    plot_metric(ax_vfe, df, VFE_METRIC, dataset)
    plot_metric(ax_trace, df, TRACE_METRIC, dataset)
    plot_metric(ax_err, df, err, dataset)
    plot_metric_vs_cost(ax_cost, df, COST_METRIC, dataset)
    axes = [ax_vfe, ax_trace, ax_err, ax_cost]
    # batch factor of Delta-VFE in its legend entry (no figure title)
    s_used = sorted(set(df.loc[df["Method"] == "DeltaVFE", "s"].dropna().astype(int)))
    if s_used:
        dvfe = METHOD_LABELS["DeltaVFE"]
        new = dvfe + " (" + ", ".join(f"$s={v}$" for v in s_used) + ")"
        for ax in axes:
            for h, label in zip(*ax.get_legend_handles_labels()):
                if label == dvfe:
                    h.set_label(new)
    shared_legend(fig.add_subplot(gs[2, :]), axes, ncol=4)
    save_show(fig, outfile)
    return fig, axes


if __name__ == "__main__":
    for dataset in datasets_from_argv(sys.argv[1:], default=list(DATASETS)):
        try:
            df, base = load_with_kernel_columns(dataset)
        except FileNotFoundError as e:
            print(f"Skipping {dataset}: {e}")
            continue
        out_dir = os.path.join(PLOTS_DIR, dataset)     # e.g. Plots/QM7/
        os.makedirs(out_dir, exist_ok=True)
        plot_results(df, base, dataset, outfile=os.path.join(out_dir, f"{dataset.lower()}_results.pdf"))