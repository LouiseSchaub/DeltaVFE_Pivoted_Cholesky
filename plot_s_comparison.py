"""
Use this script to produce Figure 3, i.e. the comparison of different s values on QM7.
This includes:
 - VFE gap per training point, (F(K) - L(K~)) / n in nats (top left)
 - Prediction Error MAE in kcal/mol (bottom left)
 - Relative Trace-Norm Error (top right)
 - VFE gap per training point against the number of kernel columns computed during the pivot selection
   (bottom right; column KernelColumns: r for RPCholesky, |S_1 u ... u S_r| <= n for Delta-VFE).

Data: the result files of core.run_full_experiment for QM7 (80/20 split, no folds), loaded with
plot_utils.load_results: Data/QM7/RPCholesky/rpcholesky_kernelwidth=..._m=2048.csv and
Data/QM7/DeltaVFE/deltavfe_kernelwidth=..._m=2048_s=<s>.csv, and the full-GP baseline in Data/QM7/Full/.
Bars are 95% confidence intervals over the runs (RunID). Delta-VFE result files written before the kernel-column
count was recorded get it via add_kernel_columns (ADD_KERNEL_COLUMNS = True; repeats only the pivot selection).
"""
import os

from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from plot_utils import (PLOTS_DIR, STYLE, add_full_baseline, baseline_value, consistent_log_ticks, format_panel,
                        lineplot_ci95_custom, load_results, save_show, shared_legend, viridis)

DATASET = "QM7"
S_VALUES = [8, 32, 128, 512] # add s=1 here to see how the Delta-VFE results match RPCholesky's.
S_STYLE = {
    1: dict(linestyle="-", marker="o"), 8: dict(linestyle=":", marker="D"),
    32: dict(linestyle=(0, (5, 2)), marker="P"),
    128: dict(linestyle=(0, (4, 2, 1, 2)), marker="*"),
    512: dict(linestyle=(0, (2, 1, 1, 1)), marker="8")}

VFE_METRIC = "VFEGapPerPoint"    # (F(K) - L(K~)) / n, as in the dataset figures; "VFEGapRel" for the relative gap
VFE_LABEL = r"$(\mathcal{F}(K)-\mathcal{L}(\widetilde{K}))/n$"
TRACE_LABEL = r"$|\operatorname{tr}(K)-\operatorname{tr}(\widetilde K_r)|/\operatorname{tr}(K)$"
MAE_METRIC = "MAE_orig"          # MAE in kcal/mol (the column "MAE" is on the standardized scale)
MAE_LABEL = "MAE"
COST_LABEL = "Kernel columns computed"
ADD_KERNEL_COLUMNS = True        # add missing kernel-column counts to the Delta-VFE result files before plotting



def load_qm7():
    """
    RPCholesky and Delta-VFE (s in S_VALUES) results of QM7 and the full-GP MAE.
    :return: dvfe, rpch (one row per run and rank) and the full-GP MAE in kcal/mol.
    """
    df, base = load_results(DATASET, methods=["RPCholesky", "DeltaVFE"], s="all")
    missing = sorted(set(df.loc[(df["Method"] == "DeltaVFE") & df["s"].isin(S_VALUES) & df["KernelColumns"].isna(),
                                "s"].astype(int)))
    if ADD_KERNEL_COLUMNS and missing:
        from add_kernel_columns import add_kernel_columns
        print(f"{DATASET}: adding kernel-column counts for s={missing} ...")
        add_kernel_columns(DATASET, s=missing)
        df, base = load_results(DATASET, methods=["RPCholesky", "DeltaVFE"], s="all")
    dvfe = df[(df["Method"] == "DeltaVFE") & df["s"].isin(S_VALUES)].copy()
    dvfe["s"] = dvfe["s"].astype(int)
    rpch = df[df["Method"] == "RPCholesky"].copy()
    return dvfe, rpch, baseline_value(base, MAE_METRIC)


def plot_vs_kernel_columns(ax, data, metric, label, color, linestyle="-", marker="o", linewidth=1.6):
    """
    metric against the number of kernel columns computed: one point per rank at (mean KernelColumns, mean metric) over
    the runs, with the 95% CI of the metric as error bars (as lineplot_ci95_custom does against the rank).
    """
    data = data.dropna(subset=["KernelColumns", metric])
    data = data[data[metric] > 0]
    if data.empty:
        return
    stats = data.groupby("Rank").agg(x=("KernelColumns", "mean"), y=(metric, "mean"), sd=(metric, "std"),
                                     n=(metric, "count")).sort_index()
    ci = (1.96 * stats["sd"].fillna(0.0) / np.sqrt(stats["n"].clip(lower=1))).to_numpy()
    y = stats["y"].to_numpy()
    ax.errorbar(stats["x"], y, yerr=[np.minimum(ci, y * (1.0 - 1e-12)), ci], label=label, color=color,
                linestyle=linestyle, marker=marker, markersize=4.8, linewidth=linewidth, markeredgewidth=0.0,
                capsize=2, elinewidth=1.0, capthick=1.0)


def format_cost_panel(ax, ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    consistent_log_ticks(ax.xaxis)
    consistent_log_ticks(ax.yaxis)
    ax.set_xlabel(COST_LABEL)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", linestyle="-", color="0.85")
    ax.grid(True, which="minor", linestyle=":", color="0.9")


def plot_s_comparison(dvfe, rpch, mae_full, outfile=None):
    x_min = int(min(dvfe["Rank"].min(), rpch["Rank"].min()))
    x_max = int(max(dvfe["Rank"].max(), rpch["Rank"].max()))
    available_s = [s for s in S_VALUES if s in set(dvfe["s"].dropna().astype(int))]
    palette = sns.color_palette("viridis", n_colors=len(available_s))

    fig = plt.figure(figsize=(11.2, 9.0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[5, 5, 1.2], width_ratios=[1, 1], hspace=0.35, wspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    ax_vfe, ax_trace, ax_mae, ax_cost = axes
    legend_ax = fig.add_subplot(gs[2, :])

    # RPCholesky first, so that the Delta-VFE curves (brighter hues) are drawn on top of it
    rp_style = STYLE["RPCholesky"]
    for ax, metric in [(ax_vfe, VFE_METRIC), (ax_trace, "TraceErrRel"), (ax_mae, MAE_METRIC)]:
        lineplot_ci95_custom(ax, rpch, metric, "RPCholesky", color=viridis(0.00), linestyle=rp_style["linestyle"], marker=rp_style["marker"], linewidth=2.0)
    plot_vs_kernel_columns(ax_cost, rpch, VFE_METRIC, "RPCholesky", color=viridis(0.00),
                           linestyle=rp_style["linestyle"], marker=rp_style["marker"], linewidth=2.0)

    for color, s in zip(palette, available_s):
        label = fr"$\Delta$-VFE ($s={s}$)"
        style = S_STYLE[s]
        data = dvfe[dvfe["s"] == s]
        for ax, metric in [(ax_vfe, VFE_METRIC), (ax_trace, "TraceErrRel"), (ax_mae, MAE_METRIC)]:
            lineplot_ci95_custom(ax, data, metric, label, color=color, **style)
        plot_vs_kernel_columns(ax_cost, data, VFE_METRIC, label, color=color, **style)

    #add_full_baseline(ax_mae, x_min, x_max, mae_full)
    for ax, ylabel in zip(axes[:3], [VFE_LABEL, TRACE_LABEL, MAE_LABEL]):
        format_panel(ax, ylabel, log_y=True)
    format_cost_panel(ax_cost, VFE_LABEL)
    shared_legend(legend_ax, axes, ncol=5)
    save_show(fig, outfile)
    return fig, (*axes, legend_ax)

def plot_s_comparison_separate(
    dvfe,
    rpch,
    mae_full,
    outfile_prefix="s_comparison",
):
    x_min = int(min(dvfe["Rank"].min(), rpch["Rank"].min()))
    x_max = int(max(dvfe["Rank"].max(), rpch["Rank"].max()))

    available_s = [
        s for s in S_VALUES
        if s in set(dvfe["s"].dropna().astype(int))
    ]
    palette = sns.color_palette("viridis", n_colors=len(available_s))

    def _single(metric, ylabel, outfile, add_baseline=False):
        fig = plt.figure(figsize=(12.8, 7.2))#(6.2, 4.6))
        ax = fig.add_subplot(111)
        ax.tick_params(axis="both")

        # RPCholesky first, so that the Delta-VFE curves are drawn on top of it
        rp_style = STYLE["RPCholesky"]
        lineplot_ci95_custom(
            ax,
            rpch,
            metric,
            "RPCholesky",
            color=viridis(0.00),
            linestyle=rp_style["linestyle"],
            marker=rp_style["marker"],
            linewidth=2.0,
        )

        for color, s in zip(palette, available_s):
            label = fr"$\Delta$-VFE ($s={s}$)"
            style = S_STYLE[s]
            data = dvfe[dvfe["s"] == s]

            lineplot_ci95_custom(
                ax,
                data,
                metric,
                label,
                color=color,
                **style,
            )

        if add_baseline:
            add_full_baseline(ax, x_min, x_max, mae_full)

        format_panel(ax, ylabel, log_y=True)
        ax.legend(frameon=True, ncol=2)

        save_show(fig, outfile)
        return fig, ax

    fig_vfe, ax_vfe = _single(
        VFE_METRIC,
        VFE_LABEL,
        f"{outfile_prefix}_vfe_gap_per_point.pdf",
    )

    fig_trace, ax_trace = _single(
        "TraceErrRel",
        TRACE_LABEL,
        f"{outfile_prefix}_relative_trace_error.pdf",
    )

    fig_mae, ax_mae = _single(
        MAE_METRIC,
        MAE_LABEL,
        f"{outfile_prefix}_mae.pdf",
        add_baseline=True,
    )

    # VFE gap per point against the kernel columns computed
    fig_cost = plt.figure(figsize=(12.8, 7.2))
    ax_cost = fig_cost.add_subplot(111)
    rp_style = STYLE["RPCholesky"]
    plot_vs_kernel_columns(ax_cost, rpch, VFE_METRIC, "RPCholesky", color=viridis(0.00),
                           linestyle=rp_style["linestyle"], marker=rp_style["marker"], linewidth=2.0)
    for color, s in zip(palette, available_s):
        plot_vs_kernel_columns(ax_cost, dvfe[dvfe["s"] == s], VFE_METRIC, fr"$\Delta$-VFE ($s={s}$)", color=color,
                               **S_STYLE[s])
    format_cost_panel(ax_cost, VFE_LABEL)
    ax_cost.legend(frameon=True, ncol=2)
    save_show(fig_cost, f"{outfile_prefix}_vfe_gap_vs_kernel_columns.pdf")

    return {
        "vfe": (fig_vfe, ax_vfe),
        "trace": (fig_trace, ax_trace),
        "mae": (fig_mae, ax_mae),
        "cost": (fig_cost, ax_cost),
    }


if __name__ == "__main__":
    dvfe, rpch, mae_full = load_qm7()

    plot_s_comparison(dvfe, rpch, mae_full,
                      outfile=os.path.join(PLOTS_DIR, "qm7_fourpanel_shared_legend_vfe_per_point.pdf"))

    plot_s_comparison_separate(
        dvfe,
        rpch,
        mae_full,
        outfile_prefix=os.path.join(PLOTS_DIR, "qm7_s_comparison"),
    )