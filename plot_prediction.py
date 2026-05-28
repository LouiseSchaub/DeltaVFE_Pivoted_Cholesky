"""
Use this script to produce Figure 2, i.e. a side-by-side prediction error for Abalone and QM7.
"""
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt

from plot_utils import (METHODS, PLOTS_DIR, aggregate_prediction, add_full_baseline, format_panel, load_full_baseline,
                        load_method_data, load_trace, save_show, seaborn_line_ci95, seaborn_line_sd, shared_legend)


def plot_prediction_side_by_side(df_qm7, df_abalone, mae_full_qm7, rmse_full_abalone, outfile=None):
    """
    Plotting routine for Figure 1.

    :param df_qm7: DataFrame for QM7.
    :param df_abalone: DataFrame for Abalone.
    :param mae_full_qm7: Reference MAE baseline.
    :param rmse_full_abalone: Reference RMSE baseline.
    :param outfile: Filename
    :return: Figure with subplots.
    """
    x_min = int(min(df_qm7["Rank"].min(), df_abalone["Rank"].min()))
    x_max = int(max(df_qm7["Rank"].max(), df_abalone["Rank"].max()))

    fig = plt.figure(figsize=(10.8, 5.4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[5, 1.15], width_ratios=[1, 1], hspace=0.25, wspace=0.25)
    ax_abalone, ax_qm7, legend_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    for method in METHODS:
        if method in df_abalone["Method"].unique():
            seaborn_line_sd(ax_abalone, df_abalone[df_abalone["Method"] == method], "RMSE", method)
        if method in df_qm7["Method"].unique():
            seaborn_line_ci95(ax_qm7, df_qm7[df_qm7["Method"] == method], "MAE", method)

    add_full_baseline(ax_abalone, x_min, x_max, rmse_full_abalone)
    add_full_baseline(ax_qm7, x_min, x_max, mae_full_qm7)
    format_panel(ax_abalone, "RMSE")
    format_panel(ax_qm7, "MAE")
    shared_legend(legend_ax, [ax_abalone, ax_qm7], ncol=4)
    save_show(fig, outfile)
    return fig, (ax_abalone, ax_qm7, legend_ax)


if __name__ == "__main__":
    qm7 = aggregate_prediction(load_method_data("qm7", load_trace("qm7")), "qm7")
    abalone = aggregate_prediction(load_method_data("abalone", load_trace("abalone")), "abalone")
    mae_qm7, _ = load_full_baseline("qm7")
    rmse_abalone, _ = load_full_baseline("abalone")

    plot_prediction_side_by_side(qm7, abalone, mae_qm7, rmse_abalone,
                                 outfile=os.path.join(PLOTS_DIR, "abalone_qm7_prediction_error_side_by_side_test.pdf"))