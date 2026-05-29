"""
Use this script to produce Figure 1, i.e. the four panel view of relative VFE and approximation errors. This includes:
    - Relative VFE Gap (Abalone; top left)
    - Relative VFE Gap (QM7; top right)
    - Relative Trace-Norm Error (Abalone; bottom left)
    - Relative Trace-Norm Error (QM7; bottom right)
"""
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt

from plot_utils import (METHODS, PLOTS_DIR, aggregate_vfe, format_panel, load_full_baseline, load_method_data,
                        load_trace, matplotlib_line_errorbars, save_show, seaborn_line_ci95, seaborn_line_sd,
                        shared_legend)

VFE_LABEL = r"$(\mathcal{F}(K)-\mathcal{L}(\widetilde{K}))/|\mathcal{F}(K)|$"
TRACE_LABEL = r"$|\,\mathrm{tr}(K)-\mathrm{tr}(\widetilde K)\,|/\mathrm{tr}(K)$"


def plot_fourpanel(df_qm7, df_abalone, outfile=None):
    fig = plt.figure(figsize=(10.8, 8.4))
    gs = gridspec.GridSpec(3, 2, height_ratios=[5, 5, 1.2], width_ratios=[1, 1], hspace=0.28, wspace=0.25)
    axes = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    ax_ab_vfe, ax_ab_trace, ax_qm7_vfe, ax_qm7_trace = axes
    legend_ax = fig.add_subplot(gs[2, :])

    for method in METHODS:
        ab = df_abalone[df_abalone["Method"] == method]
        qm = df_qm7[df_qm7["Method"] == method]
        if not ab.empty:
            matplotlib_line_errorbars(ax_ab_vfe, ab, "RelativeVFE", method, error="sd")
            seaborn_line_sd(ax_ab_trace, ab, "TraceErrRel", method)
        if not qm.empty:
            matplotlib_line_errorbars(ax_qm7_vfe, qm, "RelativeVFE", method, error="ci95")
            seaborn_line_ci95(ax_qm7_trace, qm, "TraceErrRel", method)

    for ax, ylabel in zip(axes, [VFE_LABEL, TRACE_LABEL, VFE_LABEL, TRACE_LABEL]):
        format_panel(ax, ylabel)
    shared_legend(legend_ax, axes, ncol=4)
    save_show(fig, outfile)
    return fig, (*axes, legend_ax)


if __name__ == "__main__":
    _, elbo_qm7 = load_full_baseline("qm7")
    _, elbo_abalone = load_full_baseline("abalone")
    qm7 = aggregate_vfe(load_method_data("qm7", load_trace("qm7")), "qm7", elbo_qm7)
    abalone = aggregate_vfe(load_method_data("abalone", load_trace("abalone")), "abalone", elbo_abalone)

    plot_fourpanel(qm7, abalone,
                   outfile=os.path.join(PLOTS_DIR, "abalone_qm7_relative_vfe_relative_approx_fourpanel_log.pdf"))