"""
Use this script to produce Figure 3, i.e. a four panel view of the comparison of different s values on QM7.
This includes:
 - Relative VFE Gap (top left)
 - Prediction Error MAE (bottom left)
 - Relative Trace-Norm Error (top right)
 - Difference of the relative trace-norm errors of Delta-VFE using different s and RPCholesky (bottom right).
"""
import os
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from plot_utils import (PLOTS_DIR, QM7_DATA_DIR, STYLE, VFE_GAP_FLOOR, add_fold_from_source,
                        add_full_baseline, add_s_from_source, aggregate_qm7_s, format_panel, lineplot_ci95_custom,
                        load_csvs_any, load_full_folds_qm7, load_trace, save_show, shared_legend, to_numeric)

S_VALUES = [8, 32, 128, 512] # add s=1 here to see how the Delta-VFE results match RPCholesky's.
S_STYLE = {
    1: dict(linestyle="-", marker="o"), 8: dict(linestyle=":", marker="D"),
    32: dict(linestyle=(0, (5, 2)), marker="P"),
    128: dict(linestyle=(0, (4, 2, 1, 2)), marker="*"),
    512: dict(linestyle=(0, (2, 1, 1, 1)), marker="8")}

VFE_LABEL = r"$(\mathcal{F}(K)-\mathcal{L}(\widetilde{K}))/|\mathcal{F}(K)|$"
TRACE_LABEL = r"$|\operatorname{tr}(K)-\operatorname{tr}(\widetilde K_r)|/\operatorname{tr}(K)$"


def load_qm7_method(pattern, method, trace, full_folds, with_s=False):
    df = load_csvs_any(*pattern).assign(Method=method)
    if "Fold" not in df.columns:
        df = add_fold_from_source(df)
    if with_s:
        df = add_s_from_source(df)
    to_numeric(df, ["Rank", "Kernelwidth", "Noise2", "RMSE", "MAE", "ELBO", "TraceErr", "s"])
    df = df.dropna(subset=["Rank", "Fold", *( ["s"] if with_s else [] )]).copy()
    df["Rank"] = df["Rank"].astype(int)
    df["Fold"] = df["Fold"].astype("Int64")
    if with_s:
        df["s"] = df["s"].astype(int)
        df = df[df["s"].isin(S_VALUES)]
    df = df.merge(trace, on="Fold", how="left").merge(full_folds[["Fold", "ELBO"]].rename(columns={"ELBO": "ELBO_FULL"}), on="Fold", how="left")
    df["TraceErrRel"] = df["TraceErr"].abs() / df["TrK"]
    df["VFEGapRel"] = (df["ELBO"].sub(df["ELBO_FULL"]).abs() / df["ELBO_FULL"].abs()).clip(lower=VFE_GAP_FLOOR)
    return df.sort_values((["s"] if with_s else []) + ["Fold", "Rank"]).reset_index(drop=True)


def diff_to_rpcholesky(dvfe, rpch):
    base = rpch.rename(columns={"TraceErrRel": "TraceErrRel_RPCholesky"})[["Fold", "Rank", "TraceErrRel_RPCholesky"]]
    out = dvfe.merge(base, on=["Fold", "Rank"], how="inner")
    out["TraceErrRel_diff"] = out["TraceErrRel"] - out["TraceErrRel_RPCholesky"]
    return out.sort_values(["s", "Fold", "Rank"]).reset_index(drop=True)


def plot_s_comparison(dvfe, rpch, diff, mae_full, outfile=None):
    x_min = int(min(dvfe["Rank"].min(), rpch["Rank"].min()))
    x_max = int(max(dvfe["Rank"].max(), rpch["Rank"].max()))
    available_s = [s for s in S_VALUES if s in set(dvfe["s"].dropna().astype(int))]
    palette = sns.color_palette("viridis", n_colors=len(available_s))

    fig = plt.figure(figsize=(11.2, 9.0))
    gs = gridspec.GridSpec(3, 2, height_ratios=[5, 5, 1.2], width_ratios=[1, 1], hspace=0.35, wspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    ax_vfe, ax_trace, ax_mae, ax_theta = axes
    legend_ax = fig.add_subplot(gs[2, :])

    for color, s in zip(palette, available_s):
        label = fr"$\Delta$-VFE ($s={s}$)"
        style = S_STYLE[s]
        data = dvfe[dvfe["s"] == s]
        for ax, metric in [(ax_vfe, "VFEGapRel"), (ax_trace, "TraceErrRel"), (ax_mae, "MAE")]:
            lineplot_ci95_custom(ax, data, metric, label, color=color, **style)
        lineplot_ci95_custom(ax_theta, diff[diff["s"] == s], "TraceErrRel_diff", label, color=color, require_positive=False, **style)

    rp_style = STYLE["RPCholesky"]
    for ax, metric in [(ax_vfe, "VFEGapRel"), (ax_trace, "TraceErrRel"), (ax_mae, "MAE")]:
        lineplot_ci95_custom(ax, rpch, metric, "RPCholesky", color=rp_style["color"], linestyle=rp_style["linestyle"], marker=rp_style["marker"], linewidth=2.0)

    add_full_baseline(ax_mae, x_min, x_max, mae_full)
    for ax, ylabel, log_y in zip(axes, [VFE_LABEL, TRACE_LABEL, "MAE", r"$\theta$"], [True, True, True, False]):
        format_panel(ax, ylabel, log_y=log_y)
    shared_legend(legend_ax, axes, ncol=3)
    save_show(fig, outfile)
    return fig, (*axes, legend_ax)


if __name__ == "__main__":
    trace = load_trace("qm7")
    full_folds = load_full_folds_qm7()
    dvfe_raw = load_qm7_method((f"{QM7_DATA_DIR}/DeltaVFE/deltavfe_fold=*_kernelwidth=*_noise2=*_m=2048_s=*.csv",
                                f"{QM7_DATA_DIR}/DeltaVFE/deltavfe_*fold=*_m=2048*_s=*.csv"), "DeltaVFE", trace,
                               full_folds, with_s=True)
    rpch_raw = load_qm7_method((f"{QM7_DATA_DIR}/RPCholesky/rpcholesky_fold=*_kernelwidth=*_noise2=*_m=2048.csv",
                                f"{QM7_DATA_DIR}/RPCholesky/rpcholesky_*fold=*_m=2048*.csv"), "RPCholesky",
                               trace, full_folds)
    dvfe = aggregate_qm7_s(dvfe_raw, has_s=True)
    rpch = aggregate_qm7_s(rpch_raw, has_s=False)

    plot_s_comparison(dvfe, rpch, diff_to_rpcholesky(dvfe, rpch), float(full_folds["MAE"].mean()),
                      outfile=os.path.join(PLOTS_DIR, "qm7_fourpanel_shared_legend_relative_vfe.pdf"))