"""
Shared plotting theme, data loading, aggregation, and figure helpers.
"""
from __future__ import annotations

import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, LogLocator

# Global configuration
PLOTS_DIR = "Plots"
VFE_GAP_FLOOR = 1e-12
METHODS = ["Greedy", "RPCholesky", "DeltaVFE"]

# Per dataset: result folder, prediction-error measure(s) (first available column is used), default batch factor s for
# the method comparison, and the error band (sd or 95% confidence interval over the runs).
DATASETS = {
    "Abalone": dict(dir="Data/Abalone", error=("RMSE",), s=8, band="sd"),
    "QM7":     dict(dir="Data/QM7", error=("MAE_orig", "MAE"), s=16, band="ci95", unit="kcal/mol"),
    "KIN40K":  dict(dir="Data/KIN40K", error=("SMSE",), s=32, band="ci95"),
    "QM9":     dict(dir="Data/QM9", error=("MAE_orig", "MAE"), s=32, band="ci95", unit="eV"),
}
QM7_DATA_DIR = DATASETS["QM7"]["dir"]
ABALONE_DATA_DIR = DATASETS["Abalone"]["dir"]

VFE_REL_LABEL = r"$(\mathcal{F}(K)-\mathcal{L}(\widetilde{K}))/|\mathcal{F}(K)|$"
VFE_PP_LABEL = r"$(\mathcal{F}(K)-\mathcal{L}(\widetilde{K}))/n$"
TRACE_LABEL = r"$|\,\mathrm{tr}(K)-\mathrm{tr}(\widetilde K)\,|/\mathrm{tr}(K)$"
METRIC_LABELS = {"VFEGapRel": VFE_REL_LABEL, "VFEGapPerPoint": VFE_PP_LABEL,
                 "VFEGap": r"$\mathcal{F}(K)-\mathcal{L}(\widetilde{K})$", "TraceErrRel": TRACE_LABEL,
                 "RMSE": "RMSE", "MAE": "MAE", "SMSE": "SMSE", "MSLL": "MSLL", "NLPD": "NLPD",
                 "RMSE_orig": "RMSE", "MAE_orig": "MAE"}
LINEAR_METRICS = {"MSLL", "NLPD"}          # may be negative: linear y-axis
# column of the full-GP baseline file that corresponds to a metric of the result files
BASELINE_COLUMN = {"RMSE": "RMSE_std", "MAE": "MAE_std", "RMSE_orig": "RMSE_orig", "MAE_orig": "MAE_orig", "SMSE": "SMSE"}

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "axes.unicode_minus": False})
sns.set_context(context="paper", font_scale=1.35, rc={"lines.linewidth": 1.6, "lines.markersize": 4.8,
                                                      "legend.fontsize": 10, "axes.labelsize": 11,
                                                      "axes.titlesize": 11, "xtick.labelsize": 10,
                                                      "ytick.labelsize": 10})

# All figures take their colours from viridis (colour-blind safe, readable in greyscale). Positions on the colour map
# from 0 (dark purple) to 1 (yellow); the palest end (> 0.9) is avoided because it fades on white.
def viridis(t: float) -> str:
    return mpl.colors.to_hex(mpl.colormaps["viridis"](t))


COLORS = {
    "Greedy":     viridis(0.00),   # dark purple
    "RPCholesky": viridis(0.33),   # blue
    "DeltaVFE":   viridis(0.67),   # green
}
STYLE = {
    "Greedy":     dict(color=COLORS["Greedy"],     linestyle="--",              marker="o"),
    "RPCholesky": dict(color=COLORS["RPCholesky"], linestyle=(8, (5, 7)),       marker="s"),
    "DeltaVFE":   dict(color=COLORS["DeltaVFE"],   linestyle=(2, (2, 1, 2, 7)), marker="^"),
}
# Values of s of Delta-VFE: the green-to-yellow part of viridis around COLORS["DeltaVFE"], away from RPCholesky's blue.
S_RANGE = (0.48, 0.90)


def s_colors(s_values) -> dict:
    """Colour of each s (sorted ascending), spread evenly over S_RANGE of viridis."""
    s_values = sorted(s_values)
    pos = np.linspace(*S_RANGE, len(s_values)) if len(s_values) > 1 else [0.67]
    return {s: viridis(t) for s, t in zip(s_values, pos)}
METHOD_LABELS = {"Greedy": "Greedy", "RPCholesky": "RPCholesky", "DeltaVFE": r"$\Delta$-VFE"}

# Figure sizes and grid layouts of the original plotting scripts, used by every figure:
#   one row of panels plus a legend row (as Figure 2, prediction error side by side)
FIGSIZE_ROW = (10.8, 5.4)
GRID_ROW = dict(height_ratios=[5, 1.15], hspace=0.25, wspace=0.25)
#   two rows of panels plus a legend row (as Figure 1, four panels)
FIGSIZE_TWO_ROWS = (10.8, 8.4)
GRID_TWO_ROWS = dict(height_ratios=[5, 5, 1.2], hspace=0.28, wspace=0.25)
#   two rows of panels plus a legend row (as Figure 3, comparison of s)
FIGSIZE_S_COMPARISON = (11.2, 9.0)
GRID_S_COMPARISON = dict(height_ratios=[5, 5, 1.2], hspace=0.35, wspace=0.28)
ERRORBAR_KWS = dict(err_style="bars", err_kws={"capsize": 2})


def dataset_info(dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    return DATASETS[dataset]


def metric_label(metric: str, dataset: str | None = None) -> str:
    label = METRIC_LABELS.get(metric, metric)
    unit = DATASETS.get(dataset, {}).get("unit")
    if metric in ("RMSE_orig", "MAE_orig") and unit:
        label += f" [{unit}]"
    return label


# File loading
def load_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return pd.concat([pd.read_csv(f).assign(__source__=os.path.basename(f)) for f in files], ignore_index=True)


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_s_from_source(df: pd.DataFrame) -> pd.DataFrame:
    df["s"] = pd.to_numeric(df["__source__"].astype(str).str.extract(r"_s=(\d+)")[0], errors="coerce")
    return df


def _m_of(path: str) -> int:
    return int(re.search(r"_m=(\d+)", os.path.basename(path)).group(1))


def method_files(dataset: str, method: str, s="*", m: int | None = None) -> list[str]:
    """
    Result files of one method written by core.run_full_experiment:
        {dir}/{Method}/{method}_kernelwidth=<ell>_noise2=<sigma2>_m=<m>[_s=<s>].csv
    Files with a fold label (results of the former five-fold QM7 setup) and the combined deltavfe_ALLS files do not match.
    If m is None, the largest available m is used.
    """
    root = dataset_info(dataset)["dir"]
    sub, base = {"Greedy": ("Greedy", "greedy"), "RPCholesky": ("RPCholesky", "rpcholesky"),
                 "DeltaVFE": ("DeltaVFE", "deltavfe")}[method]
    s_part = f"_s={s}" if method == "DeltaVFE" else ""
    files = sorted(glob.glob(f"{root}/{sub}/{base}_kernelwidth=*_noise2=*_m={'*' if m is None else m}{s_part}.csv"))
    if files and m is None:
        m_max = max(_m_of(f) for f in files)
        files = [f for f in files if _m_of(f) == m_max]
    return files


def load_baseline(dataset: str) -> pd.DataFrame:
    """All full-GP baseline files of a dataset (one row per hyperparameter setting)."""
    files = sorted(glob.glob(f"{dataset_info(dataset)['dir']}/Full/baseline_kernelwidth=*_noise2=*.csv"))
    if not files:
        raise FileNotFoundError(f"No full-GP baseline found in {dataset_info(dataset)['dir']}/Full "
                                f"(run the experiment with skip_baseline=False).")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return to_numeric(df, list(df.columns))


def choose_one_hyper_setting(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Kernelwidth", "Noise2"}.issubset(df.columns):
        return df.copy()
    kw, nz = df.groupby(["Kernelwidth", "Noise2"]).size().sort_values(ascending=False).index[0]
    return df[(df["Kernelwidth"] == kw) & (df["Noise2"] == nz)].copy()


def baseline_for(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.Series:
    """Baseline row with the same hyperparameters as the results in df."""
    kw, nz = float(df["Kernelwidth"].iloc[0]), float(df["Noise2"].iloc[0])
    hit = baseline[np.isclose(baseline["Kernelwidth"], kw, rtol=1e-12) & np.isclose(baseline["Noise2"], nz, rtol=1e-12)]
    if hit.empty:
        raise FileNotFoundError(f"No full-GP baseline for kernelwidth={kw}, noise2={nz}.")
    return hit.iloc[0]


def add_derived_columns(df: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    """
    Make sure the derived columns exist. Result files written by the current core.py already contain them; for older
    files they are computed from ELBO, TraceErr and the full-GP baseline (log marginal likelihood, trace, n).
    """
    df = df.copy()
    logml = float(base["Logml"])
    if "TraceErrRel" not in df.columns:
        df["TraceErrRel"] = df["TraceErr"].abs() / float(base["TrK"])
    if "VFEGap" not in df.columns:
        df["VFEGap"] = logml - df["ELBO"]
    if "VFEGapPerPoint" not in df.columns:
        df["VFEGapPerPoint"] = df["VFEGap"] / float(base["n_train"])
    if "VFEGapRel" not in df.columns:
        df["VFEGapRel"] = (df["ELBO"] - logml).abs() / abs(logml)
    df["VFEGapRel"] = df["VFEGapRel"].clip(lower=VFE_GAP_FLOOR)
    return df


def load_results(dataset: str, methods=METHODS, s=None, m: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the result files of a dataset, one hyperparameter setting, with derived columns.

    :param dataset: "Abalone", "QM7", "KIN40K" or "QM9".
    :param methods: methods to load.
    :param s: batch factor(s) of Delta-VFE: an int, a list of ints, or "all". Default: the dataset's default s.
    :param m: maximum rank of the runs (default: the largest available).
    :return: (results with columns Method, s, RunID, Rank, metrics ..., full-GP baseline row)
    """
    s = dataset_info(dataset)["s"] if s is None else s
    s_list = None if s == "all" else ([s] if np.isscalar(s) else list(s))
    parts = []
    for method in methods:
        if method == "DeltaVFE":
            files = [f for v in (s_list or ["*"]) for f in method_files(dataset, method, s=v, m=m)]
            available = sorted({int(re.search(r"_s=(\d+)", os.path.basename(f)).group(1))
                                for f in method_files(dataset, method, s="*", m=m)})
            if not files and s_list and len(s_list) == 1 and available:
                # requested s not (yet) computed: use the available s closest to it (on a log scale)
                s_use = min(available, key=lambda v: (abs(np.log2(v) - np.log2(s_list[0])), -v))
                print(f"NOTE: no Delta-VFE results with s={s_list[0]} for {dataset}; using s={s_use} "
                      f"(available: {available})")
                files = method_files(dataset, method, s=s_use, m=m)
        else:
            files = method_files(dataset, method, m=m)
        if not files:
            root = dataset_info(dataset)["dir"]
            sub = {"Greedy": "Greedy", "RPCholesky": "RPCholesky", "DeltaVFE": "DeltaVFE"}[method]
            present = sorted(os.path.basename(f) for f in glob.glob(f"{root}/{sub}/*.csv"))
            print(f"WARNING: no {method} results for {dataset} matching "
                  f"'{method_files.__name__}' (files in {root}/{sub}: {present[:6]}{' ...' if len(present) > 6 else ''})")
            continue
        df = pd.concat([pd.read_csv(f).assign(__source__=os.path.basename(f)) for f in files], ignore_index=True)
        df = add_s_from_source(df).assign(Method=method)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No result files for {dataset} in {dataset_info(dataset)['dir']}.")
    df = pd.concat(parts, ignore_index=True)
    df = to_numeric(df, [c for c in df.columns if c not in ("Method", "__source__")])
    df = choose_one_hyper_setting(df.dropna(subset=["Rank"]))
    df["Rank"] = df["Rank"].astype(int)
    base = baseline_for(df, load_baseline(dataset))
    df = add_derived_columns(df, base)
    # distinct kernel columns evaluated during the selection up to rank r (written by core.run_method): r for Greedy
    # and RPCholesky; for Delta-VFE every column among the candidates drawn so far, counted once (at most n).
    # Result files written before this column existed: r for Greedy/RPCholesky, missing (NaN) for Delta-VFE.
    if "KernelColumns" not in df.columns:
        df["KernelColumns"] = np.nan
    one_per_pivot = df["Method"] != "DeltaVFE"
    kc = pd.to_numeric(df["KernelColumns"], errors="coerce").to_numpy(dtype=float)
    df["KernelColumns"] = np.where(one_per_pivot.to_numpy() & np.isnan(kc), df["Rank"].to_numpy(dtype=float), kc)
    if df.loc[~one_per_pivot, "KernelColumns"].isna().any():
        print(f"NOTE: Delta-VFE results of {dataset} have no KernelColumns column (written before it was added); "
              f"they are left out of the kernel-columns panel. Add it with add_kernel_columns.add_kernel_columns"
              f"('{dataset}') or ADD_KERNEL_COLUMNS = True in plot_results.py")
    return df.sort_values(["Method", "s", "RunID", "Rank"]).reset_index(drop=True), base


def error_metric(df: pd.DataFrame, dataset: str) -> str:
    """The prediction-error column used for a dataset (first available of DATASETS[dataset]['error'])."""
    for col in dataset_info(dataset)["error"]:
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError(f"None of {dataset_info(dataset)['error']} in the results of {dataset}.")


def baseline_value(base: pd.Series, metric: str) -> float:
    col = BASELINE_COLUMN.get(metric)
    return float(base[col]) if col in base.index and pd.notna(base[col]) else np.nan


# Plot helpers
def set_log2_x(ax):
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def consistent_log_ticks(axis):
    """
    Major ticks at every power of 10 that matplotlib labels and minor ticks at 1..9 x 10^k in every decade, also when
    the axis spans many decades (matplotlib's default then thins out the major ticks and drops the minor ticks, so the
    minor grid disappears, e.g. for Abalone).
    """
    axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0), numticks=999))
    # minor labels: matplotlib's default (only on axes spanning about one decade or less, e.g. the RMSE panels)


def format_panel(ax, ylabel: str, log_y: bool = True):
    ax.set_xlabel(r"Rank $r$")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        consistent_log_ticks(ax.yaxis)
    ax.grid(True, which="major", linestyle="-", color="0.85")
    ax.grid(True, which="minor", linestyle=":", color="0.9")
    set_log2_x(ax)


def add_full_baseline(ax, x_min: int, x_max: int, y_value: float, label="Full"):
    ax.plot([x_min, x_max], [y_value, y_value], color="black", linestyle=":", linewidth=2.2, alpha=1.0, zorder=0, label=label)


def positive_finite(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Keep only rows that are safe to draw on a logarithmic y-axis."""
    vals = pd.to_numeric(data[metric], errors="coerce")
    return data[np.isfinite(vals) & (vals > 0)].copy()


def collect_unique_legend_handles(*axes):
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in seen:
                handles.append(h); labels.append(label); seen.add(label)
    return handles, labels


def shared_legend(legend_ax, axes, ncol: int):
    handles, labels = collect_unique_legend_handles(*axes)
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", ncol=ncol, frameon=True, borderaxespad=0.0, handlelength=2.2, handletextpad=0.6, columnspacing=1.2)


def save_show(fig, outfile: str | None):
    if outfile:
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        print("Saved:", outfile)
    plt.show()


def seaborn_line(ax, data, metric, method, error="ci95", markersize=4.8, linewidth=1.6, require_positive=True):
    if require_positive:
        data = positive_finite(data, metric)
    else:
        vals = pd.to_numeric(data[metric], errors="coerce")
        data = data[np.isfinite(vals)].copy()
    if data.empty:
        return
    st = STYLE[method]
    kwargs = dict(
        data=data, x="Rank", y=metric, label=METHOD_LABELS.get(method, method),
        color=st["color"], linestyle=st["linestyle"], marker=st["marker"], markers=True,
        markersize=markersize, linewidth=linewidth, markeredgewidth=0.0, ax=ax, **ERRORBAR_KWS,
    )
    modern = {"sd": "sd", "ci95": ("ci", 95)}[error]
    legacy = {"sd": "sd", "ci95": 95}[error]
    try:
        sns.lineplot(**kwargs, estimator="mean" if error == "sd" else "mean", errorbar=modern)
    except TypeError:
        sns.lineplot(**kwargs, ci=legacy)


def seaborn_line_ci95(ax, data, metric, method, **kw):
    seaborn_line(ax, data, metric, method, error="ci95", **kw)


def seaborn_line_sd(ax, data, metric, method, **kw):
    seaborn_line(ax, data, metric, method, error="sd", **kw)


def lineplot_ci95_custom(ax, data, metric, label, color, linestyle="-", marker="o", markersize=4.8, linewidth=1.6, require_positive=True):
    if require_positive:
        data = positive_finite(data, metric)
    else:
        vals = pd.to_numeric(data[metric], errors="coerce")
        data = data[np.isfinite(vals)].copy()
    if data.empty:
        return
    kwargs = dict(data=data, x="Rank", y=metric, label=label, color=color, linestyle=linestyle, marker=marker,
                  markers=True, markersize=markersize, linewidth=linewidth, markeredgewidth=0.0, ax=ax, **ERRORBAR_KWS)
    try:
        sns.lineplot(**kwargs, errorbar=("ci", 95))
    except TypeError:
        sns.lineplot(**kwargs, ci=95)


def matplotlib_line_errorbars(ax, data, metric, method, error="sd", markersize=4.8, linewidth=1.6, require_positive=True):
    if require_positive:
        data = positive_finite(data, metric)
    else:
        vals = pd.to_numeric(data[metric], errors="coerce")
        data = data[np.isfinite(vals)].copy()
    if data.empty:
        return
    st = STYLE[method]
    stats = data.groupby("Rank", as_index=False)[metric].agg(mean="mean", std="std", n="count").sort_values("Rank")
    y = stats["mean"].to_numpy(dtype=float)
    err = stats["std"].fillna(0.0)
    yerr = (1.96 * err / np.sqrt(stats["n"].clip(lower=1)) if error == "ci95" else err).to_numpy(dtype=float)
    ax.errorbar(stats["Rank"], y, yerr=[np.minimum(yerr, np.maximum(y * (1.0 - 1e-12), 0.0)), yerr],
                label=METHOD_LABELS.get(method, method), color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=markersize, linewidth=linewidth, markeredgewidth=0.0, capsize=3,
                elinewidth=1.0, capthick=1.0, zorder=2)

def plot_metric(ax, df: pd.DataFrame, metric: str, dataset: str, methods=METHODS, baseline: float | None = None):
    """
    One panel: metric against rank for each method, with the dataset's error band (sd or 95% CI over the runs).
    VFE gaps are drawn with error bars, the other metrics with seaborn. Metrics that can be negative (MSLL, NLPD) get
    a linear y-axis. If baseline is a finite number, the full-GP value is drawn as a horizontal line.
    """
    band = dataset_info(dataset)["band"]
    log_y = metric not in LINEAR_METRICS
    for method in methods:
        d = df[df["Method"] == method]
        if d.empty or metric not in d.columns or d[metric].isna().all():
            continue
        if metric.startswith("VFEGap"):
            matplotlib_line_errorbars(ax, d, metric, method, error=band, require_positive=log_y)
        else:
            seaborn_line(ax, d, metric, method, error=band, require_positive=log_y)
    if baseline is not None and np.isfinite(baseline):
        add_full_baseline(ax, int(df["Rank"].min()), int(df["Rank"].max()), baseline)
    format_panel(ax, metric_label(metric, dataset), log_y=log_y)


def datasets_from_argv(argv, default):
    """Dataset names from the command line (case-insensitive), e.g. `python plot_results.py KIN40K QM9`."""
    names = {k.lower(): k for k in DATASETS}
    chosen = [names[a.lower()] for a in argv] if argv else list(default)
    unknown = [a for a in argv if a.lower() not in names]
    if unknown:
        raise SystemExit(f"Unknown dataset(s) {unknown}; choose from {list(DATASETS)}")
    return chosen


def plot_metric_vs_cost(ax, df: pd.DataFrame, metric: str, dataset: str, methods=METHODS):
    """
    One panel: metric against the number of kernel columns computed during the selection (KernelColumns; each
    computed column is kept, so a candidate drawn again is not recomputed),
    mean over the runs with the dataset's error band (sd or 95% CI), log-log axes.
    """
    band = dataset_info(dataset)["band"]
    for method in methods:
        d = df[df["Method"] == method]
        if d.empty or metric not in d.columns or d[metric].isna().all():
            continue
        d = positive_finite(d.dropna(subset=["KernelColumns"]), metric)
        if d.empty:
            continue
        st = STYLE[method]
        stats = d.groupby("Rank").agg(x=("KernelColumns", "mean"), mean=(metric, "mean"), std=(metric, "std"),
                                      n=(metric, "count")).sort_values("x")
        y = stats["mean"].to_numpy(dtype=float)
        err = stats["std"].fillna(0.0)
        yerr = (1.96 * err / np.sqrt(stats["n"].clip(lower=1)) if band == "ci95" else err).to_numpy(dtype=float)
        ax.errorbar(stats["x"], y, yerr=[np.minimum(yerr, np.maximum(y * (1.0 - 1e-12), 0.0)), yerr],
                    label=METHOD_LABELS.get(method, method), color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=4.8, linewidth=1.6, markeredgewidth=0.0, capsize=3,
                    elinewidth=1.0, capthick=1.0, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    consistent_log_ticks(ax.xaxis)
    consistent_log_ticks(ax.yaxis)
    ax.set_xlabel("Kernel columns computed")
    ax.set_ylabel(metric_label(metric, dataset))
    ax.grid(True, which="major", linestyle="-", color="0.85")
    ax.grid(True, which="minor", linestyle=":", color="0.9")