"""Shared plotting theme, data loading, aggregation, and figure helpers."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Global configuration
QM7_DATA_DIR = "Data/QM7"
ABALONE_DATA_DIR = "Data/Abalone"
PLOTS_DIR = "Plots"
VFE_GAP_FLOOR = 1e-12
METHODS = ["Greedy", "RPCholesky", "DeltaVFE"]

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "axes.unicode_minus": False})
sns.set_context(context="paper", font_scale=1.35, rc={"lines.linewidth": 1.6, "lines.markersize": 4.8,
                                                      "legend.fontsize": 10, "axes.labelsize": 11,
                                                      "axes.titlesize": 11, "xtick.labelsize": 10,
                                                      "ytick.labelsize": 10})

STYLE = {
    "Greedy":     dict(color="deepskyblue", linestyle="--",              marker="o"),
    "RPCholesky": dict(color="chocolate",   linestyle=(8, (5, 7)),       marker="s"),
    "DeltaVFE":   dict(color="forestgreen", linestyle=(2, (2, 1, 2, 7)), marker="^"),
}
METHOD_LABELS = {"Greedy": "Greedy", "RPCholesky": "RPCholesky", "DeltaVFE": r"$\Delta$-VFE"}
ERRORBAR_KWS = dict(err_style="bars", err_kws={"capsize": 2})

# File loading
def load_csvs(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return pd.concat([pd.read_csv(f).assign(__source__=os.path.basename(f)) for f in files], ignore_index=True)


def load_csvs_any(*patterns: str) -> pd.DataFrame:
    """
    Load all CSVs matching the first non-empty pattern. Supports different file naming conventions.
    """
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return pd.concat([pd.read_csv(f).assign(__source__=os.path.basename(f)) for f in files], ignore_index=True)
    raise FileNotFoundError(f"No files matched any of: {patterns}")


def first_existing(*patterns: str) -> list[str]:
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return files
    raise FileNotFoundError(f"No files matched any of: {patterns}")


def full_baseline_files(dataset: str) -> list[str]:
    """
    Return full-baseline files, using the filename convention to separate datasets.

    New baseline files contain all full quantities in one CSV.  Files with ``fold=`` in the name are QM7 fold baselines,
    while files without ``fold=`` are Abalone baselines.
    """
    if dataset == "qm7":
        return first_existing(
            f"{QM7_DATA_DIR}/Full/baseline_fold=*_kernelwidth=*_noise2=*.csv",
            f"{QM7_DATA_DIR}/Full/full_laplacian_fold=*_hypers*.dat",
        )

    if dataset == "abalone":
        files = sorted(glob.glob(f"{ABALONE_DATA_DIR}/Full/baseline_kernelwidth=*_noise2=*.csv"))
        files = [f for f in files if "fold=" not in os.path.basename(f)]
        if files:
            return files
        return first_existing(f"{ABALONE_DATA_DIR}/Full/full_gaussian*_hypers*.dat")

    raise ValueError("dataset must be 'qm7' or 'abalone'")


def scalar(df: pd.DataFrame, *columns: str) -> float:
    for col in columns:
        if col in df.columns:
            return float(df[col].iloc[0])
    raise KeyError(f"None of {columns} found in columns {list(df.columns)}")


def fold_from_name(name: str) -> float:
    return pd.to_numeric(pd.Series([Path(name).name]).str.extract(r"fold=(\d+)")[0], errors="coerce").iloc[0]


def to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_fold_from_source(df: pd.DataFrame) -> pd.DataFrame:
    df["Fold"] = pd.to_numeric(df["__source__"].astype(str).str.extract(r"fold=(\d+)")[0], errors="coerce").astype("Int64")
    return df


def add_s_from_source(df: pd.DataFrame) -> pd.DataFrame:
    df["s"] = pd.to_numeric(df["__source__"].astype(str).str.extract(r"_s=(\d+)")[0], errors="coerce")
    return df

# Shared data loaders
def load_full_baseline(dataset: str) -> tuple[float, float]:
    files = full_baseline_files(dataset)
    err_cols = ("MAE_std", "MAE Std", "MAE_orig") if dataset == "qm7" else ("RMSE_std", "RMSE Std", "RMSE_orig")

    dfs = [pd.read_csv(f) for f in files]
    err = float(np.mean([scalar(df, *err_cols) for df in dfs]))
    elbo = float(np.mean([scalar(df, "Logml", "ELBO") for df in dfs]))
    return err, elbo


def load_full_folds_qm7() -> pd.DataFrame:
    rows = []
    for f in full_baseline_files("qm7"):
        df = pd.read_csv(f)
        rows.append({
            "Fold": fold_from_name(f),
            "MAE": scalar(df, "MAE_std", "MAE Std", "MAE_orig"),
            "ELBO": scalar(df, "Logml", "ELBO"),
            "TrK": scalar(df, "TrK") if "TrK" in df.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Fold").reset_index(drop=True)


def load_trace(dataset: str) -> pd.DataFrame:
    if dataset == "qm7":
        try:
            files = full_baseline_files("qm7")
        except FileNotFoundError:
            files = first_existing(f"{QM7_DATA_DIR}/Full/trace_laplacian_fold=*_kernelwidth=*_noise2=*.csv")
        return pd.DataFrame([{"Fold": fold_from_name(f), "TrK": float(pd.read_csv(f)["TrK"].iloc[0])} for f in files]).sort_values("Fold").reset_index(drop=True)

    if dataset == "abalone":
        try:
            files = full_baseline_files("abalone")
        except FileNotFoundError:
            files = first_existing(f"{ABALONE_DATA_DIR}/Full/trace_gaussian*.csv")
        rows = []
        for f in files:
            df = pd.read_csv(f)
            rows.append({
                "TrK": float(df["TrK"].iloc[0]),
                "Kernelwidth": float(df["Kernelwidth"].iloc[0]) if "Kernelwidth" in df.columns else np.nan,
                "Noise2": float(df["Noise2"].iloc[0]) if "Noise2" in df.columns else np.nan,
            })
        return pd.DataFrame(rows)

    raise ValueError("dataset must be 'qm7' or 'abalone'")


def method_patterns(dataset: str, s: int = 16) -> dict[str, tuple[str, ...]]:
    """
    Patterns for method result files.

    New convention from ``run_full_experiment``:
      QM7:     method_fold=<k>_kernelwidth=<ell>_noise2=<sigma2>_m=2048[ _s=<s>].csv
      Abalone: method_kernelwidth=<ell>_noise2=<sigma2>_m=2048[ _s=<s>].csv

    The DeltaVFE patterns intentionally target the per-s files only and do NOT match ``deltavfe_ALLS...csv``!
    """
    root = QM7_DATA_DIR if dataset == "qm7" else ABALONE_DATA_DIR
    if dataset == "qm7":
        return {
            "Greedy": (
                f"{root}/Greedy/greedy_fold=*_kernelwidth=*_noise2=*_m=2048.csv",
                f"{root}/Greedy/greedy_*fold=*_m=2048*.csv",
            ),
            "RPCholesky": (
                f"{root}/RPCholesky/rpcholesky_fold=*_kernelwidth=*_noise2=*_m=2048.csv",
                f"{root}/RPCholesky/rpcholesky_*fold=*_m=2048*.csv",
            ),
            "DeltaVFE": (
                f"{root}/DeltaVFE/deltavfe_fold=*_kernelwidth=*_noise2=*_m=2048_s={s}.csv",
                f"{root}/DeltaVFE/deltavfe_*fold=*_m=2048*_s={s}*.csv",
            )}

    if dataset == "abalone":
        return {
            "Greedy": (
                f"{root}/Greedy/greedy_kernelwidth=*_noise2=*_m=2048.csv",
                f"{root}/Greedy/greedy_*_m=2048*.csv",
            ),
            "RPCholesky": (
                f"{root}/RPCholesky/rpcholesky_kernelwidth=*_noise2=*_m=2048.csv",
                f"{root}/RPCholesky/rpcholesky_*_m=2048*.csv",
            ),
            "DeltaVFE": (
                f"{root}/DeltaVFE/deltavfe_kernelwidth=*_noise2=*_m=2048_s={s}.csv",
                f"{root}/DeltaVFE/deltavfe_*_m=2048*_s={s}*.csv",
            )}

    raise ValueError("dataset must be 'qm7' or 'abalone'")


def load_method_data(dataset: str, trace: pd.DataFrame, s: int = 16) -> pd.DataFrame:
    df = pd.concat([load_csvs_any(*patterns).assign(Method=method) for method, patterns in method_patterns(dataset, s).items()], ignore_index=True)
    to_numeric(df, ["RunID", "Seed", "Rank", "Kernelwidth", "Noise2", "RMSE", "MAE", "ELBO", "TraceErr"])
    df = df.dropna(subset=["Rank"]).copy()
    df["Rank"] = df["Rank"].astype(int)

    if dataset == "qm7":
        if "Fold" not in df.columns:
            df = add_fold_from_source(df)
        df = df.dropna(subset=["Fold"])
        df["Fold"] = pd.to_numeric(df["Fold"], errors="coerce").astype("Int64")
        df = df.merge(trace, on="Fold", how="left")
        return df.sort_values(["Method", "Fold", "Rank"]).reset_index(drop=True)

    if {"Kernelwidth", "Noise2"}.issubset(df.columns) and {"Kernelwidth", "Noise2"}.issubset(trace.columns):
        df = df.merge(trace[["Kernelwidth", "Noise2", "TrK"]].drop_duplicates(), on=["Kernelwidth", "Noise2"], how="left")
    else:
        df["TrK"] = float(trace["TrK"].mean())
    return df.sort_values(["Method", "Rank"]).reset_index(drop=True)


def choose_one_hyper_setting(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Kernelwidth", "Noise2"}.issubset(df.columns):
        return df.copy()
    kw, nz = df.groupby(["Kernelwidth", "Noise2"]).size().sort_values(ascending=False).index[0]
    return df[(df["Kernelwidth"] == kw) & (df["Noise2"] == nz)].copy()


def choose_one_hyper_setting_per_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not {"Kernelwidth", "Noise2"}.issubset(df.columns):
        return df.copy()
    parts = []
    for _, g in df.groupby(group_cols, dropna=False):
        kw, nz = g.groupby(["Kernelwidth", "Noise2"]).size().sort_values(ascending=False).index[0]
        parts.append(g[(g["Kernelwidth"] == kw) & (g["Noise2"] == nz)].copy())
    return pd.concat(parts, ignore_index=True)


# Aggregation
def aggregate_prediction(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    metric = "MAE" if dataset == "qm7" else "RMSE"
    groups = ["Method", "Fold", "Rank"] if dataset == "qm7" else ["Method", *( ["RunID"] if "RunID" in df.columns else [] ), "Rank"]
    return choose_one_hyper_setting(df).dropna(subset=[metric]).groupby(groups, as_index=False).agg(**{metric: (metric, "mean")})


def aggregate_vfe(df: pd.DataFrame, dataset: str, elbo_full: float) -> pd.DataFrame:
    needed = ["ELBO", "TraceErr", "MAE" if dataset == "qm7" else "RMSE"]
    groups = ["Method", "Fold", "Rank"] if dataset == "qm7" else ["Method", *( ["RunID"] if "RunID" in df.columns else [] ), "Rank"]
    extra = {c: (c, "first") for c in ["Kernelwidth", "Noise2", "TrK"] if c in df.columns}
    out = (choose_one_hyper_setting(df).dropna(subset=needed).groupby(groups, as_index=False).agg(ELBO=("ELBO", "mean"), TraceErr=("TraceErr", "mean"), **extra))
    out["TraceErrRel"] = out["TraceErr"].abs() / out["TrK"]
    out["RelativeVFE"] = (out["ELBO"].sub(elbo_full).abs() / abs(elbo_full)).clip(lower=VFE_GAP_FLOOR)
    return out


def aggregate_qm7_s(df: pd.DataFrame, has_s: bool) -> pd.DataFrame:
    group = ["Method", "s"] if has_s else ["Method"]
    out_group = ["Method", *( ["s"] if has_s else [] ), "Fold", "Rank"]
    return (choose_one_hyper_setting_per_group(df, group).dropna(subset=["MAE", "VFEGapRel", "TraceErrRel"])
            .groupby(out_group, as_index=False).agg(MAE=("MAE", "mean"), VFEGapRel=("VFEGapRel", "mean"), TraceErrRel=("TraceErrRel", "mean")))


# Plot helpers
def set_log2_x(ax):
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(axis="x", style="plain")


def format_panel(ax, ylabel: str, log_y: bool = True):
    ax.set_xlabel(r"Rank $r$")
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
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