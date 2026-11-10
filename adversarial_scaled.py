"""
RBF two-objective stress test.

Goal
----
Construct a standard RBF kernel whose best low-rank Frobenius subspace is nearly
orthogonal to a Cholesky subspace that is much better for the VFE objective.

Construction
------------
- 10 well-separated tight input clusters in one dimension.
- 6 "nuisance" clusters contain 25 points each.
- 4 "signal" clusters contain 20 points each.
- Because the clusters are far apart relative to the RBF lengthscale, K is
  approximately block diagonal.
- The larger nuisance blocks have the larger leading eigenvalues, so the
  Frobenius-optimal rank-4 eigenspace lies in the nuisance-cluster subspace.
- The response y is supported only on the four signal clusters: y = amplitude on them and 0 elsewhere
  (amplitude = 1 by default, i.e. about one unit of standardized energy per response-cluster direction).
  Optionally y = f + noise_std * N(0, I) (noisy observations), or normalize=True: y rescaled so that
  y^T (K + sigma2 I)^(-1) y = n (the former high signal-to-noise version, response amplitude about 7.75).
  The construction is deliberately adversarial in how y is aligned with the kernel eigenspaces; it is not a typical
  draw from the GP prior.

Measures
--------
- VFE gap F(K) - L(K~) per data point (the objective).
- Share of the data fit explained (data_fit_share): fraction of the full-kernel reduction of the quadratic data-fit
  term y^T(.)^(-1)y relative to rank 0 that the rank-r approximation recovers. Invariant to the scale of y for a
  fixed approximation.

This script runs the main example (amplitude 1): VFE gap and data-fit share against the rank. The sensitivity
experiments (amplitude sweep, noisy responses, high signal-to-noise version, low-output variants) are in
run_rbf_stress_test_sensitivity.py, which reuses the functions of this script.

Methods
-------
- RPCholesky
- Delta-VFE, s=8 (candidates drawn proportional to the residual diagonal, all exact-scored, best accepted)
- Exhaustive exact Delta-VFE greedy oracle
- Best Frobenius rank-r approximation (Eckart-Young oracle)

Repository functions used
-------------------------
- RPCholesky:                      rpcholesky.rpcholesky_pivots
- Delta-VFE:                       delta_vfe_fast.delta_vfe_pivots_fast
- Exhaustive exact greedy:         delta_vfe_fast.delta_vfe_pivots_fast with s = n (every remaining point is scored)
- Cholesky factor from pivots:     helpers.build_L_from_pivots_paper
- VFE bound / history:             metrics.vfe_bound, helpers.compute_B, helpers.vfe_history
- Full-GP log marginal likelihood: metrics.gp_neg_log_marginal_likelihood
- RBF kernel:                      kernels.gaussian_kernel
- Plotting theme:                  plot_utils

Output: Data/Adversarial/ (CSV, diagnostics), Plots/Adversarial/rbf_clusters.pdf (data overview and cluster zoom),
Plots/Adversarial/rbf_rank_vs_vfe_gap_and_data_fit.pdf (VFE gap per point and
share of the data fit explained against the rank)
"""

from itertools import combinations
import re
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from delta_vfe_fast import delta_vfe_pivots_fast
from helpers import build_L_from_pivots_paper, compute_B, vfe_history
from kernels import gaussian_kernel
from metrics import gp_neg_log_marginal_likelihood, vfe_bound
from matplotlib import gridspec
from plot_utils import (FIGSIZE_ROW, GRID_ROW, PLOTS_DIR, STYLE, VFE_PP_LABEL, consistent_log_ticks,  # also applies
                        shared_legend, viridis)                                                      # the theme
from rpcholesky import rpcholesky_pivots

# Methods use the line styles of the dataset figures (plot_utils.STYLE: RPCholesky 0.33 blue, Delta-VFE 0.67 green).
# The two deterministic oracles take mid-range viridis colours next to the method they belong to, avoiding the
# near-black and pale-yellow ends: the Frobenius oracle purple (next to RPCholesky), the exhaustive Delta-VFE greedy
# teal (next to Delta-VFE).
LINE_KWS = dict(linewidth=1.6, markersize=4.8, markeredgewidth=0.0)
ORACLE_STYLE = {
    "Best Frobenius rank-r":             dict(color=viridis(0.12), linestyle="--", marker="*", markersize=7.0),
    "Exhaustive exact Delta-VFE greedy": dict(color=viridis(0.50), linestyle="-.", marker="D"),
}
# Data overview: zero-output and response clusters with two different symbols and two viridis colours far apart in
# lightness (dark purple vs. green), so that they also differ in black and white
CLUSTER_COLORS = {"zero": viridis(0.12), "response": viridis(0.72)}
CLUSTER_MARKERS = {"zero": "o", "response": "D"}


def method_style(name):
    """plot_utils.STYLE entry of a method name of this script, e.g. "Delta-VFE (s=8)" -> STYLE["DeltaVFE"]."""
    key = "DeltaVFE" if name.startswith("Delta-VFE") else name
    return STYLE[key]


def full_gp_lml(K, y, sigma2):
    """Exact full-GP log marginal likelihood (no jitter)."""
    return -gp_neg_log_marginal_likelihood(K, y, sigma2, jitter=0.0)


def vfe(K, L, y, sigma2):
    """VFE bound of the approximation K ~ L L^T."""
    return vfe_bound(K, L, compute_B(L, sigma2), sigma2, y)


def data_fit_share(K, L, y, sigma2):
    """
    Share of the data fit explained by K ~ L L^T:
        eta = (y^T y / sigma2 - y^T (L L^T + sigma2 I)^(-1) y) / (y^T y / sigma2 - y^T (K + sigma2 I)^(-1) y),
    i.e. the fraction of the full-kernel reduction of the quadratic data-fit term (relative to rank 0) that is recovered.
    (L L^T + sigma2 I)^(-1) is applied with the Woodbury identity via compute_B.
    """
    yy = float(y @ y) / sigma2
    full = float(y @ np.linalg.solve(K + sigma2 * np.eye(len(y)), y))
    if L.shape[1] == 0:
        return 0.0
    g = L.T @ y
    approx = yy - float(g @ compute_B(L, sigma2) @ g) / sigma2 ** 2
    return (yy - approx) / (yy - full)


def gap_terms(K, L, y, sigma2):
    """
    Split of the VFE gap F(K) - L(K~) of K~ = L L^T into its three terms (they sum to the gap):
        complexity = 1/2 [log|L L^T + sigma2 I| - log|K + sigma2 I|]          (<= 0)
        data_fit   = 1/2 [y^T (L L^T + sigma2 I)^(-1) y - y^T (K + sigma2 I)^(-1) y]
        trace      = tr(K - L L^T) / (2 sigma2)
    """
    n = len(y)
    C = K + sigma2 * np.eye(n)
    full_fit = float(y @ np.linalg.solve(C, y))
    full_logdet = np.linalg.slogdet(C)[1]
    r = L.shape[1]
    if r == 0:
        logdet, fit = n * np.log(sigma2), float(y @ y) / sigma2
    else:
        B = compute_B(L, sigma2)
        g = L.T @ y
        fit = float(y @ y) / sigma2 - float(g @ B @ g) / sigma2 ** 2
        # matrix determinant lemma: |L L^T + sigma2 I| = sigma2^n |I + L^T L / sigma2| = sigma2^n / |B|
        logdet = n * np.log(sigma2) - np.linalg.slogdet(B)[1]
    trace = (float(np.trace(K)) - float(np.sum(L * L))) / (2.0 * sigma2)
    return {"complexity": 0.5 * (logdet - full_logdet), "data_fit": 0.5 * (fit - full_fit), "trace": trace}


def data_fit_history(K, y, sigma2, pivots):
    """data_fit_share of the approximations from the first r pivots, r = 1, ..., len(pivots)."""
    L = build_L_from_pivots_paper(K, pivots)
    return np.array([data_fit_share(K, L[:, :r], y, sigma2) for r in range(1, L.shape[1] + 1)])


def rbf_kernel(x, lengthscale):
    return gaussian_kernel(x[:, None], x[:, None], lengthscale)


def build_problem(nuisance_clusters=6, signal_clusters=4, nuisance_size=25,signal_size=20, cluster_width=0.02,
                  cluster_spacing=1.0, lengthscale=0.08, sigma2=1.0, amplitude=1.0, noise_std=0.0, noise_seed=0,
                  normalize=False):
    n_clusters = nuisance_clusters + signal_clusters
    centers = (np.arange(n_clusters, dtype=float) - 0.5 * (n_clusters - 1)) * cluster_spacing

    xs = []
    labels = []
    signal_mask = []

    for c in range(n_clusters):
        size = nuisance_size if c < nuisance_clusters else signal_size
        points = np.linspace(centers[c] - cluster_width / 2.0, centers[c] + cluster_width / 2.0, size)
        xs.append(points)
        labels.extend([c] * size)
        signal_mask.extend([c >= nuisance_clusters] * size)

    x = np.concatenate(xs)
    labels = np.asarray(labels, dtype=int)
    signal_mask = np.asarray(signal_mask, dtype=bool)

    K = rbf_kernel(x, lengthscale)

    # Response f is constant (= amplitude) on the signal clusters and zero elsewhere.
    f = np.zeros(len(x), dtype=float)
    f[signal_mask] = amplitude

    if normalize:
        # Former high-SNR version: total Mahalanobis energy equal to its prior expectation n
        # (concentrated in the four response-cluster directions).
        C = K + sigma2 * np.eye(len(x))
        f *= np.sqrt(len(x) / float(f @ np.linalg.solve(C, f)))

    # Optional noisy observations y = f + noise_std * eps, eps ~ N(0, I).
    y = f + noise_std * np.random.default_rng(noise_seed).standard_normal(len(x)) if noise_std > 0 else f.copy()

    representatives = []
    for c in range(n_clusters):
        idx = np.flatnonzero(labels == c)
        representatives.append(int(idx[len(idx) // 2]))

    return {
        "x": x,
        "K": K,
        "y": y,
        "f": f,
        "sigma2": sigma2,
        "labels": labels,
        "signal_mask": signal_mask,
        "centers": centers,
        "representatives": np.asarray(representatives, dtype=int),
        "nuisance_clusters": nuisance_clusters,
        "signal_clusters": signal_clusters,
        "nuisance_size": nuisance_size,
        "signal_size": signal_size,
        "lengthscale": lengthscale,
        "cluster_width": cluster_width,
        "cluster_spacing": cluster_spacing,
    }


def exhaustive_delta_vfe(K, y, sigma2, rank):
    """Exact greedy Delta-VFE over every admissible pivot (Delta-VFE with s = n)."""
    pivots = np.asarray(delta_vfe_pivots_fast(K, y, rank, sigma2, s=len(y), seed=0))
    return build_L_from_pivots_paper(K, pivots), pivots, vfe_history(K, y, sigma2, pivots)


def exhaustive_data_fit(K, y, sigma2, pivots):
    """Data-fit share of the exhaustive greedy approximations, ranks 1, ..., len(pivots)."""
    return data_fit_history(K, y, sigma2, pivots)


def frobenius_oracle_curve(K, y, sigma2, max_rank):
    """Best rank-r Frobenius approximations by eigentruncation."""
    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    full = full_gp_lml(K, y, sigma2)
    gaps = []
    fro_errors = []
    shares = []

    K_norm = np.linalg.norm(K, ord="fro")

    for r in range(1, max_rank + 1):
        U = eigvecs[:, :r]
        lam = eigvals[:r]
        K_tilde = (U * lam) @ U.T

        L_value = vfe(K, U * np.sqrt(np.maximum(lam, 0.0)), y, sigma2)
        shares.append(data_fit_share(K, U * np.sqrt(np.maximum(lam, 0.0)), y, sigma2))

        gaps.append(full - L_value)
        fro_errors.append(np.linalg.norm(K - K_tilde, ord="fro") / K_norm)

    return (
        eigvals,
        eigvecs,
        np.asarray(gaps),
        np.asarray(fro_errors),
        np.asarray(shares),
    )


def cluster_subset_oracle(problem, rank):
    """
    Exhaustively search cluster representatives only.

    This is a cluster-level Cholesky oracle, not a proof of the globally VFE-optimal
    arbitrary rank-r PSD approximation.
    """
    K = problem["K"]
    y = problem["y"]
    sigma2 = problem["sigma2"]
    reps = problem["representatives"]
    labels = problem["labels"]
    nuisance_clusters = problem["nuisance_clusters"]

    full = full_gp_lml(K, y, sigma2)

    best_gap = np.inf
    best_clusters = None
    best_L = None

    for cluster_subset in combinations(range(len(reps)), rank):
        pivots = [reps[c] for c in cluster_subset]
        L = build_L_from_pivots_paper(K, pivots)
        gap = full - vfe(K, L, y, sigma2)

        if gap < best_gap:
            best_gap = gap
            best_clusters = tuple(cluster_subset)
            best_L = L

    return best_gap, best_clusters, best_L


def orthonormal_basis(A, tol=1e-12):
    Q, R = np.linalg.qr(A)
    diag = np.abs(np.diag(R))
    keep = diag > tol
    return Q[:, keep]


def principal_angles_deg(U, V):
    U = orthonormal_basis(U)
    V = orthonormal_basis(V)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


def signal_pivot_count(pivots, labels, nuisance_clusters):
    cluster_labels = labels[np.asarray(pivots, dtype=int)]
    return np.cumsum(cluster_labels >= nuisance_clusters)


def run_trials(problem, max_rank=10, trials=200, s=8):
    K = problem["K"]
    y = problem["y"]
    sigma2 = problem["sigma2"]
    labels = problem["labels"]
    nuisance_clusters = problem["nuisance_clusters"]

    full = full_gp_lml(K, y, sigma2)

    methods = {
        "RPCholesky": [],
        f"Delta-VFE (s={s})": [],
    }

    signal_counts = {
        name: []
        for name in methods
    }
    shares = {name: [] for name in methods}

    timings = {
        name: []
        for name in methods
    }

    for trial in range(trials):
        seed = trial

        start = time.perf_counter()
        piv_rpc = rpcholesky_pivots(K, max_rank, seed=seed)
        timings["RPCholesky"].append(time.perf_counter() - start)
        hist_rpc = vfe_history(K, y, sigma2, piv_rpc)
        methods["RPCholesky"].append(full - hist_rpc[1:])
        shares["RPCholesky"].append(data_fit_history(K, y, sigma2, piv_rpc))
        signal_counts["RPCholesky"].append(
            signal_pivot_count(piv_rpc, labels, nuisance_clusters)
        )

        name = f"Delta-VFE (s={s})"
        start = time.perf_counter()
        piv_exact = delta_vfe_pivots_fast(K, y, max_rank, sigma2, s=s, seed=seed)
        timings[name].append(time.perf_counter() - start)
        hist_exact = vfe_history(K, y, sigma2, piv_exact)
        methods[name].append(full - hist_exact[1:])
        shares[name].append(data_fit_history(K, y, sigma2, piv_exact))
        signal_counts[name].append(
            signal_pivot_count(piv_exact, labels, nuisance_clusters)
        )

    for name in methods:
        methods[name] = np.asarray(methods[name])
        signal_counts[name] = np.asarray(signal_counts[name])
        timings[name] = np.asarray(timings[name])
        shares[name] = np.asarray(shares[name])

    return methods, signal_counts, timings, shares


def summarize_methods(methods, signal_counts, timings, shares=None):
    rows = []

    for name, gaps in methods.items():
        for r in range(1, gaps.shape[1] + 1):
            vals = gaps[:, r - 1]
            sc = signal_counts[name][:, r - 1]

            rows.append({
                "method": name,
                "rank": r,
                "mean_vfe_gap": float(np.mean(vals)),
                "sd_vfe_gap": float(np.std(vals, ddof=1)),
                "ci95_vfe_gap": float(1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals))),
                "mean_signal_pivots": float(np.mean(sc)),
                "sd_signal_pivots": float(np.std(sc, ddof=1)),
                "mean_total_runtime_s": float(np.mean(timings[name])),
                **({"mean_data_fit_share": float(np.mean(shares[name][:, r - 1])),
                    "sd_data_fit_share": float(np.std(shares[name][:, r - 1], ddof=1)),
                    "ci95_data_fit_share": float(1.96 * np.std(shares[name][:, r - 1], ddof=1)
                                                 / np.sqrt(len(shares[name])))} if shares else {}),
            })

    return pd.DataFrame(rows)


def plot_clusters(problem, filename, cluster_id=None, labels_text=None):
    """Data overview (left) and zoom into one existing cluster (right) in one figure.

    The zoom defaults to the first response-bearing cluster; pass cluster_id=0 to inspect the first zero-response
    cluster instead. Neither x nor y is jittered or modified for this visualization.
    labels_text: optional legend labels (zero-output clusters, response clusters) replacing the default ones.
    """
    x = problem["x"]
    y = problem["y"]
    labels = problem["labels"]
    nuisance_clusters = problem["nuisance_clusters"]

    fig = plt.figure(figsize=FIGSIZE_ROW)
    gs = gridspec.GridSpec(2, 2, **GRID_ROW)
    ax, ax_zoom, legend_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # left: all clusters
    nuisance = labels < nuisance_clusters
    signal = ~nuisance

    amp = float(problem["f"][signal][0])          # signal level f on the response clusters (y = f [+ noise])
    noisy = not np.allclose(y, problem["f"])
    if labels_text is None:
        labels_text = ((r"Clusters with $f = 0$", rf"Clusters with $f = {amp:g}$") if noisy
                       else ("Zero output clusters", "Non-zero output clusters"))
    ax.scatter(x[nuisance], y[nuisance], s=22, color=CLUSTER_COLORS["zero"], marker=CLUSTER_MARKERS["zero"],
               label=labels_text[0])
    ax.scatter(x[signal], y[signal], s=20, color=CLUSTER_COLORS["response"], marker=CLUSTER_MARKERS["response"],
               label=labels_text[1])
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $y$")

    # right: one cluster
    if cluster_id is None:
        cluster_id = nuisance_clusters
    cluster_id = int(cluster_id)
    if not 0 <= cluster_id < len(problem["centers"]):
        raise ValueError("cluster_id must identify an existing cluster")

    mask = labels == cluster_id
    x_cluster = x[mask]
    y_cluster = y[mask]
    if len(x_cluster) == 0:
        raise ValueError("The selected cluster contains no points")

    # The response is intentionally constant inside each cluster. The
    # x-coordinates, however, are different and are plotted without jitter.
    center_y = float(np.mean(y_cluster))
    ypad = 0.5  # display-only padding, not a change in the response data
    kind = "response" if cluster_id >= nuisance_clusters else "zero"
    ax_zoom.scatter(x_cluster, y_cluster, s=56, color=CLUSTER_COLORS[kind], marker=CLUSTER_MARKERS[kind],
                    edgecolors="white", linewidths=0.7, zorder=3,
                    label=f"{len(x_cluster)} individual observations (zoomed cluster)")

    span = float(np.ptp(x_cluster))
    margin = max(0.08 * span, 1e-5)
    ax_zoom.set_xlim(float(np.min(x_cluster)) - margin, float(np.max(x_cluster)) + margin)
    if np.ptp(y_cluster) > 0:      # noisy responses: show the whole spread of the cluster
        pad = 0.15 * float(np.ptp(y_cluster))
        ax_zoom.set_ylim(float(np.min(y_cluster)) - pad, float(np.max(y_cluster)) + pad)
    else:
        ax_zoom.set_ylim(center_y - ypad, center_y + 0.25)
    ax_zoom.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax_zoom.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.4f"))
    ax_zoom.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax_zoom.set_xlabel(r"Input $x$")
    ax_zoom.set_ylabel(r"Output $y$")
    add_grid(ax)
    add_grid(ax_zoom)

    shared_legend(legend_ax, [ax, ax_zoom], ncol=3)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    fig.show()
    plt.close(fig)


def display_label(name):
    """Legend/title text of a method name: "Delta-VFE (s=8)" -> "$\\Delta$-VFE ($s=8$)" (data keeps the plain names)."""
    return re.sub(r"\(s=(\d+)\)", r"($s=\1$)", name.replace("Delta-VFE", r"$\Delta$-VFE"))


def add_grid(ax):
    """Major and minor grid as in the dataset figures (minor ticks in every decade on log axes)."""
    if ax.get_yscale() == "log":
        consistent_log_ticks(ax.yaxis)
    else:
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(True, which="major", linestyle="-", color="0.85")
    ax.grid(True, which="minor", linestyle=":", color="0.9")


def draw_rank_curves(ax, summary, fro_values, exhaustive_values, metric="vfe_gap", n=None):
    """
    Draw one measure against the rank on ax: RPCholesky and Delta-VFE (mean over the seeds with 95% confidence
    intervals as error bars, as in the dataset figures) and the two (deterministic) oracles.
    metric="vfe_gap": VFE gap F(K) - L(K~), divided by n if n is given (gap per data point, log scale);
    metric="data_fit_share": share of the data fit explained (fro_values / exhaustive_values are the oracle shares).
    """
    share = metric == "data_fit_share"
    scale = 1.0 if (n is None or share) else 1.0 / n
    for method in summary["method"].unique():
        data = summary[summary["method"] == method].sort_values("rank")
        mean = data[f"mean_{metric}"].to_numpy() * scale
        ci = data[f"ci95_{metric}"].fillna(0.0).to_numpy() * scale if f"ci95_{metric}" in data else 0.0 * mean
        lower = np.minimum(ci, np.maximum(mean * (1.0 - 1e-12), 0.0)) if not share else ci   # positive on log axes
        ax.errorbar(data["rank"], mean, yerr=[lower, ci], label=display_label(method), capsize=3,
                    elinewidth=1.0, capthick=1.0, zorder=3, **method_style(method), **LINE_KWS)
    ranks = np.arange(1, len(fro_values) + 1)
    # drawn with errorbar (no error) as well, so that the legend keeps the order of the methods
    for name, values in [("Best Frobenius rank-r", fro_values), ("Exhaustive exact Delta-VFE greedy", exhaustive_values)]:
        ax.errorbar(ranks, np.asarray(values) * scale, label=display_label(name), zorder=2,
                    **{**LINE_KWS, **ORACLE_STYLE[name]})

    ax.set_xlabel("Rank")
    if share:
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("Share of data fit explained")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(VFE_PP_LABEL if n is not None else r"$\mathcal{F}(K)-\mathcal{L}(\widetilde{K})$  [nats]")
    add_grid(ax)


def plot_vfe_curve(summary, fro_gaps, exhaustive_gaps, filename, n=None, metric="vfe_gap"):
    """
    One measure against the rank in its own figure (see draw_rank_curves): the VFE gap F(K) - L(K~) (per data point if
    n is given) or, with metric="data_fit_share", the share of the data fit explained.
    """
    fig = plt.figure(figsize=FIGSIZE_ROW)
    gs = gridspec.GridSpec(2, 1, **GRID_ROW)
    ax, legend_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    draw_rank_curves(ax, summary, fro_gaps, exhaustive_gaps, metric=metric, n=n)
    shared_legend(legend_ax, [ax], ncol=2)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    fig.show()
    plt.close(fig)


def plot_gap_and_data_fit(summary, fro_gaps, exhaustive_gaps, fro_shares, exhaustive_shares, filename, n=None):
    """
    VFE gap per data point (left) and share of the data fit explained (right) against the rank in one figure, with one
    legend of 4 columns below both panels.
    """
    fig = plt.figure(figsize=FIGSIZE_ROW)
    gs = gridspec.GridSpec(2, 2, **GRID_ROW)
    ax_gap, ax_share, legend_ax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    draw_rank_curves(ax_gap, summary, fro_gaps, exhaustive_gaps, metric="vfe_gap", n=n)
    draw_rank_curves(ax_share, summary, fro_shares, exhaustive_shares, metric="data_fit_share")
    shared_legend(legend_ax, [ax_gap], ncol=4)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    fig.show()
    plt.close(fig)


def plot_signal_pivots(summary, filename):
    fig, ax = plt.subplots(figsize=FIGSIZE_ROW)

    for method in summary["method"].unique():
        data = summary[summary["method"] == method].sort_values("rank")
        ax.plot(
            data["rank"],
            data["mean_signal_pivots"],
            label=display_label(method),
            **method_style(method),
            **LINE_KWS,
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Mean number of response-cluster pivots selected")
    ax.set_title("Where the stochastic methods spend their pivots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    fig.show()
    plt.close(fig)


def main():
    problem = build_problem(nuisance_clusters=6, signal_clusters=4, nuisance_size=25, signal_size=20,
                            cluster_width=0.02, cluster_spacing=1, lengthscale=0.08, sigma2=1.0, amplitude=1.0)

    max_rank = 12
    trials = 200
    s = 4

    output_dir = Path("Data/Adversarial")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(PLOTS_DIR) / "Adversarial"
    plots_dir.mkdir(parents=True, exist_ok=True)

    K = problem["K"]
    y = problem["y"]
    sigma2 = problem["sigma2"]
    full = full_gp_lml(K, y, sigma2)

    eigvals, eigvecs, fro_gaps, fro_errors, fro_shares = frobenius_oracle_curve(K, y, sigma2, max_rank)

    L_exhaustive, piv_exhaustive, hist_exhaustive = exhaustive_delta_vfe(K, y, sigma2, max_rank)
    exhaustive_gaps = full - hist_exhaustive[1:]
    exhaustive_shares = exhaustive_data_fit(K, y, sigma2, piv_exhaustive)

    rank_focus = 4
    cluster_gap, cluster_subset, L_cluster = cluster_subset_oracle(problem, rank=rank_focus)

    U_frob4 = eigvecs[:, :rank_focus]
    U_cluster4 = orthonormal_basis(L_cluster)
    angles = principal_angles_deg(U_frob4, U_cluster4)

    # Signal-indicator subspace, for an additional transparent diagnostic.
    signal_vectors = []
    labels = problem["labels"]
    for c in range(problem["nuisance_clusters"], problem["nuisance_clusters"] + problem["signal_clusters"]):
        vec = np.zeros(len(y))
        idx = np.flatnonzero(labels == c)
        vec[idx] = 1.0 / np.sqrt(len(idx))
        signal_vectors.append(vec)
    U_signal = np.column_stack(signal_vectors)
    signal_angles = principal_angles_deg(U_frob4, U_signal)

    methods, signal_counts, timings, shares = run_trials(problem, max_rank=max_rank, trials=trials, s=s)

    summary = summarize_methods(methods, signal_counts, timings, shares)
    summary.to_csv(output_dir / "rank_summary.csv", index=False)

    deterministic = pd.DataFrame({
        "rank": np.arange(1, max_rank + 1),
        "frobenius_oracle_vfe_gap": fro_gaps,
        "frobenius_relative_error": fro_errors,
        "exhaustive_delta_vfe_gap": exhaustive_gaps,
        "frobenius_oracle_data_fit_share": fro_shares,
        "exhaustive_delta_vfe_data_fit_share": exhaustive_shares,
    })
    deterministic.to_csv(
        output_dir / "deterministic_oracles.csv",
        index=False,
    )

    plot_clusters(problem, plots_dir / "rbf_clusters.pdf")
    plot_gap_and_data_fit(summary, fro_gaps, exhaustive_gaps, fro_shares, exhaustive_shares,
                          plots_dir / "rbf_rank_vs_vfe_gap_and_data_fit.pdf", n=len(y))

    #plot_signal_pivots(summary, plots_dir / "rbf_signal_pivots_vs_rank.png")

    with open(output_dir / "diagnostics.txt", "w", encoding="utf-8") as fh:
        fh.write("RBF clustered Frobenius-vs-VFE stress test\n")
        fh.write("=" * 60 + "\n\n")

        fh.write(f"n = {len(y)}\n")
        fh.write(
            f"clusters = {problem['nuisance_clusters']} nuisance x "
            f"{problem['nuisance_size']} points + "
            f"{problem['signal_clusters']} signal x "
            f"{problem['signal_size']} points\n"
        )
        fh.write(f"RBF lengthscale = {problem['lengthscale']}\n")
        fh.write(f"cluster width = {problem['cluster_width']}\n")
        fh.write(f"cluster spacing = {problem['cluster_spacing']}\n")
        fh.write(f"noise variance = {sigma2}\n")
        fh.write(
            "Mahalanobis energy y^T(K+sigma2 I)^(-1)y = "
            f"{float(y @ np.linalg.solve(K + sigma2*np.eye(len(y)), y)):.12f}\n\n"
        )

        fh.write("Leading eigenvalues of K:\n")
        fh.write(np.array2string(eigvals[:12], precision=6))
        fh.write("\n\n")

        fh.write(
            f"Best rank-{rank_focus} cluster-representative VFE subset: "
            f"{cluster_subset}\n"
        )
        fh.write(
            f"Cluster-representative VFE gap at rank {rank_focus}: "
            f"{cluster_gap:.12g}\n"
        )
        fh.write(
            f"Frobenius-oracle VFE gap at rank {rank_focus}: "
            f"{fro_gaps[rank_focus-1]:.12g}\n"
        )
        fh.write(
            f"Exhaustive greedy Delta-VFE gap at rank {rank_focus}: "
            f"{exhaustive_gaps[rank_focus-1]:.12g}\n\n"
        )

        fh.write(
            "Principal angles (degrees) between top-4 Frobenius eigenspace "
            "and best rank-4 cluster-representative Cholesky subspace:\n"
        )
        fh.write(np.array2string(angles, precision=6))
        fh.write("\n\n")

        fh.write(
            "Principal angles (degrees) between top-4 Frobenius eigenspace "
            "and response-cluster indicator subspace:\n"
        )
        fh.write(np.array2string(signal_angles, precision=6))
        fh.write("\n\n")

        fh.write("Exhaustive Delta-VFE pivot clusters:\n")
        fh.write(
            np.array2string(
                problem["labels"][piv_exhaustive],
                separator=", ",
            )
        )
        fh.write("\n\n")

        fh.write("Mean total runtime over all 10 ranks:\n")
        for name, arr in timings.items():
            fh.write(f"  {name}: {np.mean(arr):.6f} s\n")

    print("Experiment complete.")
    print("Output directory:", output_dir)

    print("\nKey rank-4 diagnostics:")
    print("Best cluster-representative subset:", cluster_subset)
    print("Principal angles Frobenius vs cluster-VFE subspace (deg):", angles)
    print("Frobenius-oracle rank-4 VFE gap:", fro_gaps[3])
    print("Best cluster-representative rank-4 VFE gap:", cluster_gap)
    print("Exhaustive greedy rank-4 VFE gap:", exhaustive_gaps[3])

    rank4 = summary[summary["rank"] == 4][
        ["method", "mean_vfe_gap", "mean_signal_pivots", "mean_total_runtime_s"]
    ]
    print(rank4.to_string(index=False))


if __name__ == "__main__":
    main()