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
- The response y is supported only on the four signal clusters and is scaled so
  that y^T (K + sigma2 I)^(-1) y = n.

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

Output: Data/RBFStressTest/ (CSV, diagnostics) and Plots/rbf_*.pdf
"""

from itertools import combinations
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
from plot_utils import PLOTS_DIR  # importing plot_utils also applies the plotting theme of the repository
from rpcholesky import rpcholesky_pivots


def full_gp_lml(K, y, sigma2):
    """Exact full-GP log marginal likelihood (no jitter)."""
    return -gp_neg_log_marginal_likelihood(K, y, sigma2, jitter=0.0)


def vfe(K, L, y, sigma2):
    """VFE bound of the approximation K ~ L L^T."""
    return vfe_bound(K, L, compute_B(L, sigma2), sigma2, y)


def rbf_kernel(x, lengthscale):
    return gaussian_kernel(x[:, None], x[:, None], lengthscale)


def build_problem(nuisance_clusters=6, signal_clusters=4, nuisance_size=25,signal_size=20, cluster_width=0.02,
                  cluster_spacing=1.0, lengthscale=0.08, sigma2=1.0):
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

    # Response is constant on the signal clusters and zero elsewhere.
    y = np.zeros(len(x), dtype=float)
    y[signal_mask] = 1.0

    # Normalize to the typical full-GP Mahalanobis energy:
    # E[y^T C^-1 y] = n for y \sim N(0,C).
    C = K + sigma2 * np.eye(len(x))
    energy = float(y @ np.linalg.solve(C, y))
    y *= np.sqrt(len(x) / energy)

    representatives = []
    for c in range(n_clusters):
        idx = np.flatnonzero(labels == c)
        representatives.append(int(idx[len(idx) // 2]))

    return {
        "x": x,
        "K": K,
        "y": y,
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


def frobenius_oracle_curve(K, y, sigma2, max_rank):
    """Best rank-r Frobenius approximations by eigentruncation."""
    eigvals, eigvecs = np.linalg.eigh(K)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    full = full_gp_lml(K, y, sigma2)
    gaps = []
    fro_errors = []

    K_norm = np.linalg.norm(K, ord="fro")

    for r in range(1, max_rank + 1):
        U = eigvecs[:, :r]
        lam = eigvals[:r]
        K_tilde = (U * lam) @ U.T

        L_value = vfe(K, U * np.sqrt(np.maximum(lam, 0.0)), y, sigma2)

        gaps.append(full - L_value)
        fro_errors.append(np.linalg.norm(K - K_tilde, ord="fro") / K_norm)

    return (
        eigvals,
        eigvecs,
        np.asarray(gaps),
        np.asarray(fro_errors),
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
        signal_counts["RPCholesky"].append(
            signal_pivot_count(piv_rpc, labels, nuisance_clusters)
        )

        name = f"Delta-VFE (s={s})"
        start = time.perf_counter()
        piv_exact = delta_vfe_pivots_fast(K, y, max_rank, sigma2, s=s, seed=seed)
        timings[name].append(time.perf_counter() - start)
        hist_exact = vfe_history(K, y, sigma2, piv_exact)
        methods[name].append(full - hist_exact[1:])
        signal_counts[name].append(
            signal_pivot_count(piv_exact, labels, nuisance_clusters)
        )

    for name in methods:
        methods[name] = np.asarray(methods[name])
        signal_counts[name] = np.asarray(signal_counts[name])
        timings[name] = np.asarray(timings[name])

    return methods, signal_counts, timings


def summarize_methods(methods, signal_counts, timings):
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
                "mean_signal_pivots": float(np.mean(sc)),
                "sd_signal_pivots": float(np.std(sc, ddof=1)),
                "mean_total_runtime_s": float(np.mean(timings[name])),
            })

    return pd.DataFrame(rows)


def plot_data(problem, filename):
    # sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    x = problem["x"]
    y = problem["y"]
    labels = problem["labels"]
    nuisance_clusters = problem["nuisance_clusters"]

    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    nuisance = labels < nuisance_clusters
    signal = ~nuisance

    ax.scatter(x[nuisance], y[nuisance], s=18, label="Zero output clusters")
    ax.scatter(x[signal], y[signal], s=18, label="Non-zero output clusters")

    #ax.set_xscale("log",base=2)
    #ax.set_yscale("log")
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $y$")
    #ax.set_title("Adversarial example - Clustered data")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    fig.show()
    plt.close(fig)

def plot_cluster_zoom(problem, filename, cluster_id=None):
    """Zoom in on one *existing* cluster to reveal individual x locations.

    Defaults to the first response-bearing cluster; pass cluster_id=0 to
    inspect the first zero-response cluster instead. Neither x nor y is
    jittered or modified for this visualization.
    """
    x = problem["x"]
    y = problem["y"]
    labels = problem["labels"]
    n_nuisance = problem["nuisance_clusters"]

    if cluster_id is None:
        cluster_id = n_nuisance
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
    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.scatter(x_cluster, y_cluster, s=56, edgecolors="white",
               linewidths=0.7, zorder=3,
               label=f"{len(x_cluster)} individual observations")

    span = float(np.ptp(x_cluster))
    margin = max(0.08 * span, 1e-5)
    ax.set_xlim(float(np.min(x_cluster)) - margin,
                float(np.max(x_cluster)) + margin)
    ax.set_ylim(center_y - ypad, center_y + 0.25)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(7))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.4f"))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    kind = "Non-zero response cluster" if cluster_id >= n_nuisance else "Zero-response"
    # ax.set_title(f"Zoom: {kind.lower()} cluster {cluster_id + 1} "
    #              f"({len(x_cluster)} distinct input points)")
    ax.set_xlabel(r"Input $x$")
    ax.set_ylabel(r"Output $y$")
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    fig.show()
    plt.close(fig)


def plot_vfe_curve(summary, fro_gaps, exhaustive_gaps, filename):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    markers = ['s', '^', 'o']
    for method, marker in zip(summary["method"].unique(), markers):
        data = summary[summary["method"] == method].sort_values("rank")
        ax.plot(
            data["rank"],
            data["mean_vfe_gap"],
            marker=marker,
            label=method,
        )

    ranks = np.arange(1, len(fro_gaps) + 1)

    ax.plot(ranks, fro_gaps, marker="*", linestyle="--", label="Best Frobenius rank-r")

    ax.plot(
        ranks,
        exhaustive_gaps,
        marker="o",
        linestyle="--",
        label="Exhaustive exact Delta-VFE greedy",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("VFE gap")
    #ax.set_title("RBF clustered example: rank vs VFE gap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    fig.show()
    plt.close(fig)


def plot_signal_pivots(summary, filename):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for method in summary["method"].unique():
        data = summary[summary["method"] == method].sort_values("rank")
        ax.plot(
            data["rank"],
            data["mean_signal_pivots"],
            marker="o",
            label=method,
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
                            cluster_width=0.02, cluster_spacing=1, lengthscale=0.08, sigma2=1.0)

    max_rank = 12
    trials = 200
    s = 8

    output_dir = Path("Data/QM7_Folds/Data/RBFStressTest")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(PLOTS_DIR)

    K = problem["K"]
    y = problem["y"]
    sigma2 = problem["sigma2"]
    full = full_gp_lml(K, y, sigma2)

    eigvals, eigvecs, fro_gaps, fro_errors = frobenius_oracle_curve(K, y, sigma2, max_rank)

    L_exhaustive, piv_exhaustive, hist_exhaustive = exhaustive_delta_vfe(K, y, sigma2, max_rank)
    exhaustive_gaps = full - hist_exhaustive[1:]

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

    methods, signal_counts, timings = run_trials(problem, max_rank=max_rank, trials=trials, s=s)

    summary = summarize_methods(methods, signal_counts, timings)
    summary.to_csv(output_dir / "rank_summary.csv", index=False)

    deterministic = pd.DataFrame({
        "rank": np.arange(1, max_rank + 1),
        "frobenius_oracle_vfe_gap": fro_gaps,
        "frobenius_relative_error": fro_errors,
        "exhaustive_delta_vfe_gap": exhaustive_gaps,
    })
    deterministic.to_csv(
        output_dir / "deterministic_oracles.csv",
        index=False,
    )

    plot_data(problem, plots_dir / "rbf_data_setup.pdf")
    plot_cluster_zoom(problem, plots_dir / "rbf_cluster_zoom.pdf")
    plot_vfe_curve(summary, fro_gaps, exhaustive_gaps, plots_dir / "rbf_rank_vs_vfe_gap.pdf")
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