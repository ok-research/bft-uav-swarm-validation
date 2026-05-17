"""Analyse the polarization-metric sweep and produce paper figures + tables.

Reads results/experiment_data.npz and produces:
  results/auc_gap_vs_kappa.png
  results/auc_gap_heatmap.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    npz_path = out_dir / "experiment_data.npz"
    if not npz_path.is_file():
        sys.exit(f"Missing {npz_path} — run python run_experiment.py first.")

    data = {k: np.load(npz_path, allow_pickle=True)[k] for k in
            ("d", "kappa", "mu_spoof_norm", "auc_mahalanobis", "auc_cosine", "auc_gap")}

    DIM = sorted(set(data["d"].tolist()))
    KAPPA = sorted(set(data["kappa"].tolist()))
    MU = sorted(set(data["mu_spoof_norm"].tolist()))

    # Summary table by kappa
    print("\n## AUC gap (Mahalanobis − cosine) aggregated by κ\n")
    print("| κ | min gap | median gap | mean gap | max gap | Mahalanobis wins (%) |")
    print("|---:|---:|---:|---:|---:|---:|")
    for k in KAPPA:
        mask = data["kappa"] == k
        gaps = data["auc_gap"][mask]
        wins = float((gaps > 0).mean()) * 100
        print(f"| {k:.0f} | {gaps.min():+.4f} | {np.median(gaps):+.4f} | "
              f"{gaps.mean():+.4f} | {gaps.max():+.4f} | {wins:.1f} |")

    # Figure 1: AUC gap vs kappa, faceted by d, color by mu_spoof
    fig, axes = plt.subplots(1, len(DIM), figsize=(4 * len(DIM), 4), sharey=True)
    if len(DIM) == 1:
        axes = [axes]
    for ax, d_target in zip(axes, DIM):
        for mu in MU:
            mask = (data["d"] == d_target) & np.isclose(data["mu_spoof_norm"], mu)
            xs = sorted(set(data["kappa"][mask].tolist()))
            ys = [
                float(np.mean(data["auc_gap"][(data["d"] == d_target)
                                              & np.isclose(data["mu_spoof_norm"], mu)
                                              & (data["kappa"] == k)]))
                for k in xs
            ]
            ax.semilogx(xs, ys, marker="o", label=f"||μ||={mu}")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel(r"Covariance condition number $\kappa$")
        ax.set_title(f"d = {d_target}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("AUC(Mahalanobis) − AUC(cosine)")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Decision-statistic AUC gap as covariance anisotropy grows")
    plt.tight_layout()
    fig.savefig(out_dir / "auc_gap_vs_kappa.png", dpi=120)
    plt.close(fig)

    # Figure 2: gap heatmap, kappa × mu_spoof, averaged over d
    fig, ax = plt.subplots(figsize=(8, 4.5))
    grid = np.zeros((len(KAPPA), len(MU)))
    for i, k in enumerate(KAPPA):
        for j, mu in enumerate(MU):
            mask = (data["kappa"] == k) & np.isclose(data["mu_spoof_norm"], mu)
            grid[i, j] = float(np.mean(data["auc_gap"][mask]))
    vmax = max(abs(grid.min()), abs(grid.max()))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(MU)))
    ax.set_xticklabels([f"{m}" for m in MU])
    ax.set_yticks(range(len(KAPPA)))
    ax.set_yticklabels([f"{k}" for k in KAPPA])
    ax.set_xlabel(r"Spoof offset $\|\mu_\mathrm{spoof}\|$")
    ax.set_ylabel(r"Covariance condition number $\kappa$")
    ax.set_title("AUC(Mahalanobis) − AUC(cosine), averaged over $d \\in \\{4, 8, 16\\}$")
    for i in range(len(KAPPA)):
        for j in range(len(MU)):
            color = "white" if abs(grid[i, j]) > vmax * 0.5 else "black"
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="AUC gap (positive = Mahalanobis wins)")
    plt.tight_layout()
    fig.savefig(out_dir / "auc_gap_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
