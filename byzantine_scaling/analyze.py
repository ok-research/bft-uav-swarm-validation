"""Produce paper figure for the Byzantine-resilience scaling experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "lines.markersize": 4.0,
    "grid.linewidth": 0.4, "grid.alpha": 0.30,
    "legend.framealpha": 0.92, "legend.edgecolor": "0.7",
    "savefig.dpi": 600,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

OK_BLUE   = "#0072B2"
OK_GREEN  = "#009E73"
OK_RED    = "#D55E00"


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.8")
    ax.set_axisbelow(True)


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    npz_path = out_dir / "experiment_data.npz"
    if not npz_path.is_file():
        sys.exit(f"Missing {npz_path} — run python run_experiment.py first.")

    data = {k: np.load(npz_path, allow_pickle=True)[k] for k in np.load(npz_path).files}
    fs = data["f"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)

    # Differentiate detection rate (open markers) from final honest R (solid)
    # to avoid the two 1.0-lines overlapping.
    ax1.plot(fs, data["detection_rate"], marker="o", mfc="none", mec=OK_GREEN,
             color=OK_GREEN, linewidth=1.0, label="detection rate")
    ax1.plot(fs, data["fp_rate"], marker="x", color=OK_RED,
             linewidth=1.0, label="FP rate")
    ax1.plot(fs, data["final_honest_median_R"], marker=".", color=OK_BLUE,
             linewidth=1.0, label="final honest median R")
    ax1.axvline(1/3, color="black", ls="--", alpha=0.5,
                label=r"$f = 1/3$ (PBFT bound)")
    ax1.set_xlabel(r"Byzantine fraction $f$")
    ax1.set_ylabel("Rate / Reputation")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="center right")
    _clean_axes(ax1)

    ax2.plot(fs, data["median_detection_latency"], marker="o",
             color=OK_BLUE, linewidth=1.2)
    ax2.axvline(1/3, color="black", ls="--", alpha=0.5,
                label=r"$f = 1/3$ (PBFT bound)")
    ax2.set_xlabel(r"Byzantine fraction $f$")
    ax2.set_ylabel("Median detection latency (rounds)")
    ax2.set_ylim(0, 25)
    ax2.legend(loc="upper right")
    _clean_axes(ax2)

    fig.savefig(out_dir / "byzantine_resilience_curve.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure written to {out_dir}/byzantine_resilience_curve.png")


if __name__ == "__main__":
    main()
