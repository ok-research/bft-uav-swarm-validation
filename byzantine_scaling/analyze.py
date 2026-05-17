"""Produce paper figure for the Byzantine-resilience scaling experiment."""

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

    data = {k: np.load(npz_path, allow_pickle=True)[k] for k in np.load(npz_path).files}
    fs = data["f"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(fs, data["detection_rate"], "go-", label="detection rate")
    ax1.plot(fs, data["fp_rate"], "rx-", label="FP rate")
    ax1.plot(fs, data["final_honest_median_R"], "b.-", label="final honest median R")
    ax1.axvline(1/3, color="black", ls="--", alpha=0.5, label=r"$f = 1/3$ (PBFT bound)")
    ax1.set_xlabel("Byzantine fraction f")
    ax1.set_ylabel("Rate / Reputation")
    ax1.set_title("Detection metrics vs Byzantine fraction")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(fs, data["median_detection_latency"], "o-", color="C0", lw=2)
    ax2.axvline(1/3, color="black", ls="--", alpha=0.5, label=r"$f = 1/3$ (PBFT bound)")
    ax2.set_xlabel("Byzantine fraction f")
    ax2.set_ylabel("Median detection latency (rounds, log)")
    ax2.set_title("Detection latency vs Byzantine fraction")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "byzantine_resilience_curve.png", dpi=120)
    plt.close(fig)

    print(f"Figure written to {out_dir}/byzantine_resilience_curve.png")


if __name__ == "__main__":
    main()
