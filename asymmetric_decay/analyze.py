"""Analyse the parameter-sweep output and produce paper-ready figures + tables.

Reads results/experiment_data.npz produced by run_experiment.py and
generates two figures matching the paper:

    results/detection_latency_vs_ratio.png
    results/round_detection_rate_N100_f020.png

Plus a markdown summary table.

Usage:
    python analyze.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import Swarm, measure_run


def aggregate_by_ratio(data: dict) -> dict[float, dict]:
    """Group metrics by asymmetry ratio r."""
    out = {}
    for r_val in sorted(set(float(x) for x in data["r"])):
        mask = data["r"] == r_val
        out[r_val] = {
            "median_detection_latency": {
                "min": float(np.min(data["median_detection_latency"][mask])),
                "median": float(np.median(data["median_detection_latency"][mask])),
                "mean": float(np.mean(data["median_detection_latency"][mask])),
                "max": float(np.max(data["median_detection_latency"][mask])),
            },
            "detection_rate": float(np.mean(data["detection_rate"][mask])),
            "fp_rate": float(np.mean(data["fp_rate"][mask])),
            "fn_rate": float(np.mean(data["fn_rate"][mask])),
        }
    return out


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    npz_path = out_dir / "experiment_data.npz"
    if not npz_path.is_file():
        sys.exit(f"Missing {npz_path} — run python run_experiment.py first.")

    data = np.load(npz_path, allow_pickle=True)
    data = {k: data[k] for k in data.files}
    by_ratio = aggregate_by_ratio(data)

    # --- Markdown table ---
    print("\n## Median detection latency by asymmetry ratio (aggregated across N×f×seeds)\n")
    print("| r = α_neg / α_pos | min | median | mean | max | det rate | FP rate |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r, m in by_ratio.items():
        L = m["median_detection_latency"]
        print(
            f"| {r:.1f} | {L['min']:.1f} | {L['median']:.1f} | "
            f"{L['mean']:.1f} | {L['max']:.1f} | "
            f"{m['detection_rate']:.3f} | {m['fp_rate']:.3f} |"
        )

    # --- Figure 1: latency vs ratio at N=100, f=0.20 ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mask = (data["n"] == 100) & (np.isclose(data["f"], 0.20))
    schedules = data["schedule"][mask]
    ratios = data["r"][mask]
    latencies = data["median_detection_latency"][mask]
    # Sort by ratio for line plot
    order = np.argsort(ratios)
    ax.plot(ratios[order], latencies[order], "o-", color="C0", linewidth=2, markersize=10)
    for r, lat, sched in zip(ratios, latencies, schedules):
        ax.annotate(str(sched), (r, lat), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel(r"Asymmetry ratio $r = \alpha_\mathrm{neg} / \alpha_\mathrm{pos}$")
    ax.set_ylabel("Median detection latency (rounds)")
    ax.set_title(f"Detection latency vs asymmetry ratio (N=100, f=0.20)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "detection_latency_vs_ratio.png", dpi=120)
    plt.close(fig)

    # --- Figure 2: per-round detection curve at N=100, f=0.20 ---
    # Re-run a representative seed per schedule to capture trajectories
    # (round_detection_rates is per-run, not aggregated in the .npz).
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, a_pos, a_neg in [
        ("symmetric_low",  0.05,  0.05),
        ("symmetric_med",  0.10,  0.10),
        ("asym_r2",        0.05,  0.10),
        ("asym_r3",        0.05,  0.15),
        ("asym_r4",        0.05,  0.20),
        ("asym_r8",        0.025, 0.20),
    ]:
        sw = Swarm(n=100, f_byzantine=0.20,
                   alpha_pos=a_pos, alpha_neg=a_neg, seed=0)
        m = measure_run(sw, n_rounds=300)
        ax.plot(m["round_detection_rates"], label=label, linewidth=1.5)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Byzantine detection rate")
    ax.set_title("Detection rate over rounds (N=100, f=0.20, single representative seed)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "round_detection_rate_N100_f020.png", dpi=120)
    plt.close(fig)

    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
