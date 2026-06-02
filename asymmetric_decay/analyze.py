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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# IEEE-style global rcParams: Times-like serif, axis-cleanup, hi-dpi
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

# Okabe-Ito palette (colourblind-safe)
OK_BLUE   = "#0072B2"
OK_ORANGE = "#E69F00"
OK_GREEN  = "#009E73"
OK_RED    = "#D55E00"
OK_PURPLE = "#CC79A7"
OK_CYAN   = "#56B4E9"


def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="0.8")
    ax.set_axisbelow(True)

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
    # Restructured per ТЗ: line only through alpha_pos=0.05 family
    # (symmetric_low, asym_r2, asym_r3, asym_r4); symmetric_med (alpha_pos=0.10)
    # and asym_r8 (alpha_pos=0.025) plotted as separate markers off-line.
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    mask = (data["n"] == 100) & (np.isclose(data["f"], 0.20))
    schedules = data["schedule"][mask]
    ratios = data["r"][mask]
    latencies = data["median_detection_latency"][mask]

    # Separate alpha_pos=0.05 family (line) from off-family points (markers)
    in_line = np.array([s in ("symmetric_low", "asym_r2", "asym_r3", "asym_r4")
                        for s in schedules])
    line_r   = ratios[in_line]
    line_lat = latencies[in_line]
    line_order = np.argsort(line_r)
    ax.plot(line_r[line_order], line_lat[line_order],
            marker="o", color=OK_BLUE, linewidth=1.0, markersize=5,
            label=r"$\alpha_\mathrm{pos}=0.05$ family")

    for sched, m_color, m_shape, m_label in [
        ("symmetric_med", OK_ORANGE, "s", r"symmetric_med ($\alpha_\mathrm{pos}=0.10$)"),
        ("asym_r8",       OK_GREEN,  "D", r"asym_r8 ($\alpha_\mathrm{pos}=0.025$)"),
    ]:
        sel = schedules == sched
        if sel.any():
            ax.plot(ratios[sel], latencies[sel], marker=m_shape,
                    linestyle="", color=m_color, markersize=5, label=m_label)

    # Annotate design-recommended r=3 point
    r3_lat = float(latencies[schedules == "asym_r3"][0]) if (schedules == "asym_r3").any() else None
    if r3_lat is not None:
        ax.annotate("design point\n($r=3$)", xy=(3, r3_lat),
                    xytext=(4.5, 35), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6))

    ax.set_xlabel(r"Asymmetry ratio $r = \alpha_\mathrm{neg} / \alpha_\mathrm{pos}$")
    ax.set_ylabel("Median detection latency (rounds)")
    ax.set_ylim(10, 78)
    ax.legend(loc="upper right")
    _clean_axes(ax)
    fig.savefig(out_dir / "detection_latency_vs_ratio.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-round detection curve at N=100, f=0.20 ---
    # Re-run a representative seed per schedule to capture trajectories
    # (round_detection_rates is per-run, not aggregated in the .npz).
    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    schedule_spec = [
        ("symmetric_low", 0.05,  0.05,  OK_BLUE,   "-"),
        ("symmetric_med", 0.10,  0.10,  OK_ORANGE, "-"),
        ("asym_r2",       0.05,  0.10,  OK_GREEN,  "-"),
        ("asym_r3",       0.05,  0.15,  OK_RED,    "-"),
        ("asym_r4",       0.05,  0.20,  OK_PURPLE, "-"),
        ("asym_r8",       0.025, 0.20,  OK_CYAN,   "--"),  # different ls vs r4
    ]
    for label, a_pos, a_neg, color, ls in schedule_spec:
        sw = Swarm(n=100, f_byzantine=0.20,
                   alpha_pos=a_pos, alpha_neg=a_neg, seed=0)
        m = measure_run(sw, n_rounds=300)
        ax.plot(m["round_detection_rates"], label=label, color=color,
                linestyle=ls, linewidth=1.2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative Byzantine detection rate")
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    _clean_axes(ax)
    fig.savefig(out_dir / "round_detection_rate_N100_f020.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
