"""Byzantine-fraction sweep across the theoretical 1/3 boundary.

Sweeps f in {0.05, 0.10, ..., 0.45} with N=100 UAVs and 5 random seeds each.
Uses the saturation-point asymmetric schedule (alpha_neg = 4 * alpha_pos)
established in the asymmetric_decay experiment.

Usage:
    python run_experiment.py    # ~3-5 minutes on a modern CPU
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import simulate

N_UAVS = 100
ALPHA_POS = 0.05
ALPHA_NEG = 0.20    # r = 4
F_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50]
SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    per_run = []
    for f in F_GRID:
        for s in SEEDS:
            r = simulate(N_UAVS, f, ALPHA_POS, ALPHA_NEG, seed=s)
            per_run.append({
                "f": r.f, "seed": r.seed,
                "detection_rate": r.detection_rate,
                "fp_rate": r.fp_rate,
                "median_detection_latency": (
                    r.median_detection_latency
                    if np.isfinite(r.median_detection_latency) else 300.0
                ),
                "final_honest_median_R": r.final_honest_median_R,
            })

    aggregated = []
    print("\n## Per-f aggregated metrics (mean across 5 seeds)\n")
    print("| f | detection_rate | fp_rate | median_latency | final_honest_R |")
    print("|---:|---:|---:|---:|---:|")
    for f in F_GRID:
        cell = [r for r in per_run if r["f"] == f]
        agg = {
            "f": f,
            "detection_rate": float(np.mean([r["detection_rate"] for r in cell])),
            "fp_rate": float(np.mean([r["fp_rate"] for r in cell])),
            "median_detection_latency": float(np.median(
                [r["median_detection_latency"] for r in cell]
            )),
            "final_honest_median_R": float(np.mean(
                [r["final_honest_median_R"] for r in cell]
            )),
        }
        aggregated.append(agg)
        print(f"| {f} | {agg['detection_rate']:.3f} | {agg['fp_rate']:.3f} | "
              f"{agg['median_detection_latency']:.1f} | "
              f"{agg['final_honest_median_R']:.3f} |")

    with open(out_dir / "results.json", "w") as fp:
        json.dump({"per_run": per_run, "aggregated": aggregated}, fp, indent=2)

    arrays = {k: np.array([row[k] for row in aggregated]) for k in aggregated[0]}
    np.savez_compressed(out_dir / "experiment_data.npz", **arrays)
    print(f"\nWrote {out_dir}/  ({len(aggregated)} f values × {len(SEEDS)} seeds)")


if __name__ == "__main__":
    main()
