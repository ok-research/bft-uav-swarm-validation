"""Parameter-sweep entry point for the asymmetric reputation decay experiment.

Sweeps the asymmetry ratio r = alpha_neg / alpha_pos across five schedules
(two symmetric baselines at r=1, three asymmetric at r=2,4,8) on a grid of
swarm sizes (N in {50, 100, 200}) and Byzantine fractions (f in {0.10, 0.20, 0.33})
with multiple random seeds. Outputs:

    results/experiment_data.npz   — full results array
    results/results.json          — human-readable table

Usage:
    python run_experiment.py

Compute: ~10-15 minutes on a modern CPU with default settings (3 sizes ×
3 fractions × 5 schedules × 3 seeds = 135 sims, ~5-10 seconds each).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import Swarm, measure_run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_ROUNDS = 300

SWARM_SIZES = [50, 100, 200]
BYZANTINE_FRACTIONS = [0.10, 0.20, 0.33]
SEEDS = [0, 1, 2]

# (label, alpha_pos, alpha_neg) — six schedules covering r ∈ {1, 2, 3, 4, 8}
SCHEDULES = [
    ("symmetric_low",  0.05,  0.05),    # r = 1 baseline at moderate alpha
    ("symmetric_med",  0.10,  0.10),    # r = 1 with larger alpha
    ("asym_r2",        0.05,  0.10),    # r = 2
    ("asym_r3",        0.05,  0.15),    # r = 3 — paper's specified default
    ("asym_r4",        0.05,  0.20),    # r = 4 — sweet spot
    ("asym_r8",        0.025, 0.20),    # r = 8 — aggressive
]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    for n in SWARM_SIZES:
        for f in BYZANTINE_FRACTIONS:
            for label, a_pos, a_neg in SCHEDULES:
                per_seed = []
                for seed in SEEDS:
                    sw = Swarm(
                        n=n, f_byzantine=f,
                        alpha_pos=a_pos, alpha_neg=a_neg,
                        seed=seed,
                    )
                    m = measure_run(sw, n_rounds=N_ROUNDS)
                    per_seed.append(m)
                aggregated = {
                    "n": n,
                    "f": f,
                    "schedule": label,
                    "alpha_pos": a_pos,
                    "alpha_neg": a_neg,
                    "r": a_neg / a_pos,
                    "detection_rate": float(np.mean([m["detection_rate"] for m in per_seed])),
                    "fp_rate": float(np.mean([m["fp_rate"] for m in per_seed])),
                    "fn_rate": float(np.mean([m["fn_rate"] for m in per_seed])),
                    "median_detection_latency": float(np.mean(
                        [m["median_detection_latency"] for m in per_seed]
                    )),
                }
                rows.append(aggregated)
                print(
                    f"N={n} f={f:.2f} {label:<14} r={aggregated['r']:.1f}  "
                    f"detect={aggregated['detection_rate']:.3f}  "
                    f"fp={aggregated['fp_rate']:.3f}  "
                    f"latency={aggregated['median_detection_latency']:.1f}"
                )

    # Save in two formats
    with open(out_dir / "results.json", "w") as fp:
        json.dump(rows, fp, indent=2)

    arrays = {
        k: np.array([row[k] for row in rows])
        for k in ("n", "f", "alpha_pos", "alpha_neg", "r",
                  "detection_rate", "fp_rate", "fn_rate", "median_detection_latency")
    }
    arrays["schedule"] = np.array([row["schedule"] for row in rows])
    np.savez_compressed(out_dir / "experiment_data.npz", **arrays)

    print(f"\nWrote {out_dir}/results.json and {out_dir}/experiment_data.npz")
    print(f"Total conditions: {len(rows)}")


if __name__ == "__main__":
    main()
