"""Parameter sweep for the clamp-convergence experiment.

Sweeps alpha_neg × p_obs_err (4 × 4 = 16 cells) with N=100 UAVs, f=0.20,
and 3 random seeds per cell.

Usage:
    python run_experiment.py    # ~5-8 minutes on a modern CPU
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import measure_cell

N_UAVS = 100
F_BYZANTINE = 0.20
N_ROUNDS = 250
ALPHA_POS = 0.05
ALPHA_NEG_GRID = [0.05, 0.10, 0.20, 0.40]
P_OBS_ERR_GRID = [0.00, 0.05, 0.10, 0.20]
EPS_GRID = [0.05, 0.01, 0.001]
SEEDS = list(range(25))


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for alpha_neg in ALPHA_NEG_GRID:
        for p_obs in P_OBS_ERR_GRID:
            print(f"alpha_neg={alpha_neg}, p_obs_err={p_obs} ...", flush=True)
            cell = measure_cell(
                N_UAVS, F_BYZANTINE, ALPHA_POS, alpha_neg, p_obs,
                EPS_GRID, N_ROUNDS, SEEDS,
            )
            rows.append({
                "alpha_pos": ALPHA_POS,
                "alpha_neg": cell.alpha_neg,
                "p_obs_err": cell.p_obs_err,
                "median_T_eps_005": cell.median_T_eps[0.05],
                "median_T_eps_001": cell.median_T_eps[0.01],
                "median_T_eps_0001": cell.median_T_eps[0.001],
                "theory_ratio_005": cell.theory_ratio[0.05],
                "theory_ratio_001": cell.theory_ratio[0.01],
                "theory_ratio_0001": cell.theory_ratio[0.001],
                "is_converged_005": cell.is_converged[0.05],
                "is_converged_001": cell.is_converged[0.01],
                "is_converged_0001": cell.is_converged[0.001],
                "monotonicity_violations": cell.mean_monotonicity_violations,
                "R_j_variance_at_end": cell.R_j_variance_at_end,
            })

    with open(out_dir / "results.json", "w") as fp:
        json.dump(rows, fp, indent=2)
    arrays = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(out_dir / "experiment_data.npz", **arrays)

    print("\n## Median T_eps (eps=0.01)\n")
    print("| alpha_neg | p_obs_err | T_0.01 | theory_ratio_0.01 | converged | mono_viol |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["is_converged_001"]:
            ratio_str = f"{r['theory_ratio_001']:.3f}"
        else:
            ratio_str = "  --  "
        print(f"| {r['alpha_neg']} | {r['p_obs_err']} | "
              f"{r['median_T_eps_001']:.1f} | {ratio_str} | "
              f"{'yes' if r['is_converged_001'] else 'no':>3} | "
              f"{r['monotonicity_violations']:.1f} |")

    # Headline statistic — only over actually-convergent cells (those where
    # the simulation reached eps within n_rounds; otherwise the theory_ratio
    # is NaN and would skew the average).
    convergent_ratios = [
        r["theory_ratio_001"] for r in rows if r["is_converged_001"]
    ]
    n_total = len(rows)
    n_conv = len(convergent_ratios)
    mean_ratio = float(np.mean(convergent_ratios)) if convergent_ratios else float("nan")
    print(f"\nConvergent cells: {n_conv} / {n_total}")
    print(f"Mean theory_ratio over convergent cells (eps=0.01): {mean_ratio:.3f}")


if __name__ == "__main__":
    main()
