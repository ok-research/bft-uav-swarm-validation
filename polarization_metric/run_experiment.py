"""Parameter sweep for the polarization-metric comparison experiment.

Sweeps (kappa, mu_spoof_norm, d) and produces AUC for Mahalanobis vs cosine.

Usage:
    python run_experiment.py    # ~1-2 minutes on a modern CPU
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import evaluate_cell

KAPPA_GRID = [1, 5, 25, 100, 500]
DIM_GRID = [4, 8, 16]
MU_SPOOF_NORM_GRID = [0.5, 1.0, 2.0, 5.0]
SEED = 0


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for d in DIM_GRID:
        for kappa in KAPPA_GRID:
            for mu in MU_SPOOF_NORM_GRID:
                r = evaluate_cell(d, kappa, mu, seed=SEED)
                rows.append(r.__dict__)
                print(
                    f"d={d:<3} kappa={kappa:<5} mu={mu:<5.2f}  "
                    f"AUC_M={r.auc_mahalanobis:.4f}  AUC_C={r.auc_cosine:.4f}  "
                    f"gap={r.auc_gap:+.4f}"
                )

    with open(out_dir / "results.json", "w") as fp:
        json.dump(rows, fp, indent=2)
    arrays = {
        k: np.array([row[k] for row in rows])
        for k in ("d", "kappa", "mu_spoof_norm",
                  "auc_mahalanobis", "auc_cosine", "auc_gap")
    }
    np.savez_compressed(out_dir / "experiment_data.npz", **arrays)
    print(f"\nWrote {out_dir}/  ({len(rows)} cells)")


if __name__ == "__main__":
    main()
