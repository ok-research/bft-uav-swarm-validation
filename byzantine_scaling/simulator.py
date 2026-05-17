"""Simulator for the Byzantine-fraction resilience-scaling experiment.

Uses the asymmetric reputation-decay isolation mechanism (alpha_neg = 4 * alpha_pos
with alpha_pos = 0.05 — the saturation-point schedule identified in the
asymmetric_decay experiment) and measures detection rate, false-positive
rate, latency, and final honest-sub-swarm reputation health as the
Byzantine fraction f varies across the theoretical 1/3 boundary.

Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunResult:
    f: float
    seed: int
    detection_rate: float
    fp_rate: float
    median_detection_latency: float
    final_honest_median_R: float
    isolation_curve: list[float]


def simulate(
    n: int, f_byzantine: float,
    alpha_pos: float = 0.05, alpha_neg: float = 0.20,
    isolation_threshold: float = 0.20, p_obs_error: float = 0.05,
    n_rounds: int = 300, seed: int = 0,
) -> RunResult:
    rng_init = np.random.default_rng(seed)
    n_byz = max(1, int(round(n * f_byzantine)))
    is_byzantine = np.zeros(n, dtype=bool)
    is_byzantine[rng_init.choice(n, n_byz, replace=False)] = True

    R = np.full((n, n), 0.5, dtype=float)
    np.fill_diagonal(R, 1.0)

    isolated_byz: dict[int, int] = {int(j): -1 for j in np.where(is_byzantine)[0]}
    isolated_honest: dict[int, int] = {int(j): -1 for j in np.where(~is_byzantine)[0]}
    isolation_curve: list[float] = []
    honest_idx_all = np.where(~is_byzantine)[0]

    rng = np.random.default_rng(seed + 1)
    n_obs = max(5, n // 10)
    candidate_pool = np.arange(n)
    for t in range(n_rounds):
        for i in range(n):
            if is_byzantine[i]:
                continue
            candidates = candidate_pool[candidate_pool != i]
            peers = rng.choice(candidates, size=min(n_obs, len(candidates)), replace=False)
            for j in peers:
                truth_byz = bool(is_byzantine[j])
                obs_byz = truth_byz != (rng.random() < p_obs_error)
                if obs_byz:
                    R[i, j] *= 1.0 - alpha_neg
                else:
                    R[i, j] += alpha_pos * (1.0 - R[i, j])

        for j in range(n):
            if is_byzantine[j] and isolated_byz[j] >= 0:
                continue
            if (not is_byzantine[j]) and isolated_honest[j] >= 0:
                continue
            observers = honest_idx_all[honest_idx_all != j]
            if len(observers) == 0:
                continue
            if float(np.median(R[observers, j])) < isolation_threshold:
                if is_byzantine[j]:
                    isolated_byz[j] = t
                else:
                    isolated_honest[j] = t

        isolated_so_far = sum(1 for v in isolated_byz.values() if v >= 0)
        isolation_curve.append(isolated_so_far / max(len(isolated_byz), 1))

    # Final honest sub-swarm consensus health: median pairwise honest R
    honest_R = R[np.ix_(honest_idx_all, honest_idx_all)].copy()
    np.fill_diagonal(honest_R, np.nan)
    final_honest = float(np.nanmedian(honest_R))

    detected = [v for v in isolated_byz.values() if v >= 0]
    fp_count = sum(1 for v in isolated_honest.values() if v >= 0)
    return RunResult(
        f=f_byzantine, seed=seed,
        detection_rate=len(detected) / max(len(isolated_byz), 1),
        fp_rate=fp_count / max(len(isolated_honest), 1),
        median_detection_latency=float(np.median(detected)) if detected else float("inf"),
        final_honest_median_R=final_honest,
        isolation_curve=isolation_curve,
    )
