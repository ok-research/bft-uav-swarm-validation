"""Trajectory-tracking simulator for the clamped aggregate reputation
convergence experiment.

Reuses the multiplicative reputation update from the asymmetric_decay
simulator but tracks the FULL trajectory R_j(t) (rather than only the
first isolation crossing) so that convergence rate, monotonicity, and
inter-observer variance can be measured.

Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Swarm:
    n: int
    f_byzantine: float
    alpha_pos: float
    alpha_neg: float
    initial_r: float = 0.5
    p_obs_error: float = 0.05
    seed: int = 0

    is_byzantine: np.ndarray = field(init=False)
    reputation: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_byz = int(round(self.n * self.f_byzantine))
        self.is_byzantine = np.zeros(self.n, dtype=bool)
        self.is_byzantine[rng.choice(self.n, n_byz, replace=False)] = True
        self.reputation = np.full((self.n, self.n), self.initial_r, dtype=float)
        np.fill_diagonal(self.reputation, 1.0)

    def n_obs_per_round(self) -> int:
        return max(5, self.n // 10)

    def observe_round(self, rng: np.random.Generator) -> None:
        obs_size = self.n_obs_per_round()
        all_idx = np.arange(self.n)
        for i in range(self.n):
            if self.is_byzantine[i]:
                continue
            candidates = all_idx[all_idx != i]
            peers = rng.choice(candidates, size=min(obs_size, len(candidates)), replace=False)
            for j in peers:
                truth_byz = bool(self.is_byzantine[j])
                obs_byz = truth_byz != (rng.random() < self.p_obs_error)
                if obs_byz:
                    self.reputation[i, j] *= 1.0 - self.alpha_neg
                else:
                    self.reputation[i, j] += self.alpha_pos * (1.0 - self.reputation[i, j])


def simulate_trajectories(
    n: int, f_byzantine: float, alpha_pos: float, alpha_neg: float,
    p_obs_error: float = 0.05, n_rounds: int = 250, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trace[T, N, N], is_byzantine[N])."""
    sw = Swarm(
        n=n, f_byzantine=f_byzantine,
        alpha_pos=alpha_pos, alpha_neg=alpha_neg,
        p_obs_error=p_obs_error, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    trace = np.empty((n_rounds, n, n), dtype=float)
    for t in range(n_rounds):
        sw.observe_round(rng)
        trace[t] = sw.reputation
    return trace, sw.is_byzantine


def aggregate_R_j(trace: np.ndarray, j: int, honest_mask: np.ndarray) -> np.ndarray:
    """Median R_j across honest observers (excluding j itself) per round."""
    honest_idx = np.where(honest_mask)[0]
    honest_idx = honest_idx[honest_idx != j]
    return np.median(trace[:, honest_idx, j], axis=1)


@dataclass
class CellResult:
    alpha_neg: float
    p_obs_err: float
    median_T_eps: dict[float, float]      # eps -> rounds (n_rounds + 1 if not reached, treated as sentinel)
    mean_monotonicity_violations: float
    R_j_variance_at_end: float
    theory_ratio: dict[float, float]      # T_emp / T_theory per eps (NaN if non-convergent)
    is_converged: dict[float, bool]       # eps -> True iff median_T_eps < n_rounds (real convergence)


def measure_cell(
    n: int, f_byzantine: float, alpha_pos: float, alpha_neg: float,
    p_obs_err: float, eps_grid: list[float], n_rounds: int, seeds: list[int],
) -> CellResult:
    """Measure convergence behaviour across multiple seeds for one cell."""
    T_eps_per_seed = {eps: [] for eps in eps_grid}
    mono_per_seed: list[float] = []
    var_per_seed: list[float] = []

    for s in seeds:
        trace, is_byz = simulate_trajectories(
            n, f_byzantine, alpha_pos, alpha_neg, p_obs_err, n_rounds, seed=s,
        )
        honest_mask = ~is_byz
        byz_idx = np.where(is_byz)[0]

        per_byz_T_eps = {eps: [] for eps in eps_grid}
        mono_violations: list[int] = []
        end_var: list[float] = []
        for j in byz_idx:
            R_traj = aggregate_R_j(trace, j, honest_mask)
            for eps in eps_grid:
                below = np.where(R_traj < eps)[0]
                per_byz_T_eps[eps].append(int(below[0]) if len(below) else n_rounds + 1)
            diffs = np.diff(R_traj)
            first_drop = int(np.argmax(diffs < 0)) if (diffs < 0).any() else 0
            mono_violations.append(int((diffs[first_drop:] > 1e-6).sum()))

            honest_idx = np.where(honest_mask)[0]
            honest_idx = honest_idx[honest_idx != j]
            end_var.append(float(np.var(trace[-1, honest_idx, j])))

        for eps in eps_grid:
            T_eps_per_seed[eps].append(float(np.median(per_byz_T_eps[eps])))
        mono_per_seed.append(float(np.mean(mono_violations)))
        var_per_seed.append(float(np.mean(end_var)))

    median_T_eps = {eps: float(np.median(T_eps_per_seed[eps])) for eps in eps_grid}

    # A cell is "convergent" iff the median across seeds reaches the eps
    # threshold within n_rounds. Cells where median_T_eps == n_rounds + 1
    # (sentinel for non-reach) are flagged non-convergent and excluded from
    # theory-ratio averaging — including them produces a spurious "near 1"
    # ratio when the sentinel just happens to be close to T_theory.
    is_converged = {eps: median_T_eps[eps] < n_rounds for eps in eps_grid}

    # Theory: T ≈ log(R_init / eps) / (-log(1 - alpha_neg)) / f_obs
    # Reported only for cells where the simulation actually converged;
    # NaN otherwise so downstream filtering is straightforward.
    f_obs = max(5, n // 10) / n
    theory_ratio = {}
    for eps in eps_grid:
        if not is_converged[eps] or not (0 < alpha_neg < 1):
            theory_ratio[eps] = float("nan")
            continue
        T_theory = math.log(0.5 / eps) / (-math.log(1 - alpha_neg)) / f_obs
        theory_ratio[eps] = (
            median_T_eps[eps] / T_theory if T_theory > 0 else float("nan")
        )

    return CellResult(
        alpha_neg=alpha_neg, p_obs_err=p_obs_err,
        median_T_eps=median_T_eps,
        mean_monotonicity_violations=float(np.mean(mono_per_seed)),
        R_j_variance_at_end=float(np.mean(var_per_seed)),
        theory_ratio=theory_ratio,
        is_converged=is_converged,
    )
