"""Discrete-time UAV-swarm consensus simulator with reputation tracking.

Pure numpy implementation of the asymmetric additive reputation
update rule (matches the paper's Eqs. 1-3, §IV-C):
    VERIFIED:    R_ij ← clamp(R_ij + alpha_pos · R_k, 0, 1)
    UNVERIFIED:  R_ij ← clamp(R_ij - alpha_neg · R_k, 0, 1)
where R_k is the verifier's reputation weight; in this simulator all
observers are honest and operate in non-gossip mode with self-confidence
R_k = 1.0 (the deployed case). The analytical bound of Proposition 3
uses R_bar_honest = 0.53 as a conservative lower estimate, so empirical
latency under R_k=1.0 should beat the analytical prediction.

A UAV j is considered isolated when the median of {R_ij} across
honest observers i (excluding j itself) drops below the isolation
threshold theta. Observer noise is modelled as an independent
per-event misclassification probability p_obs_error.

See the parent paper, Section "Asymmetric Reputation System", for the
analytical motivation of this update family; this simulator empirically
characterises detection latency as a function of the asymmetry ratio
r = alpha_neg / alpha_pos.

No external dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Swarm:
    """Reputation state of an N-UAV swarm with a Byzantine subset.

    Reputation is an N x N matrix R[i, j] = i's belief about j.
    Diagonal is fixed at 1.0 (self-trust). Byzantine UAVs do not run
    the honest tracking logic in this model; they are passive targets
    of observation by honest UAVs.
    """

    n: int                                  # swarm size
    f_byzantine: float                      # fraction of Byzantine UAVs
    alpha_pos: float                        # positive-event coefficient
    alpha_neg: float                        # negative-event coefficient
    initial_r: float = 0.5                  # neutral prior
    isolation_threshold: float = 0.20       # theta — median R_j below this => isolated
    p_obs_error: float = 0.05               # observer noise
    seed: int = 0

    r_k_verifier: float = 1.0               # honest verifier reputation weight (non-gossip)

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
        """Observations per honest UAV per round.

        Scaled with swarm size so per-peer observation frequency stays
        approximately constant as N grows; floor at 5 to maintain
        statistical power at small N.
        """
        return max(5, self.n // 10)

    def observe_round(self, rng: np.random.Generator) -> None:
        """Each honest UAV observes a random subset of peers and updates reputation."""
        obs_size = self.n_obs_per_round()
        candidate_pool = np.arange(self.n)
        for i in range(self.n):
            if self.is_byzantine[i]:
                continue
            candidates = candidate_pool[candidate_pool != i]
            peers = rng.choice(candidates, size=min(obs_size, len(candidates)), replace=False)
            for j in peers:
                true_event_byzantine = bool(self.is_byzantine[j])
                # Observer noise: flip event with probability p_obs_error
                observed_byzantine = true_event_byzantine != (rng.random() < self.p_obs_error)
                if observed_byzantine:
                    self.reputation[i, j] = max(
                        0.0, self.reputation[i, j] - self.alpha_neg * self.r_k_verifier
                    )
                else:
                    self.reputation[i, j] = min(
                        1.0, self.reputation[i, j] + self.alpha_pos * self.r_k_verifier
                    )

    def swarm_belief(self, j: int) -> float:
        """Median R_j across honest observers (excluding j itself)."""
        honest_idx = np.where(~self.is_byzantine)[0]
        honest_idx = honest_idx[honest_idx != j]
        if len(honest_idx) == 0:
            return self.initial_r
        return float(np.median(self.reputation[honest_idx, j]))


def measure_run(swarm: Swarm, n_rounds: int = 300) -> dict:
    """Run the simulator for n_rounds and return metrics.

    Metrics:
        median_detection_latency: median round at which true-Byzantine
            UAVs cross the isolation threshold (n_rounds if not crossed).
        detection_rate: fraction of Byzantine UAVs eventually isolated.
        fp_rate: fraction of honest UAVs incorrectly isolated.
        fn_rate: 1 - detection_rate.
        round_detection_rates: per-round cumulative Byzantine isolation rate.
    """
    rng = np.random.default_rng(swarm.seed + 1)

    byzantine_idx = np.where(swarm.is_byzantine)[0]
    honest_idx = np.where(~swarm.is_byzantine)[0]

    detection_round = {int(j): -1 for j in byzantine_idx}
    fp_round = {int(j): -1 for j in honest_idx}
    round_detection_rates: list[float] = []

    for t in range(n_rounds):
        swarm.observe_round(rng)
        beliefs = np.array([swarm.swarm_belief(j) for j in range(swarm.n)])
        isolated = beliefs < swarm.isolation_threshold

        for j in byzantine_idx:
            if isolated[j] and detection_round[int(j)] == -1:
                detection_round[int(j)] = t
        for j in honest_idx:
            if isolated[j] and fp_round[int(j)] == -1:
                fp_round[int(j)] = t

        detected_so_far = sum(1 for r in detection_round.values() if r >= 0)
        round_detection_rates.append(detected_so_far / max(len(byzantine_idx), 1))

    detected_rounds = [r for r in detection_round.values() if r >= 0]
    return {
        "median_detection_latency": (
            float(np.median(detected_rounds)) if detected_rounds else float(n_rounds)
        ),
        "detection_rate": len(detected_rounds) / max(len(byzantine_idx), 1),
        "fp_rate": sum(1 for r in fp_round.values() if r >= 0) / max(len(honest_idx), 1),
        "fn_rate": 1.0 - len(detected_rounds) / max(len(byzantine_idx), 1),
        "round_detection_rates": round_detection_rates,
    }
