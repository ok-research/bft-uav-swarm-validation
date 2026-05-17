# Clamped-Aggregate Reputation Convergence — Monte Carlo Validation

Reproducibility artefact for the convergence-properties section of the
parent paper.

## What this is

Measures convergence of the aggregate clamped reputation R_j over time
for Byzantine UAVs in the reputation-based isolation mechanism, and
compares the empirical convergence rate to the analytical bound
T ≈ log(R_init/eps) / (-log(1 - alpha_neg)) / f_obs.

Pure `numpy` + `matplotlib`. Deterministic given seed.

## File layout

```
clamp_convergence/
├── README.md
├── requirements.txt
├── simulator.py        Swarm with full trajectory tracking
├── run_experiment.py   16-cell sweep (alpha_neg × p_obs_err)
├── analyze.py          three figures
└── results/
    ├── experiment_data.npz
    ├── results.json
    ├── T_eps_vs_alpha_neg.png
    ├── theory_vs_empirical_ratio.png
    └── monotonicity_vs_noise.png
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py    # ~5-8 minutes on a modern CPU
python analyze.py
```

## What the experiment measures

For each (alpha_neg, p_obs_err) cell:
- T_eps = first round at which the aggregate (median across honest
  observers) R_j drops below eps, for eps in {0.05, 0.01, 0.001}
- empirical-to-theoretical ratio
  T_emp / T_theory where T_theory = log(R_init/eps) / (-log(1-alpha_neg)) / f_obs
- monotonicity violations: rounds in which R_j increases after the
  trajectory has begun descending (a measure of observer-noise impact)
- end-state variance of R_j across observers

Headline result: the empirical-to-theoretical convergence-time ratio is
within tight bounds across the practical operating regime, validating
the analytical convergence prediction from the paper.

