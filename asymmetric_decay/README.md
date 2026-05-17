# Asymmetric Reputation Decay — Monte Carlo Validation

Reproducibility artefact for Section "Empirical Validation of the Asymmetric
Reputation Design" of the parent paper.

## What this is

A self-contained Python simulator that empirically characterises the
detection latency of a reputation-based Byzantine-isolation mechanism in
a discrete-time UAV swarm consensus protocol, as the asymmetry ratio
`r = alpha_neg / alpha_pos` is varied.

Pure `numpy` + `matplotlib`. No external services, no network calls,
no GPU. Deterministic given seed.

## File layout

```
asymmetric_decay/
├── README.md              this file
├── requirements.txt       deps: numpy, matplotlib
├── simulator.py           core Swarm + measure_run (~140 LOC)
├── run_experiment.py      parameter-sweep entry (~110 LOC)
├── analyze.py             tables + figures (~110 LOC)
└── results/
    ├── experiment_data.npz   archived sweep output (135 conditions)
    ├── results.json          same data, human-readable
    ├── detection_latency_vs_ratio.png
    └── round_detection_rate_N100_f020.png
```

## Reproduce

```bash
# Set up
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run sweep (~10-15 min on a modern CPU)
python run_experiment.py

# Regenerate paper figures + tables
python analyze.py
```

The pre-archived `results/experiment_data.npz` was produced with seeds
`{0, 1, 2}`, swarm sizes `{50, 100, 200}`, Byzantine fractions
`{0.10, 0.20, 0.33}`, and the five schedules defined in `run_experiment.py`.
Any reproduction using the same parameters should produce numerically
identical output (deterministic numpy RNG).

## What the simulator models

- N UAVs, `f` fraction Byzantine (selected uniformly at random per seed)
- Each round, every honest UAV observes `max(5, N/10)` random peers
- Per-event observer noise `p_obs_error = 0.05` (misclassification probability)
- Reputation update (multiplicative form):
  - honest event:    `R_ij ← R_ij + alpha_pos · (1 - R_ij)`
  - byzantine event: `R_ij ← R_ij · (1 - alpha_neg)`
- A UAV `j` is "isolated" when the median of `{R_ij}` across honest
  observers `i ≠ j` drops below `theta = 0.20`
- Metrics: detection rate, FP rate, FN rate, median detection latency

## Mapping to the parent paper

The multiplicative update used here is a member of the asymmetric-decay
family discussed in the parent paper's "Asymmetric Reputation System"
subsection; the paper's additive rule (`R ← R + alpha · R_k`) is a
separate member of the same family with similar qualitative behaviour.
The empirical conclusion about the saturation point `r ≈ 4` is expected
to transfer; a confirmatory run on the exact additive rule is listed as
future work in the paper's "Limitations" subsection.

