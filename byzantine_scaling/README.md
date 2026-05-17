# Byzantine-Fraction Resilience Scaling — Monte Carlo Validation

Reproducibility artefact for the empirical-resilience section of the
parent paper.

## What this is

Characterises the empirical Byzantine resilience curve of the
reputation-based isolation mechanism as the Byzantine fraction f
varies from 0.05 to 0.45 — straddling the classical PBFT bound f = 1/3.
Uses the saturation-point asymmetric schedule (alpha_neg = 4 alpha_pos
with alpha_pos = 0.05) established in the asymmetric_decay experiment.

Pure `numpy` + `matplotlib`. Deterministic given seed.

## File layout

```
byzantine_scaling/
├── README.md
├── requirements.txt
├── simulator.py        single-run simulator
├── run_experiment.py   10-f sweep with 5 seeds per f
├── analyze.py          two-panel resilience curve figure
└── results/
    ├── experiment_data.npz
    ├── results.json
    └── byzantine_resilience_curve.png
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py    # ~3-5 minutes on a modern CPU
python analyze.py
```

## What the experiment measures

For each f in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45}
with N=100 and 5 random seeds:
- detection rate (fraction of true-Byzantine UAVs eventually isolated)
- false-positive rate (honest UAVs incorrectly isolated)
- median detection latency (rounds)
- final honest sub-swarm reputation health (median pairwise R)

Headline result: full Byzantine detection (100%, zero FP) is maintained
well beyond the classical PBFT bound f = 1/3, up to at least f = 0.45,
with detection latency remaining flat at ~48 rounds. The reputation
isolation mechanism filters Byzantine UAVs PRE-consensus, decoupling
the effective fault-tolerance budget from the classical algorithm-level
bound.

