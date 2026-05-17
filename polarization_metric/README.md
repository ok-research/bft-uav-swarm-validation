# Polarization Metric Comparison — Monte Carlo Validation

Reproducibility artefact for Section "Empirical Comparison of Polarization-Domain
Decision Statistics" of the parent paper.

## What this is

Compares Mahalanobis distance against cosine similarity as decision
statistics for distinguishing honest from spoofed multivariate residual
vectors. Sweeps the covariance condition number `kappa`, the spoof
offset magnitude `mu_spoof_norm`, and the residual dimensionality `d`.

Pure `numpy` + `matplotlib` + `scikit-learn` (only `roc_auc_score`).
No external services, no network calls, no GPU. Deterministic given
seed.

## File layout

```
polarization_metric/
├── README.md
├── requirements.txt    deps: numpy, matplotlib, scikit-learn
├── simulator.py        synthetic residual generator + ROC-AUC computation
├── run_experiment.py   parameter sweep (60 cells)
├── analyze.py          tables + figures
└── results/
    ├── experiment_data.npz
    ├── results.json
    ├── auc_gap_vs_kappa.png
    └── auc_gap_heatmap.png
```

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py    # ~1-2 minutes on a modern CPU
python analyze.py           # regenerate paper figures + tables
```

## What the experiment models

- Honest residuals: 2000 samples from N(0, Sigma) with controllable Sigma
- Spoof residuals: 200 samples from N(mu_spoof_norm * mu_direction, Sigma)
- Sigma has condition number = `kappa` (log-spaced eigenvalues), random orthogonal frame
- Mahalanobis: D = sqrt((x - 0)^T Sigma^{-1} (x - 0))
- Cosine: C = |cos(x, mu_direction)|
- Metric: ROC AUC over (honest, spoof) labels for each statistic separately
- Reported: AUC gap = AUC(Mahalanobis) − AUC(cosine), positive ⇒ Mahalanobis better

