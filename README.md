# BFT UAV Swarm Validation

Reproducibility code accompanying the paper *"Byzantine-Fault-Tolerant
Hierarchical Navigation for GNSS-Denied UAV Swarms: Architecture,
Theoretical Analysis, and Evaluation Methodology"*.

This repository contains four standalone Monte Carlo simulation modules
that empirically validate distinct design choices and theoretical
predictions from the paper. Each module is self-contained, deterministic
given seed, and depends only on `numpy` + `matplotlib` (plus `scikit-learn`
for one module).

## Modules

| Module | Validates | Paper section |
|---|---|---|
| `asymmetric_decay/` | Asymmetric reputation decay design (α_neg ≥ 2·α_pos; saturation at r ≈ 3, plateau to r=4 and beyond) | §VII |
| `clamp_convergence/` | Linear-drain convergence bound for naïve-Byzantine exclusion | §VIII |
| `byzantine_scaling/` | Empirical Byzantine-fraction resilience past f=1/3 (PBFT bound) | §IX |
| `polarization_metric/` | Mahalanobis vs cosine for anisotropic spoof detection (standalone — included for follow-on work; not referenced in the published paper) | — |

All three referenced simulators implement the paper's **additive** reputation
update rule (Eqs. 1–3 of §IV-C): `R_j ← clamp(R_j ± α · R_k, 0, 1)` with
verifier reputation weight `R_k = 1` (non-gossip self-confidence). This
matches the architecture described in the paper directly; an earlier
multiplicative variant is preserved in the repository git history.

## Quick reproduction

For any one module:

```bash
cd <module-dir>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py    # generates results/
python analyze.py           # generates figures + tables
```

Total runtime per module: 1–15 minutes on a modern CPU. See each
module's `README.md` for parameter details and expected output.

## Layout (within each module)

```
<module>/
├── README.md              module-specific reproduction notes
├── requirements.txt       deps (numpy, matplotlib, [sklearn])
├── simulator.py           core simulator (~100–165 LOC, numpy-only)
├── run_experiment.py      parameter sweep entry point
├── analyze.py             generates figures + tables from results
└── results/
    ├── experiment_data.npz   archived experimental output
    ├── results.json          human-readable
    └── *.png                  paper figures
```

## License

MIT (see `LICENSE`). Use the code freely with attribution to the paper.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{kalynovskyi2026bft,
  title   = {Byzantine-Fault-Tolerant Hierarchical Navigation for GNSS-Denied UAV Swarms:
             Architecture, Theoretical Analysis, and Evaluation Methodology},
  author  = {Kalynovskyi, Oleksandr},
  journal = {arXiv preprint},
  year    = {2026}
}
```

(Update the BibTeX entry with the actual arXiv ID once available.)

## Status

Release v1.1 (May 2026) — three validation modules referenced in the
paper plus one standalone module for follow-on work. Simulators in v1.1
implement the paper's additive reputation rule directly (v1.0 used a
qualitatively-similar multiplicative variant; the change closes a
multi-perspective audit finding that the validation should match the
architecture exactly). All numerical claims in the paper are reproducible
from this code.
