"""Produce paper figures for the clamp-convergence experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = Path(__file__).parent / "results"
    npz_path = out_dir / "experiment_data.npz"
    if not npz_path.is_file():
        sys.exit(f"Missing {npz_path} — run python run_experiment.py first.")

    data = {k: np.load(npz_path, allow_pickle=True)[k] for k in np.load(npz_path).files}

    ALPHA_NEG = sorted(set(data["alpha_neg"].tolist()))
    P_OBS = sorted(set(data["p_obs_err"].tolist()))

    # Figure 1: T_{0.01} vs alpha_neg, faceted by p_obs_err
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in P_OBS:
        mask = np.isclose(data["p_obs_err"], p)
        xs = sorted(set(data["alpha_neg"][mask].tolist()))
        ys = [float(np.mean(data["median_T_eps_001"][
            (data["alpha_neg"] == a) & np.isclose(data["p_obs_err"], p)
        ])) for a in xs]
        ax.plot(xs, ys, marker="o", label=f"p_obs_err={p}")
    ax.set_xlabel(r"$\alpha_\mathrm{neg}$")
    ax.set_ylabel(r"Median $T_{0.01}$ (rounds to $\bar{R}_j < 0.01$)")
    ax.set_title("Convergence time vs decay coefficient and observer noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "T_eps_vs_alpha_neg.png", dpi=120)
    plt.close(fig)

    # Figure 2: theory ratio vs alpha_neg, faceted by p_obs_err.
    # Under the paper's additive rule with R_k=1, all 16 cells converge
    # within n_rounds. Any non-convergent cells (none expected) would be
    # excluded from the plot since their ratio is undefined.
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in P_OBS:
        xs, ys = [], []
        for a in sorted(set(data["alpha_neg"][np.isclose(data["p_obs_err"], p)].tolist())):
            cell_mask = (data["alpha_neg"] == a) & np.isclose(data["p_obs_err"], p)
            converged = data["is_converged_001"][cell_mask]
            ratios = data["theory_ratio_001"][cell_mask]
            # Only include points where the cell actually converged
            valid_ratios = ratios[converged.astype(bool)]
            if len(valid_ratios) > 0:
                xs.append(a)
                ys.append(float(np.mean(valid_ratios)))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"p_obs_err={p}")
    ax.axhline(1.0, color="black", lw=0.5, ls="--", label="theory = empirical")
    ax.set_xlabel(r"$\alpha_\mathrm{neg}$")
    ax.set_ylabel(r"$T_\mathrm{empirical} / T_\mathrm{theory}$ (eps=0.01)")
    ax.set_title("Empirical-to-theoretical convergence-time ratio\n(all 16 cells; additive rule, R_k=1)")
    ax.set_ylim(0, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "theory_vs_empirical_ratio.png", dpi=120)
    plt.close(fig)

    # Figure 3: monotonicity violations vs p_obs_err, faceted by alpha_neg
    fig, ax = plt.subplots(figsize=(8, 5))
    for a in ALPHA_NEG:
        mask = data["alpha_neg"] == a
        xs = sorted(set(data["p_obs_err"][mask].tolist()))
        ys = [float(np.mean(data["monotonicity_violations"][
            (data["alpha_neg"] == a) & np.isclose(data["p_obs_err"], p)
        ])) for p in xs]
        ax.plot(xs, ys, marker="o", label=fr"$\alpha_\mathrm{{neg}}$={a}")
    ax.set_xlabel(r"Observer noise probability $p_\mathrm{obs\_err}$")
    ax.set_ylabel(r"Mean monotonicity violations per Byzantine UAV")
    ax.set_title("Trajectory monotonicity under observer noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "monotonicity_vs_noise.png", dpi=120)
    plt.close(fig)

    print(f"Figures written to {out_dir}/")


if __name__ == "__main__":
    main()
