"""Synthetic residual generator + decision-statistic comparison
for polarization-domain spoof detection.

Models honest and spoofed residual vectors as multivariate Gaussian
samples with a controllable covariance condition number (`kappa`).
Compares two anomaly-detection statistics:

  Mahalanobis: D = sqrt( (x - mu)^T Sigma^{-1} (x - mu) )
  Cosine:     C = | cos(x, mu_dir) |

The first is the maximum-likelihood test statistic under known
multivariate Gaussian; the second is scale-invariant and discards
covariance information.

Pure numpy. Only sklearn dependency is `roc_auc_score` for ROC
computation (it is the only sklearn import).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def build_covariance(d: int, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Random positive-definite covariance with condition number = kappa.

    Eigenvalues are log-spaced from 1 to kappa; orthogonal frame is sampled
    uniformly from the Haar measure via QR of a Gaussian matrix.
    """
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigs = np.logspace(0, np.log10(kappa), d)
    return Q @ np.diag(eigs) @ Q.T


def sample_residuals(
    d: int, kappa: float, mu_spoof_norm: float,
    n_honest: int = 2000, n_spoof: int = 200, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (honest_samples, spoof_samples, spoof_direction, Sigma)."""
    rng = np.random.default_rng(seed)
    Sigma = build_covariance(d, kappa, rng)
    honest = rng.multivariate_normal(np.zeros(d), Sigma, size=n_honest)
    mu_dir = rng.standard_normal(d)
    mu_dir /= np.linalg.norm(mu_dir)
    spoof = rng.multivariate_normal(mu_spoof_norm * mu_dir, Sigma, size=n_spoof)
    return honest, spoof, mu_dir, Sigma


def mahalanobis_score(
    samples: np.ndarray, mu: np.ndarray, Sigma_inv: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance — larger means more anomalous."""
    diff = samples - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, Sigma_inv, diff))


def cosine_anomaly(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """|cos(x, ref)| — larger means more aligned with the spoof direction."""
    ref_norm = reference / np.linalg.norm(reference)
    return np.abs(samples @ ref_norm / (np.linalg.norm(samples, axis=1) + 1e-12))


@dataclass
class CellResult:
    d: int
    kappa: float
    mu_spoof_norm: float
    auc_mahalanobis: float
    auc_cosine: float
    auc_gap: float


def evaluate_cell(d: int, kappa: float, mu_spoof_norm: float, seed: int = 0) -> CellResult:
    """Compute ROC AUC for both decision statistics in one (d, kappa, mu) cell."""
    honest, spoof, mu_dir, Sigma = sample_residuals(d, kappa, mu_spoof_norm, seed=seed)
    Sigma_inv = np.linalg.inv(Sigma)
    mu_zero = np.zeros(d)

    D_h = mahalanobis_score(honest, mu_zero, Sigma_inv)
    D_s = mahalanobis_score(spoof, mu_zero, Sigma_inv)
    C_h = cosine_anomaly(honest, mu_dir)
    C_s = cosine_anomaly(spoof, mu_dir)

    y = np.concatenate([np.zeros(len(D_h)), np.ones(len(D_s))])
    return CellResult(
        d=d, kappa=kappa, mu_spoof_norm=mu_spoof_norm,
        auc_mahalanobis=float(roc_auc_score(y, np.concatenate([D_h, D_s]))),
        auc_cosine=float(roc_auc_score(y, np.concatenate([C_h, C_s]))),
        auc_gap=float(roc_auc_score(y, np.concatenate([D_h, D_s]))
                      - roc_auc_score(y, np.concatenate([C_h, C_s]))),
    )
