from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "production" / "analysis"))

from fss_joint_correlations import JOINT_REPLICAS, calibrate_and_derive


def synthetic_raw() -> np.ndarray:
    rng = np.random.default_rng(20260713)
    transform = np.array([[1.0, 0.0, 0.0], [-0.1, 1.0, 0.0], [0.05, 0.2, 1.0]])
    return rng.normal(size=(JOINT_REPLICAS, 3)) @ transform.T


def test_replica_by_replica_formulas_and_calibration() -> None:
    centers = np.array([0.98, 1.74, 0.124])
    sigmas = np.array([0.017, 0.0043, 0.0011])
    raw = synthetic_raw()
    calibrated, derived, summary = calibrate_and_derive(raw, centers, sigmas)
    nu, gamma_nu, beta_nu = calibrated.T
    expected = np.column_stack((
        gamma_nu * nu, beta_nu * nu, 2.0 - gamma_nu,
        1.0 + gamma_nu / beta_nu, gamma_nu + 2.0 * beta_nu,
    ))
    np.testing.assert_allclose(derived, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(calibrated.mean(axis=0), centers, atol=1e-14)
    np.testing.assert_allclose(calibrated.std(axis=0, ddof=1), sigmas, atol=1e-14)
    np.testing.assert_allclose(np.diag(summary["calibrated_covariance"]), sigmas**2, atol=1e-18)
    np.testing.assert_allclose(
        np.corrcoef(calibrated, rowvar=False), np.corrcoef(raw, rowvar=False), atol=1e-14
    )
    assert summary["positive_definite"]
    assert min(summary["covariance_eigenvalues"]) > 0.0


def test_hyperscaling_variance_identity() -> None:
    calibrated, derived, summary = calibrate_and_derive(
        synthetic_raw(), np.array([0.98, 1.74, 0.124]), np.array([0.017, 0.0043, 0.0011])
    )
    h = summary["hyperscaling_variance_decomposition"]
    assert abs(h["identity_residual"]) < 1e-15
    direct = np.var(derived[:, 4], ddof=1)
    covariance = np.cov(calibrated[:, 1], calibrated[:, 2], ddof=1)[0, 1]
    rhs = np.var(calibrated[:, 1], ddof=1) + 4 * np.var(calibrated[:, 2], ddof=1) + 4 * covariance
    np.testing.assert_allclose(direct, rhs, atol=1e-18)
