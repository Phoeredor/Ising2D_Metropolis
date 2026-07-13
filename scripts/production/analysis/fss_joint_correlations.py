#!/usr/bin/env python3
"""Correlation-only block bootstrap for the production FSS analysis.

The synchronized bootstrap estimates only Corr(nu, gamma/nu, beta/nu).
Central values, marginal errors and fit-quality statistics remain those of the
official marginal estimators in :mod:`full_fss_analysis`.
"""
from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import approx_fprime, brentq, curve_fit, minimize

JOINT_VERSION = 1
CACHE_VERSION = 2
JOINT_SEED = 20260713
JOINT_REPLICAS = 500
LS = np.array([24, 32, 48, 64, 96, 128], dtype=int)
ROOT_WINDOW = (0.435, 0.445)
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])
HEADER_FORMAT = "ii d qqq II"
HEADER_SIZE = 48


@dataclass(frozen=True)
class JointResult:
    raw: np.ndarray
    valid: np.ndarray
    calibrated: np.ndarray
    derived: np.ndarray
    summary: dict[str, Any]


def _spline(x: np.ndarray, y: np.ndarray, error: np.ndarray) -> UnivariateSpline:
    return UnivariateSpline(x, y, w=1.0 / (error + 1e-12), k=3, s=len(x))


def _source_matches(run: dict[str, Any]) -> bool:
    source = Path(run["source"])
    if not source.is_file():
        return False
    stat = source.stat()
    return stat.st_size == int(run["source_size"]) and stat.st_mtime_ns == int(run["source_mtime_ns"])


def _cache_file_valid(cache_dir: Path, run: dict[str, Any]) -> bool:
    path = cache_dir / run["cache_file"]
    if not path.is_file():
        return False
    try:
        with np.load(path) as archive:
            blocks = archive["blocks"]
            return blocks.ndim == 2 and blocks.shape == (int(run["base_blocks"]), 5) and np.isfinite(blocks).all()
    except (OSError, ValueError, KeyError):
        return False


def _rebuild_run(run: dict[str, Any], target: Path) -> None:
    """Rebuild one invalidated run; called only after explicit CLI opt-in."""
    source = Path(run["source"])
    block_size = int(run["base_block_size"])
    with source.open("rb") as stream:
        header = stream.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise RuntimeError(f"short binary header: {source}")
    struct.unpack(HEADER_FORMAT, header)
    records = np.memmap(source, dtype=RECORD_DTYPE, mode="r", offset=HEADER_SIZE)
    n_blocks = len(records) // block_size
    if n_blocks != int(run["base_blocks"]):
        raise RuntimeError(f"block-count change for {source}: {n_blocks} != {run['base_blocks']}")
    usable = n_blocks * block_size
    blocks = np.column_stack([
        np.abs(records["m"][:usable]).reshape(n_blocks, block_size).mean(axis=1),
        records["m2"][:usable].reshape(n_blocks, block_size).mean(axis=1),
        records["m4"][:usable].reshape(n_blocks, block_size).mean(axis=1),
        records["e"][:usable].reshape(n_blocks, block_size).mean(axis=1),
        records["e2"][:usable].reshape(n_blocks, block_size).mean(axis=1),
    ])
    np.savez_compressed(target, blocks=blocks)


def ensure_official_cache(repo_root: Path, rebuild: bool = False) -> tuple[Path, Path, dict[str, Any]]:
    """Validate the official cache or initialize it from the validated recovery cache.

    Invalid runs are never read from raw silently.  With ``rebuild=False`` an
    invalid run is a hard error.  ``rebuild=True`` rebuilds only invalid runs.
    """
    official_dir = repo_root / "results" / "production" / "analysis" / "fss_joint_cache"
    official_manifest = official_dir / "manifest.json"
    legacy_dir = repo_root / "tmp" / "cache"
    legacy_manifest = repo_root / "tmp" / "results" / "cache_manifest.json"

    initialized = False
    if official_manifest.is_file():
        manifest = json.loads(official_manifest.read_text())
        if manifest.get("joint_cache_version") != JOINT_VERSION:
            raise RuntimeError("unsupported official joint-cache manifest version")
    else:
        if not legacy_manifest.is_file():
            raise RuntimeError("no official or compatible recovery cache manifest; raw rebuild is not implicit")
        source_manifest = json.loads(legacy_manifest.read_text())
        if source_manifest.get("cache_version") != CACHE_VERSION:
            raise RuntimeError("incompatible recovery cache version")
        official_dir.mkdir(parents=True, exist_ok=True)
        runs = source_manifest["runs"]
        for run in runs:
            if not _source_matches(run) or not _cache_file_valid(legacy_dir, run):
                raise RuntimeError(f"legacy cache is stale for {run['source_name']}")
            shutil.copy2(legacy_dir / run["cache_file"], official_dir / run["cache_file"])
        manifest = {
            "joint_cache_version": JOINT_VERSION,
            "block_cache_version": CACHE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "initialization": "copied from tmp/cache after per-run size, mtime and NPZ validation",
            "legacy_manifest": str(legacy_manifest.relative_to(repo_root)),
            "moments": source_manifest["moments"],
            "block_multiplier": 1,
            "runs": runs,
        }
        official_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        initialized = True

    invalid = [run for run in manifest["runs"] if not _source_matches(run) or not _cache_file_valid(official_dir, run)]
    if invalid and not rebuild:
        names = ", ".join(run["source_name"] for run in invalid[:5])
        raise RuntimeError(f"{len(invalid)} invalid joint-cache run(s): {names}; rerun with --rebuild-joint-cache")
    if invalid:
        for run in invalid:
            if not _source_matches(run):
                raise RuntimeError(f"raw source metadata changed: {run['source']}")
            _rebuild_run(run, official_dir / run["cache_file"])
        manifest["last_selective_rebuild_at"] = datetime.now(timezone.utc).isoformat()
        manifest["last_selective_rebuild_runs"] = [run["source_name"] for run in invalid]
        official_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    manifest["cache_status"] = "initialized" if initialized else "hit"
    manifest["validated_runs"] = len(manifest["runs"])
    return official_dir, official_manifest, manifest


def load_cached_runs(cache_dir: Path, manifest: dict[str, Any]) -> dict[int, list[tuple[float, np.ndarray]]]:
    runs: dict[int, list[tuple[float, np.ndarray]]] = {int(L): [] for L in LS}
    for run in manifest["runs"]:
        L = int(run["L"])
        if L not in runs:
            continue
        with np.load(cache_dir / run["cache_file"]) as archive:
            blocks = archive["blocks"].copy()
        runs[L].append((float(run["beta"]), blocks))
    for L in LS:
        runs[int(L)].sort(key=lambda item: item[0])
    return runs


def _fit_correlated_leading(L: np.ndarray, y: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    u, singular, vt = np.linalg.svd(covariance)
    keep = singular > 1e-6 * singular[0]
    inverse = (vt.T * np.where(keep, 1.0 / singular, 0.0)) @ u.T

    def objective(params: np.ndarray) -> float:
        residual = y - params[0] * L**params[1]
        return float(residual @ inverse @ residual)

    result = minimize(
        objective, [y[0] / L[0], 1.0], method="L-BFGS-B",
        bounds=[(0.0, None), (0.7, 1.5)], options={"ftol": 1e-12, "gtol": 1e-8},
    )
    # Preserve the validated recovery estimator: L-BFGS-B occasionally reports
    # ABNORMAL after reaching a finite usable optimum for a noisy replica.
    if not np.isfinite(result.x).all() or not np.isfinite(objective(result.x)):
        raise RuntimeError(f"joint replica nu fit produced a non-finite optimum: {result.message}")
    return result.x


def _peak_three(x: np.ndarray, y: np.ndarray) -> float:
    index = int(np.argmax(y))
    low = max(0, min(index - 1, len(x) - 3))
    xx = x[low:low + 3]
    yy = y[low:low + 3]
    origin = xx[1]
    coeff = np.linalg.solve(np.column_stack((np.ones(3), xx - origin, (xx - origin)**2)), yy)
    if coeff[2] >= 0:
        raise ValueError("non-concave three-point peak")
    offset = -coeff[1] / (2.0 * coeff[2])
    return float(coeff[0] + coeff[1] * offset + coeff[2] * offset**2)


def _power_exponent(L: np.ndarray, y: np.ndarray, error: np.ndarray, decreasing: bool) -> float:
    def model(size: np.ndarray, amplitude: float, exponent: float) -> np.ndarray:
        return amplitude * size**(-exponent if decreasing else exponent)

    bounds = ([0.8, 0.08], [1.2, 0.18]) if decreasing else ([0.05, 1.5], [0.2, 2.0])
    initial = [1.0, 0.125] if decreasing else [0.11, 1.75]
    params, _ = curve_fit(model, L, y, p0=initial, sigma=error, absolute_sigma=True, bounds=bounds)
    return float(params[1])


def _observables(mean: np.ndarray, L: int) -> tuple[float, float, float]:
    abs_m, m2, m4, _energy, _energy2 = mean
    return float(abs_m), float(L * L * (m2 - abs_m**2)), float(1.0 - m4 / (3.0 * m2**2))


def _one_replica(
    runs: dict[int, list[tuple[float, np.ndarray]]],
    data: dict[int, dict[str, np.ndarray]],
    slope_covariance: np.ndarray,
    crossing_errors: np.ndarray,
    chi_errors: np.ndarray,
    magnetization_errors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    curves: dict[int, np.ndarray] = {}
    for L in LS:
        rows = []
        for _beta, blocks in runs[int(L)]:
            indices = rng.integers(0, len(blocks), len(blocks))
            rows.append(_observables(blocks[indices].mean(axis=0), int(L)))
        curves[int(L)] = np.asarray(rows)

    splines: dict[int, UnivariateSpline] = {}
    for L in LS:
        d = data[int(L)]
        mask = (d["beta"] >= 0.43) & (d["beta"] <= 0.45)
        splines[int(L)] = _spline(d["beta"][mask], curves[int(L)][mask, 2], d["U4_err"][mask])
    crossings = np.array([
        brentq(lambda beta: splines[int(a)](beta) - splines[int(b)](beta), *ROOT_WINDOW)
        for a, b in zip(LS[:-1], LS[1:])
    ])
    weights = 1.0 / (crossing_errors**2 + 1e-12)
    beta_pc = float(weights @ crossings / weights.sum())

    slopes = np.array([abs(splines[int(L)].derivative()(beta_pc)) for L in LS])
    selected = LS >= 32
    nu_params = _fit_correlated_leading(
        LS[selected], slopes[selected], slope_covariance[np.ix_(selected, selected)]
    )
    nu = 1.0 / nu_params[1]

    chi_peaks = []
    magnetizations = []
    for L in LS:
        d = data[int(L)]
        chi_peaks.append(_peak_three(d["beta"], curves[int(L)][:, 1]))
        magnetizations.append(float(_spline(d["beta"], curves[int(L)][:, 0], d["M_abs_err"])(beta_pc)))
    gamma_nu = _power_exponent(LS[selected], np.asarray(chi_peaks)[selected], chi_errors[selected], False)
    beta_nu = _power_exponent(
        LS[selected], np.asarray(magnetizations)[selected], magnetization_errors[selected], True
    )
    return np.array([nu, gamma_nu, beta_nu])


def calibrate_and_derive(raw_theta: np.ndarray, centers: np.ndarray, sigmas: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calibrate bootstrap marginals and propagate all derived quantities."""
    if raw_theta.shape != (JOINT_REPLICAS, 3) or not np.isfinite(raw_theta).all():
        raise RuntimeError(f"expected {JOINT_REPLICAS} finite 3-component replicas, got {raw_theta.shape}")
    correlation = np.corrcoef(raw_theta, rowvar=False)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    correlation_eigenvalues = np.linalg.eigvalsh(correlation)
    if correlation_eigenvalues[0] <= 0.0:
        raise RuntimeError("bootstrap correlation matrix is not positive definite; no PSD projection applied")

    standardized = (raw_theta - raw_theta.mean(axis=0)) / raw_theta.std(axis=0, ddof=1)
    calibrated = centers + sigmas * standardized
    covariance = correlation * np.outer(sigmas, sigmas)
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    if covariance_eigenvalues[0] <= 0.0:
        raise RuntimeError("calibrated covariance is not positive definite; no PSD projection applied")

    nu, gamma_nu, beta_nu = calibrated.T
    derived = np.column_stack([
        gamma_nu * nu,
        beta_nu * nu,
        2.0 - gamma_nu,
        1.0 + gamma_nu / beta_nu,
        gamma_nu + 2.0 * beta_nu,
    ])
    central = np.array([
        centers[1] * centers[0], centers[2] * centers[0], 2.0 - centers[1],
        1.0 + centers[1] / centers[2], centers[1] + 2.0 * centers[2],
    ])
    names = ["gamma", "beta", "eta", "delta", "d_hyper"]
    derived_summary = {
        name: {
            "central": float(central[index]),
            "error": float(derived[:, index].std(ddof=1)),
            "mean_replica": float(derived[:, index].mean()),
            "p68": np.percentile(derived[:, index], [16, 84]).tolist(),
            "p95": np.percentile(derived[:, index], [2.5, 97.5]).tolist(),
        }
        for index, name in enumerate(names)
    }

    variance_gamma_nu = float(np.var(calibrated[:, 1], ddof=1))
    four_variance_beta_nu = float(4.0 * np.var(calibrated[:, 2], ddof=1))
    covariance_gamma_beta = float(np.cov(calibrated[:, 1], calibrated[:, 2], ddof=1)[0, 1])
    four_covariance = 4.0 * covariance_gamma_beta
    variance_hyper = float(np.var(derived[:, 4], ddof=1))
    rhs = variance_gamma_nu + four_variance_beta_nu + four_covariance
    independent_error = float(np.sqrt(variance_gamma_nu + four_variance_beta_nu))
    joint_error = float(np.sqrt(variance_hyper))

    checks = {
        "mean_max_abs": float(np.max(np.abs(calibrated.mean(axis=0) - centers))),
        "std_max_abs": float(np.max(np.abs(calibrated.std(axis=0, ddof=1) - sigmas))),
        "correlation_max_abs": float(np.max(np.abs(np.corrcoef(calibrated, rowvar=False) - correlation))),
    }
    if max(checks.values()) > 1e-12:
        raise RuntimeError(f"calibration invariant failed: {checks}")

    summary = {
        "correlation_matrix": correlation.tolist(),
        "calibrated_covariance": covariance.tolist(),
        "correlation_eigenvalues": correlation_eigenvalues.tolist(),
        "covariance_eigenvalues": covariance_eigenvalues.tolist(),
        "condition_number": float(np.linalg.cond(covariance)),
        "positive_definite": True,
        "psd_projection_applied": False,
        "calibration_checks": checks,
        "derived": derived_summary,
        "hyperscaling_variance_decomposition": {
            "var_gamma_nu": variance_gamma_nu,
            "four_var_beta_nu": four_variance_beta_nu,
            "cov_gamma_nu_beta_nu": covariance_gamma_beta,
            "four_cov_gamma_nu_beta_nu": four_covariance,
            "var_d_hyper": variance_hyper,
            "rhs": rhs,
            "identity_residual": variance_hyper - rhs,
            "joint_error": joint_error,
            "independent_error": independent_error,
            "percent_change": 100.0 * (joint_error / independent_error - 1.0),
        },
    }
    return calibrated, derived, summary


def run_joint_bootstrap(
    cache_dir: Path,
    cache_manifest: dict[str, Any],
    data: dict[int, dict[str, np.ndarray]],
    slope_covariance: np.ndarray,
    crossing_errors: np.ndarray,
    chi_errors: np.ndarray,
    magnetization_errors: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
) -> JointResult:
    """Run the validated 500-replica correlation-only bootstrap."""
    runs = load_cached_runs(cache_dir, cache_manifest)
    rng = np.random.default_rng(JOINT_SEED)
    raw = np.full((JOINT_REPLICAS, 3), np.nan)
    for replica in range(JOINT_REPLICAS):
        raw[replica] = _one_replica(
            runs, data, slope_covariance, crossing_errors,
            chi_errors, magnetization_errors, rng,
        )
        if (replica + 1) % 25 == 0:
            print(f"    joint replicas {replica + 1}/{JOINT_REPLICAS}", flush=True)
    valid = np.isfinite(raw).all(axis=1)
    if valid.sum() != JOINT_REPLICAS:
        raise RuntimeError(f"joint bootstrap produced only {valid.sum()}/{JOINT_REPLICAS} finite replicas")
    calibrated, derived, summary = calibrate_and_derive(raw, centers, sigmas)
    summary.update({
        "method": "synchronized per-run block bootstrap, correlation-only, calibrated to official marginals",
        "seed": JOINT_SEED,
        "requested": JOINT_REPLICAS,
        "valid": int(valid.sum()),
        "raw_mean": raw.mean(axis=0).tolist(),
        "raw_std": raw.std(axis=0, ddof=1).tolist(),
    })
    return JointResult(raw=raw, valid=valid, calibrated=calibrated, derived=derived, summary=summary)


def save_joint_replicas(path: Path, result: JointResult) -> None:
    np.savez_compressed(
        path,
        raw=result.raw,
        valid=result.valid,
        calibrated=result.calibrated,
        derived=result.derived,
        names=np.array(["nu", "gamma_nu", "beta_nu"]),
        derived_names=np.array(["gamma", "beta", "eta", "delta", "d_hyper"]),
        R_boot=np.asarray(result.summary["correlation_matrix"]),
        C_cal=np.asarray(result.summary["calibrated_covariance"]),
    )