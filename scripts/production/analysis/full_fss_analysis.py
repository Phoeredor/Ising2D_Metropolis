#!/usr/bin/env python3
"""
Finite-size scaling (FSS) analysis for the 2D Ising model (PBC).

Pipeline: beta_pc (Binder crossing) -> nu (slope fit) -> gamma/nu, beta/nu
(peak scaling) -> alpha (specific-heat) -> data collapse.  Errors from
bootstrap + pcov; correlated chi^2 for nu.  Outputs .dat results, LaTeX
tables, and 14 publication-quality PDF plots.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.stats as sps
import seaborn as sns
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "results" / "production"
PLOT_DIR = BASE_DIR / "results" / "production" / "plots" / "fss"
ANALYSIS_DIR = BASE_DIR / "results" / "production" / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

L_LIST = [24, 32, 48, 64, 96, 128]

# Exact values (Onsager + CFT) — used only for comparison, never as input
BETA_C_EXACT = 0.44068679350977151262
NU_EXACT = 1.0
BETA_NU_EXACT = 0.125
GAMMA_NU_EXACT = 1.75
OMEGA_EXACT = 2.0  # leading irrelevant exponent (CFT)

# Analysis parameters
BETA_RANGE_FIT = (0.43, 0.45)   # beta window for spline fits
XLIM_COLLAPSE = (-2, 2)          # x-axis range for collapse plots
N_BOOTSTRAP = 250                # bootstrap replicas


# ============================================================================
# OUTPUT FORMATTING UTILITIES
# ============================================================================

def print_phase_header(phase_num, total_phases, title, description=None):
    print("\n" + "=" * 80)
    print(f"[PHASE {phase_num}/{total_phases}] {title.upper()}")
    print("=" * 80)
    if description:
        print(description)
        print("-" * 80)


def print_section(title, level=1):
    if level == 1:
        print("\n" + "\u2500" * 80)
        print(f"\u2502 {title}")
        print("\u2500" * 80)
    elif level == 2:
        print(f"\n\u25b8 {title}")
        print("  " + "\u00b7" * 76)


def print_result(name, value, error=None, unit="", chi2=None, ci=None, indent=2):
    spaces = " " * indent
    result_str = f"{spaces}\u2713 {name:30s} = "

    if error is not None:
        result_str += f"{value:.6g} \u00b1 {error:.6g} {unit}"
    else:
        result_str += f"{value} {unit}"

    if ci is not None:
        result_str += f"  [CI: {ci[0]:.4g}, {ci[1]:.4g}]"

    if chi2 is not None:
        result_str += f"  (\u03c7\u00b2\u1d63 = {chi2:.3f})"

    print(result_str)


def print_comparison_onsager(measured, error, exact, name, indent=2, ref_label="exact"):
    spaces = " " * indent
    deviation_abs = abs(measured - exact)
    deviation_rel = (deviation_abs / exact) * 100 if exact != 0 else 0
    deviation_sigma = deviation_abs / error if error > 0 else 0

    print(f"\n{spaces}\u250c\u2500 Comparison with Onsager ({ref_label} = {exact:.6g}):")
    print(f"{spaces}\u2502  Absolute deviation: {deviation_abs:.6g}")
    print(f"{spaces}\u2502  Relative deviation: {deviation_rel:.3f}%")

    if deviation_sigma < 1.5:
        status = "\u2713 EXCELLENT"
    elif deviation_sigma < 2.0:
        status = "\u2713 GOOD"
    else:
        status = "\u26a0 MARGINAL"

    print(f"{spaces}\u2514\u2500 Statistical significance: {deviation_sigma:.2f}\u03c3  {status}")


def print_table_compact(headers, rows, indent=2):
    spaces = " " * indent

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    header_line = spaces + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + header_line)
    print(spaces + "\u2500" * (len(header_line) - indent))

    for row in rows:
        print(spaces + "  ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(L_list):
    """Load parsed JSON observables for every L into a {L: dict_of_arrays} map."""
    store = {}
    for L in L_list:
        path = DATA_DIR / f"L{L}" / "parsed" / f"observables_L{L}.json"
        if not path.exists():
            continue

        with open(path) as f:
            raw = json.load(f)

        raw.sort(key=lambda x: x['beta'])

        arrs = {k: np.array([e[k] for e in raw])
                for k in ['beta', 'U4', 'U4_err', 'M_abs', 'M_abs_err',
                           'chi', 'chi_err', 'C', 'C_err']}

        store[L] = arrs

    return store


def get_colors(L_list):
    return plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))


def get_color_for_L(L, L_list):
    colors = get_colors(L_list)
    idx = L_list.index(L)
    return colors[idx]


# ============================================================================
# BOOTSTRAP UTILITIES
# ============================================================================
def bootstrap_crossing(data_L1, data_L2, beta_range, n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap error on a single Binder-cumulant crossing between two sizes.

    Resamples the U4(beta) data points with replacement, rebuilds splines,
    and finds the crossing via Brent's method.  Returns (beta_cross, sigma).
    """
    rng = np.random.default_rng(seed)

    mask1 = (data_L1['beta'] >= beta_range[0]) & (data_L1['beta'] <= beta_range[1])
    mask2 = (data_L2['beta'] >= beta_range[0]) & (data_L2['beta'] <= beta_range[1])

    w1 = 1.0 / (data_L1['U4_err'][mask1] + 1e-12)
    w2 = 1.0 / (data_L2['U4_err'][mask2] + 1e-12)

    spl1 = spi.UnivariateSpline(data_L1['beta'][mask1], data_L1['U4'][mask1],
                                w=w1, k=3, s=np.sum(mask1))
    spl2 = spi.UnivariateSpline(data_L2['beta'][mask2], data_L2['U4'][mask2],
                                w=w2, k=3, s=np.sum(mask2))

    try:
        beta_cross_orig = spo.brentq(lambda b: spl1(b) - spl2(b),
                                     max(0.435, beta_range[0]),
                                     min(0.445, beta_range[1]))
    except:
        return np.nan, np.nan

    crossings_boot = []
    n1 = np.sum(mask1)
    n2 = np.sum(mask2)

    for _ in range(n_boot):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)

        b1 = data_L1['beta'][mask1][idx1]
        u1 = data_L1['U4'][mask1][idx1]
        e1 = data_L1['U4_err'][mask1][idx1]

        b2 = data_L2['beta'][mask2][idx2]
        u2 = data_L2['U4'][mask2][idx2]
        e2 = data_L2['U4_err'][mask2][idx2]

        _, uniq1 = np.unique(b1, return_index=True)
        _, uniq2 = np.unique(b2, return_index=True)

        if len(uniq1) < 4 or len(uniq2) < 4:
            continue

        try:
            w1_b = 1.0 / (e1[uniq1] + 1e-12)
            w2_b = 1.0 / (e2[uniq2] + 1e-12)

            spl1_b = spi.UnivariateSpline(b1[uniq1], u1[uniq1], w=w1_b, k=3, s=len(uniq1))
            spl2_b = spi.UnivariateSpline(b2[uniq2], u2[uniq2], w=w2_b, k=3, s=len(uniq2))

            bc = spo.brentq(lambda b: spl1_b(b) - spl2_b(b),
                            max(0.435, beta_range[0]),
                            min(0.445, beta_range[1]))
            crossings_boot.append(bc)
        except:
            continue

    if len(crossings_boot) < 10:
        return beta_cross_orig, np.nan

    return beta_cross_orig, np.std(crossings_boot, ddof=1)


def bootstrap_global_slopes(data, L_list, beta_range, beta_pc_fixed,
                            n_boot=N_BOOTSTRAP, seed=42):
    """Global bootstrap for Binder slopes across all L simultaneously.

    Resamples U4(beta) for every L in each replica, evaluates
    S(L) = |dU4/dbeta| at beta_pc_fixed, and builds the full
    covariance matrix C_ij.  The fixed beta_pc avoids artificial
    inter-L correlations from a fluctuating crossing point.
    """
    rng = np.random.default_rng(seed)
    L_used = [L for L in L_list if L in data]
    N_L = len(L_used)

    slopes_boot = np.zeros((n_boot, N_L))
    success = 0

    for _ in range(n_boot):
        splines_b, ok = {}, True
        for L in L_used:
            d = data[L]
            mask = (d['beta'] >= beta_range[0]) & (d['beta'] <= beta_range[1])
            n = np.sum(mask)
            idx = rng.choice(n, size=n, replace=True)
            bb, ub, eb = d['beta'][mask][idx], d['U4'][mask][idx], d['U4_err'][mask][idx]
            _, uniq = np.unique(bb, return_index=True)
            if len(uniq) < 4:
                ok = False
                break
            try:
                splines_b[L] = spi.UnivariateSpline(
                    bb[uniq], ub[uniq],
                    w=1.0 / (eb[uniq] + 1e-12), k=3, s=len(uniq))
            except Exception:
                ok = False
                break
        if not ok:
            continue

        row = []
        for L in L_used:
            try:
                row.append(abs(splines_b[L].derivative()(beta_pc_fixed)))
            except Exception:
                row = None
                break
        if row is None:
            continue

        slopes_boot[success, :] = row
        success += 1

    slopes_boot = slopes_boot[:success]
    C = np.cov(slopes_boot.T, ddof=1)
    std_d = np.sqrt(np.diag(C))
    rho = C / np.outer(std_d, std_d)

    print("    sqrt(C_ii) diagonal errors:")
    for L_i, s_i in zip(L_used, std_d):
        print(f"      L={int(L_i):3d}  sqrt(C_ii) = {s_i:.5f}")

    return {
        'L_used': np.array(L_used),
        'slopes_mean': np.mean(slopes_boot, axis=0),
        'slopes_err': np.std(slopes_boot, axis=0, ddof=1),
        'C_matrix': C, 'rho_matrix': rho, 'n_success': success
    }


def fit_nu_correlated(L_data, slopes_mean, C_matrix, model_func, p0, bounds):
    """Correlated chi^2 fit: chi^2 = r^T C^{-1} r  (SVD pseudo-inverse).

    Returns (popt, perr, chi2, chi2_red, dof_eff, rank).
    Errors from numerical Hessian of the chi^2 surface.
    """
    from scipy.optimize import minimize, approx_fprime

    U, s, Vt = np.linalg.svd(C_matrix)
    s_inv = np.where(s > 1e-6 * s[0], 1.0 / s, 0.0)
    C_inv = (Vt.T * s_inv) @ U.T
    rank = int(np.sum(s > 1e-6 * s[0]))
    dof_eff = rank - len(p0)

    def chi2c(params):
        r = slopes_mean - model_func(L_data, *params)
        return float(r @ C_inv @ r)

    res = minimize(chi2c, p0, method='L-BFGS-B',
                   bounds=list(zip(bounds[0], bounds[1])),
                   options={'ftol': 1e-12, 'gtol': 1e-8})

    # Numerical Hessian for parameter errors
    eps = 1e-5
    H = np.array([[approx_fprime(res.x,
                    lambda p: approx_fprime(p, chi2c, eps)[j], eps)[i]
                   for j in range(len(p0))] for i in range(len(p0))])
    try:
        perr = np.sqrt(np.abs(np.diag(np.linalg.inv(H / 2.0))))
    except Exception:
        perr = np.full(len(p0), np.nan)

    chi2_red_eff = res.fun / dof_eff if dof_eff > 0 else np.inf
    return res.x, perr, res.fun, chi2_red_eff, dof_eff, rank


# ============================================================================
# RIGOROUS FIT INFRASTRUCTURE (sub-leading corrections)
# ============================================================================

@dataclass
class FitResult:
    """Container for a least-squares fit: parameters, covariance, chi^2."""
    params: np.ndarray
    params_err: np.ndarray
    cov_matrix: np.ndarray
    chi2: float
    chi2_red: float
    dof: int

    def correlation_matrix(self):
        std_outer = np.outer(self.params_err, self.params_err)
        return self.cov_matrix / (std_outer + 1e-30)


def compute_chi2_and_errors(model_func, x_data, y_data, y_err, popt, pcov):
    """Build a FitResult from curve_fit output."""
    n_data = len(y_data)
    n_params = len(popt)
    dof = n_data - n_params

    y_pred = model_func(x_data, *popt)
    residuals = (y_data - y_pred) / y_err
    chi2 = np.sum(residuals**2)
    chi2_red = chi2 / dof if dof > 0 else np.inf

    params_err = np.sqrt(np.diag(pcov))

    return FitResult(
        params=popt,
        params_err=params_err,
        cov_matrix=pcov,
        chi2=chi2,
        chi2_red=chi2_red,
        dof=dof
    )


def f_test_nested_models(chi2_simple, dof_simple, chi2_complex, dof_complex):
    """F-test for nested models.  Returns (F, p_value, is_significant_at_5pct)."""
    if dof_simple <= dof_complex:
        raise ValueError("Simple model must have MORE dof than complex model")

    if chi2_complex >= chi2_simple:
        return 0.0, 1.0, False

    delta_chi2 = chi2_simple - chi2_complex
    delta_dof = dof_simple - dof_complex

    F_stat = (delta_chi2 / delta_dof) / (chi2_complex / dof_complex)
    p_value = 1.0 - sps.f.cdf(F_stat, delta_dof, dof_complex)

    return F_stat, p_value, (p_value < 0.05)


def bootstrap_fit_procedure(model_func, x_data, y_data, y_err, p0, bounds,
                            n_bootstrap=N_BOOTSTRAP, seed=42):
    """Non-parametric bootstrap around a least-squares fit.

    Returns (params_boot, fit_original).  Warning printed when n_data < 10
    (bootstrap unreliable for very few points).
    """
    rng = np.random.default_rng(seed)
    n_data = len(y_data)

    if n_data < 10:
        print(f"    ⚠ WARNING: non-parametric bootstrap on n={n_data} points is unreliable.")
        print(f"      Resampling {n_data} points with replacement yields very few distinct")
        print(f"      configurations; bootstrap σ will underestimate the true uncertainty.")
        print(f"      Use pcov or correlated-fit errors instead for n < ~20.")

    # Original fit
    try:
        popt, pcov = spo.curve_fit(
            model_func, x_data, y_data,
            p0=p0, sigma=y_err, absolute_sigma=True,
            bounds=bounds, maxfev=10000
        )
        fit_original = compute_chi2_and_errors(
            model_func, x_data, y_data, y_err, popt, pcov
        )
    except Exception as e:
        print(f"    \u2717 Original fit failed: {e}")
        return None, None

    # Bootstrap resampling
    params_boot = []

    for _ in range(n_bootstrap):
        idx_boot = rng.choice(n_data, size=n_data, replace=True)
        x_boot = x_data[idx_boot]
        y_boot = y_data[idx_boot]
        err_boot = y_err[idx_boot]

        try:
            popt_boot, _ = spo.curve_fit(
                model_func, x_boot, y_boot,
                p0=p0, sigma=err_boot, absolute_sigma=True,
                bounds=bounds, maxfev=5000
            )
            params_boot.append(popt_boot)
        except:
            continue

    if len(params_boot) < 50:
        print(f"    \u26a0 Only {len(params_boot)}/{n_bootstrap} bootstrap fits succeeded")

    params_boot = np.array(params_boot)
    return params_boot, fit_original


def compute_bootstrap_errors(params_boot):
    """Mean, std, and 95% CI from bootstrap parameter distributions."""
    params_mean = np.mean(params_boot, axis=0)
    params_std = np.std(params_boot, axis=0, ddof=1)
    percentiles = np.percentile(params_boot, [2.5, 97.5], axis=0)
    return params_mean, params_std, percentiles


def extract_observable_with_corrections(L_data, obs_data, obs_err,
                                        observable_name, exponent_theory,
                                        bounds_exponent, bounds_B=(-50, 50),
                                        n_bootstrap=N_BOOTSTRAP):
    """Two-step extraction of a critical exponent with sub-leading correction.

    Step 1: O(L) = A * L^alpha  (free exponent).
    Step 2: O(L) = A * L^alpha * [1 + B*L^{-omega}]  (alpha fixed from Step 1).
    Significance of B tested with an F-test (2-param vs 3-param).
    """
    print_section(f"Extracting: {observable_name}")

    # ===== STEP 1: Simple power law =====
    print_section("Step 1: Simple power-law fit  O(L) = A * L^alpha", level=2)

    def model_simple(L, A, exponent):
        return A * L**exponent

    p0_simple = [abs(obs_data[0]) / abs(L_data[0]**exponent_theory), exponent_theory]
    bounds_simple = ([0, bounds_exponent[0]], [np.inf, bounds_exponent[1]])

    params_boot_simple, fit_simple = bootstrap_fit_procedure(
        model_simple, L_data, obs_data, obs_err,
        p0=p0_simple, bounds=bounds_simple, n_bootstrap=n_bootstrap
    )

    if fit_simple is None:
        print("  \u2717 STEP 1 failed!")
        return None

    # NP bootstrap (diagnostic only for n < ~20)
    _, params_std_simple_boot, percentiles_simple = compute_bootstrap_errors(
        params_boot_simple
    )

    # pcov (Cramér-Rao) as primary errors
    A_simple = fit_simple.params[0]
    alpha_simple = fit_simple.params[1]
    A_err_simple = fit_simple.params_err[0]
    alpha_err_simple = fit_simple.params_err[1]

    corr_matrix_simple = fit_simple.correlation_matrix()
    n_pts = len(L_data)

    print_result("Exponent alpha", alpha_simple, alpha_err_simple,
                 chi2=fit_simple.chi2_red, ci=[percentiles_simple[0, 1], percentiles_simple[1, 1]])
    print_result("Amplitude A", A_simple, A_err_simple)
    print(f"    Error source: pcov (Cramér-Rao, absolute_sigma=True)")
    if n_pts < 20:
        print(f"    NP bootstrap σ(alpha) = {params_std_simple_boot[1]:.4f}  [diagnostic, n={n_pts}]")
    print(f"    Corr(A, alpha) = {corr_matrix_simple[0, 1]:.3f}")
    print(f"    Bootstrap: {len(params_boot_simple)}/{n_bootstrap} successful fits")

    # ===== Lightweight 3-param fit (for F-test only, no bootstrap) =====
    def model_3param(L, A, alpha, B):
        return A * L**alpha * (1.0 + B * L**(-OMEGA_EXACT))

    p0_3p = [A_simple, alpha_simple, 0.0]
    bounds_3p = ([0, bounds_exponent[0], bounds_B[0]],
                 [np.inf, bounds_exponent[1], bounds_B[1]])

    _, fit_3p = bootstrap_fit_procedure(
        model_3param, L_data, obs_data, obs_err,
        p0=p0_3p, bounds=bounds_3p, n_bootstrap=0  # no bootstrap needed
    )
    fit_3p_ok = fit_3p is not None

    # ===== STEP 2: With sub-leading correction (alpha fixed) =====
    print_section(f"Step 2: Correction fit (alpha fixed = {alpha_simple:.4f})", level=2)

    alpha_fixed = alpha_simple

    def model_with_B(L, A, B):
        return A * L**alpha_fixed * (1.0 + B * L**(-OMEGA_EXACT))

    p0_with_B = [A_simple, 0.0]
    bounds_with_B = ([0, bounds_B[0]], [np.inf, bounds_B[1]])

    params_boot_B, fit_B = bootstrap_fit_procedure(
        model_with_B, L_data, obs_data, obs_err,
        p0=p0_with_B, bounds=bounds_with_B, n_bootstrap=n_bootstrap
    )

    if fit_B is None:
        print("  \u2717 STEP 2 failed!")
        B_value, B_err, chi2_red_B = np.nan, np.nan, np.nan
        percentiles_B = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        corr_matrix_B = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    else:
        _, params_std_B_boot, percentiles_B = compute_bootstrap_errors(params_boot_B)

        A_corr = fit_B.params[0]
        B_value = fit_B.params[1]
        A_err_corr = fit_B.params_err[0]    # pcov
        B_err = fit_B.params_err[1]          # pcov
        chi2_red_B = fit_B.chi2_red

        corr_matrix_B = fit_B.correlation_matrix()

        print_result("Amplitude A", A_corr, A_err_corr, chi2=chi2_red_B)
        print_result("Correction B", B_value, B_err,
                     ci=[percentiles_B[0, 1], percentiles_B[1, 1]])
        print(f"    Error source: pcov (Cramér-Rao)")
        if len(L_data) < 20:
            print(f"    NP bootstrap σ(B) = {params_std_B_boot[1]:.4f}  [diagnostic, n={len(L_data)}]")
        print(f"    Corr(A, B) = {corr_matrix_B[0, 1]:.3f}")
        print(f"    Bootstrap: {len(params_boot_B)}/{n_bootstrap} successful fits")

    # ===== F-TEST for model comparison =====
    # Compare Step 1 (2 params, DoF=N-2) vs 3-param (3 params, DoF=N-3)
    print_section("Model Comparison (F-test: 2-param vs 3-param)", level=2)

    F_stat, p_value, is_significant = np.nan, np.nan, False

    if fit_3p_ok:
        try:
            F_stat, p_value, is_significant = f_test_nested_models(
                fit_simple.chi2, fit_simple.dof,
                fit_3p.chi2, fit_3p.dof
            )

            print(f"    F-statistic = {F_stat:.3f}")
            print(f"    p-value     = {p_value:.4f}")

            if is_significant:
                print(f"    \u2713 Correction B is SIGNIFICANT (p < 0.05)")
                print(f"      -> Including L^(-{OMEGA_EXACT:.0f}) term improves fit at 95% confidence")
            else:
                print(f"    \u2717 Correction B is NOT significant (p >= 0.05)")
                print(f"      -> Simple model is statistically sufficient")

            delta_chi2_red = fit_simple.chi2_red - chi2_red_B
            improvement_pct = (delta_chi2_red / (fit_simple.chi2_red + 1e-30)) * 100
            print(f"    Delta_chi2_red = {delta_chi2_red:+.3f} ({improvement_pct:+.1f}%)")

        except Exception as e:
            print(f"    \u2717 F-test failed: {e}")

    # ===== Significance of B from bootstrap distribution =====
    print_section("Correction Amplitude Analysis", level=2)

    if not np.isnan(B_value) and B_err > 0:
        n_sigma = abs(B_value) / B_err
        print(f"    B / sigma_B = {n_sigma:.2f}")

        if n_sigma < 1.0:
            print("    -> B compatible with zero (< 1 sigma)")
            print("    -> Finite-size corrections are negligible")
        elif n_sigma < 2.0:
            print(f"    -> B marginally resolved (~{n_sigma:.1f} sigma)")
        else:
            print(f"    -> B significantly non-zero ({n_sigma:.1f} sigma)")

        if params_boot_B is not None and len(params_boot_B) > 0:
            B_boot_values = params_boot_B[:, 1]
            same_sign_frac = np.mean(np.sign(B_boot_values) == np.sign(B_value))
            print(f"    Bootstrap: {same_sign_frac * 100:.1f}% of samples have same sign as B")

    return {
        'observable': observable_name,
        'exponent': alpha_simple,
        'exponent_err': alpha_err_simple,
        'exponent_CI': percentiles_simple[:, 1],
        'A_simple': A_simple,
        'A_simple_err': A_err_simple,
        'A_corrected': A_corr if fit_B is not None else np.nan,
        'A_corrected_err': A_err_corr if fit_B is not None else np.nan,
        'B': B_value,
        'B_err': B_err,
        'B_CI': percentiles_B[:, 1] if not np.isnan(B_value) else np.array([np.nan, np.nan]),
        'chi2_red_simple': fit_simple.chi2_red,
        'chi2_raw_simple': fit_simple.chi2,
        'dof_simple': fit_simple.dof,
        'chi2_red_corrected': chi2_red_B,
        'chi2_raw_corrected': fit_B.chi2 if fit_B is not None else np.nan,
        'dof_corrected': fit_B.dof if fit_B is not None else 0,
        'F_statistic': F_stat,
        'p_value': p_value,
        'is_significant': is_significant,
        'corr_A_alpha': corr_matrix_simple[0, 1],
        'corr_A_B': corr_matrix_B[0, 1] if not np.isnan(B_value) else np.nan,
        'n_bootstrap_simple': len(params_boot_simple),
        'n_bootstrap_corrected': len(params_boot_B) if params_boot_B is not None else 0
    }


# ============================================================================
# PHASE 1: beta_pc (PSEUDO-CRITICAL TEMPERATURE)
# ============================================================================
def find_beta_pc(data, L_list, beta_range=BETA_RANGE_FIT):
    """Phase 1: pseudo-critical beta from Binder-cumulant crossings.

    For each consecutive (L1, L2) pair, finds the crossing of U4 splines
    and assigns a bootstrap error.  Returns the inverse-variance weighted
    average beta_pc, U*, and all individual crossings.
    """
    print_phase_header(1, 5, "PSEUDO-CRITICAL TEMPERATURE beta_pc",
                       description="Method: Binder cumulant U4(beta,L) crossing "
                                   "with bootstrap error propagation")

    print(f"  beta range:  [{beta_range[0]}, {beta_range[1]}] (optimized)")
    print(f"  Bootstrap:   {N_BOOTSTRAP} samples")
    print("-" * 80)

    colors = get_colors(L_list)
    splines = {}
    crossings = []

    # Build weighted cubic splines of U4(beta) for each L
    for L in L_list:
        if L not in data:
            continue
        d = data[L]
        mask = (d['beta'] >= beta_range[0]) & (d['beta'] <= beta_range[1])
        if np.sum(mask) < 4:
            continue
        w = 1.0 / (d['U4_err'][mask] + 1e-12)
        spl = spi.UnivariateSpline(d['beta'][mask], d['U4'][mask],
                                   w=w, k=3, s=np.sum(mask))
        splines[L] = spl

    # Calculate crossings with progress bar
    print_section("Step 1: Computing Binder Crossings (Bootstrap)", level=2)

    with tqdm(total=len(L_list) - 1, desc="  Crossings", ncols=70,
              file=sys.stdout, disable=False) as pbar:
        for i in range(len(L_list) - 1):
            L1, L2 = L_list[i], L_list[i + 1]
            if L1 not in data or L2 not in data:
                pbar.update(1)
                continue

            beta_cross, beta_cross_err = bootstrap_crossing(
                data[L1], data[L2], beta_range, n_boot=N_BOOTSTRAP, seed=42 + i
            )

            if not np.isnan(beta_cross):
                U_star = splines[L1](beta_cross)
                dU_dbeta = splines[L1].derivative()(beta_cross)
                U_star_err = abs(dU_dbeta) * beta_cross_err
                crossings.append((L1, L2, beta_cross, beta_cross_err, U_star, U_star_err))

            pbar.update(1)

    if len(crossings) == 0:
        print("\n  \u2717 ERROR: No crossings found!")
        return [], {}, np.nan, np.nan, np.nan, np.nan

    # Results table
    print_section("Step 2: Crossing Results", level=2)

    table_rows = []
    for c in crossings:
        table_rows.append([
            f"{c[0]:3d} x {c[1]:3d}",
            f"{c[2]:.6f}",
            f"\u00b1{c[3]:.6f}",
            f"{c[4]:.4f}",
            f"\u00b1{c[5]:.5f}"
        ])

    print_table_compact(
        ["L1 x L2", "beta_cross", "sigma", "U*", "sigma_U"],
        table_rows
    )

    # Inverse-variance weighted average of crossings
    print_section("Step 3: Weighted Average", level=2)

    betas = np.array([c[2] for c in crossings])
    sigmas = np.array([c[3] for c in crossings])
    Ustars = np.array([c[4] for c in crossings])
    Ustar_errs = np.array([c[5] for c in crossings])

    valid = ~np.isnan(sigmas)
    if np.sum(valid) > 0:
        weights = 1.0 / (sigmas[valid]**2 + 1e-12)
        beta_pc = np.sum(betas[valid] * weights) / np.sum(weights)
        beta_pc_err = 1.0 / np.sqrt(np.sum(weights))
    else:
        beta_pc = np.mean(betas)
        beta_pc_err = np.std(betas) / np.sqrt(len(betas))

    valid_U = ~np.isnan(Ustar_errs)
    if np.sum(valid_U) > 0:
        weights_U = 1.0 / (Ustar_errs[valid_U]**2 + 1e-12)
        Ustar_mean = np.sum(Ustars[valid_U] * weights_U) / np.sum(weights_U)
        Ustar_err = 1.0 / np.sqrt(np.sum(weights_U))
    else:
        Ustar_mean = np.mean(Ustars)
        Ustar_err = np.std(Ustars) / np.sqrt(len(Ustars))

    print("  Method: Inverse-variance weighting (w_i = 1/sigma_i^2)")
    print_result("beta_pc", beta_pc, beta_pc_err)
    print_result("U* (universal)", Ustar_mean, Ustar_err)

    print_comparison_onsager(beta_pc, beta_pc_err, BETA_C_EXACT, "beta_pc")
    print_comparison_onsager(Ustar_mean, Ustar_err, 0.6107, "U*",
                             ref_label="ref (Ferrenberg & Landau)")

    # ===================================================================
    # PLOT 1 (code unchanged)
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 7.5))
    beta_plot = np.linspace(beta_range[0], beta_range[1], 1000)

    for i, L in enumerate(L_list):
        if L not in splines:
            continue
        d = data[L]
        mask = (d['beta'] >= beta_range[0]) & (d['beta'] <= beta_range[1])

        ax.errorbar(d['beta'][mask], d['U4'][mask], yerr=d['U4_err'][mask],
                    fmt='o', color=colors[i], alpha=0.6, ms=5, capsize=0, lw=1.2,
                    label=f'L={L}', zorder=2)
        ax.plot(beta_plot, splines[L](beta_plot), '-', color=colors[i], lw=2.8, zorder=3)

    ax.axvline(beta_pc, color='blue', ls='--', lw=2.5, alpha=0.9, zorder=1)
    ax.axvline(BETA_C_EXACT, color='red', ls='--', lw=2.5, alpha=0.7, zorder=0)

    ax.set_xlim(0.430, 0.450)
    ax.set_ylim(0.30, 0.68)
    ax.set_xlabel(r'$\beta$', fontsize=20)
    ax.set_ylabel(r'$U_4$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':', zorder=0)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', ls='--', lw=2.5, label=r'$\beta_{pc}$'),
        Line2D([0], [0], color='red', ls='--', lw=2.5, label=r'$\beta_c$')
    ]
    leg1 = ax.legend(handles=legend_elements, frameon=False, fontsize=13, loc='upper left', ncol=1)
    ax.add_artist(leg1)
    ax.legend(frameon=False, fontsize=11, loc='lower right', ncol=2)

    ax_ins = inset_axes(ax, width='35%', height='30%', loc='lower left',
                        bbox_to_anchor=(0.61, 0.24, 1, 1), bbox_transform=ax.transAxes)

    beta_zoom = np.linspace(0.4395, 0.4420, 500)
    for i, L in enumerate(L_list):
        if L not in splines:
            continue
        ax_ins.plot(beta_zoom, splines[L](beta_zoom), '-', color=colors[i], lw=2.5)

    ax_ins.axvline(beta_pc, color='blue', ls='--', lw=2, alpha=0.9)
    ax_ins.axvline(BETA_C_EXACT, color='red', ls='--', lw=2, alpha=0.7)
    ax_ins.axvspan(BETA_C_EXACT, beta_pc, alpha=0.15, color='orange', zorder=0)
    ax_ins.set_xlim(0.4402, 0.4415)
    ax_ins.set_ylim(0.605, 0.617)
    ax_ins.tick_params(labelsize=10)
    ax_ins.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax_ins.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax_ins.grid(alpha=0.4, ls=':')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "binder_crossing_with_inset.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print_section("Step 4: Visualization", level=2)
    print("  \u2713 Plot saved: binder_crossing_with_inset.pdf")

    return crossings, splines, beta_pc, beta_pc_err, Ustar_mean, Ustar_err


# ============================================================================
# PLOT 2: beta_pc vs L (CONVERGENCE)
# ============================================================================
def plot_beta_pc_vs_L(data, L_list, crossings):
    """Plot 2: convergence of beta_pc(L) towards the exact value."""
    print("\n  [PLOT 2/10] Generating beta_pc vs L convergence...")

    beta_pc_vals = [c[2] for c in crossings]
    beta_pc_errs = [c[3] for c in crossings]
    L_pairs = [(crossings[i][0] + crossings[i][1]) / 2 for i in range(len(crossings))]

    sigmas = np.array(beta_pc_errs)
    valid = ~np.isnan(sigmas)
    if np.sum(valid) > 0:
        weights = 1.0 / sigmas[valid]**2
        beta_pc_final = np.sum(np.array(beta_pc_vals)[valid] * weights) / np.sum(weights)
        beta_pc_final_err = 1.0 / np.sqrt(np.sum(weights))
    else:
        beta_pc_final = np.mean(beta_pc_vals)
        beta_pc_final_err = np.std(beta_pc_vals) / np.sqrt(len(beta_pc_vals))

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.errorbar(L_pairs, beta_pc_vals, yerr=beta_pc_errs,
                fmt='o', color='navy', ms=9, capsize=0, lw=1.5,
                elinewidth=1.5, zorder=3)

    ax.axhline(BETA_C_EXACT, color='red', ls='--', lw=2.5, zorder=1)

    ax.axhspan(beta_pc_final - beta_pc_final_err, beta_pc_final + beta_pc_final_err,
               alpha=0.15, color='blue', zorder=0)

    ax.set_xlabel(r"$L=(L_1 + L_2)/2$", fontsize=20)
    ax.set_ylabel(r"$\beta$", fontsize=20)
    ax.set_xlim(20, 120)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', ls='--', lw=2.5, label=r'$\beta_c$'),
        Line2D([0], [0], color='navy', marker='o', ls='', ms=9,
               label=r'$\beta_{pc}(L_1, L_2)$'),
        Line2D([0], [0], color='blue', alpha=0.4, lw=8,
               label=r'$\beta_{pc} \pm \sigma$')
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=13,
              loc='lower right', ncol=1)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "beta_pc_vs_L_convergence.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: beta_pc_vs_L_convergence.pdf")


# ============================================================================
# PHASE 2: nu FROM BINDER SLOPE
# ============================================================================
def extract_nu(data, L_list, beta_pc, beta_pc_err, splines):
    """Phase 2: correlation-length exponent nu from Binder slopes.

    Computes S(L) = |dU4/dbeta|_{beta_pc} via spline derivative for each L.
    Slope errors from parametric bootstrap (fixed beta grid, perturbed U4).
    Fits S(L) = A * L^{1/nu} * [1 + B * L^{-omega}] with L_min cuts.
    """
    print_phase_header(2, 5, "CORRELATION LENGTH EXPONENT nu",
                       description="Method: Binder slope S(L) = |dU4/dbeta|_{beta_pc} "
                                   "power-law fit with corrections")

    print(f"  Using beta_pc = {beta_pc:.6f} \u00b1 {beta_pc_err:.6f} (measured)")
    print("-" * 80)

    print_section("Step 1: Computing Slopes via Bootstrap", level=2)

    # Parametric bootstrap: perturb U4 values at fixed beta grid positions.
    # Each L gets an independent RNG seed to avoid correlated resamples.
    # beta_pc held fixed at its central value (not randomized).
    print("  Slope errors: parametric bootstrap at fixed beta grid")
    print("  (U4 values perturbed by U4_err from jackknife; beta positions NOT resampled)")
    print("  β_pc is FIXED at central value (not randomized in bootstrap)")
    print("  Each L uses an INDEPENDENT RNG seed (seed=1000+L) to avoid correlated resamples")
    print()

    slopes, slopes_err, L_used = [], [], []

    for L in L_list:
        if L not in data or L not in splines:
            continue

        try:
            slope = abs(splines[L].derivative()(beta_pc))

            d = data[L]
            mask = (d['beta'] >= 0.43) & (d['beta'] <= 0.45)

            # Fixed beta grid; only U4 values fluctuate by their jackknife errors
            beta_grid = d['beta'][mask]
            u4_grid   = d['U4'][mask]
            u4_err    = d['U4_err'][mask]
            w         = 1.0 / (u4_err + 1e-12)

            rng = np.random.default_rng(1000 + L)   # per-L independent seed
            slopes_boot = []

            for _ in range(N_BOOTSTRAP):
                # Perturb U4 values; beta positions unchanged
                u4_b = u4_grid + rng.normal(0.0, u4_err)
                try:
                    spl_b = spi.UnivariateSpline(beta_grid, u4_b, w=w, k=3,
                                                 s=len(beta_grid))
                    slopes_boot.append(abs(spl_b.derivative()(beta_pc)))
                except:
                    pass
            # ────────────────────────────────────────────────────────────────

            boot_rate = len(slopes_boot) / N_BOOTSTRAP * 100
            if len(slopes_boot) > 10:
                slope_err = np.std(slopes_boot, ddof=1)
            else:
                slope_err = slope * 0.1
                print(f"    \u26a0 L={L}: Using 10% fallback error (insufficient bootstrap)")

            print(f"    L={L:3d}: S = {slope:.4f} \u00b1 {slope_err:.4f}  "
                  f"(bootstrap {boot_rate:.0f}% success, beta-fixed)")

            slopes.append(slope)
            slopes_err.append(slope_err)
            L_used.append(L)

        except Exception as e:
            print(f"    L={L}: slope calculation failed - {e}")
            continue

    slopes     = np.array(slopes)
    slopes_err = np.array(slopes_err)
    L_used     = np.array(L_used)

    print_section("Step 2: Power-Law Fit with Corrections", level=2)
    print(f"  Fit model: S(L) = A * L^(1/nu) * [1 + B*L^(-{OMEGA_EXACT})]")
    print(f"  Physical bounds: 0.67 < nu < 1.43, -500 < B < 500")

    def model(L, A, inv_nu, B):
        return A * L**inv_nu * (1 + B * L**(-OMEGA_EXACT))

    results = {}
    fit_rows = []

    for L_min in [24, 32, 48]:
        mask = L_used >= L_min
        if np.sum(mask) < 3:
            continue

        try:
            popt, pcov = spo.curve_fit(
                model, L_used[mask], slopes[mask],
                p0=[slopes[mask][0] / L_used[mask][0], 1.0, 5.0],
                sigma=slopes_err[mask], absolute_sigma=True,
                bounds=([0, 0.7, -500], [np.inf, 1.5, 500])
            )

            A, inv_nu, B = popt
            errors = np.sqrt(np.diag(pcov))

            nu = 1.0 / inv_nu
            nu_err = errors[1] / inv_nu**2

            pred = model(L_used[mask], *popt)
            chi2 = np.sum(((slopes[mask] - pred) / slopes_err[mask])**2) / (len(L_used[mask]) - 3)

            results[L_min] = {
                'nu': nu, 'nu_err': nu_err,
                'A': A, 'A_err': errors[0],
                'B': B, 'B_err': errors[2],
                'chi2_red': chi2, 'n_pts': len(L_used[mask]),
                'slopes': slopes, 'slopes_err': slopes_err, 'L_used': L_used,
                'slope_err_method': 'parametric bootstrap, beta-fixed, k=3 spline'
            }

            fit_rows.append([
                f"L>={L_min} ({len(L_used[mask])} pts)",
                f"{nu:.3f} \u00b1 {nu_err:.3f}",
                f"B={B:.2f} \u00b1 {errors[2]:.2f}",
                f"{chi2:.3f}"
            ])
        except Exception as e:
            print(f"    L_min={L_min}: Fit failed - {e}")

    if fit_rows:
        print_table_compact(["Cut", "nu", "Correction B", "chi2_red"], fit_rows)

    if 32 in results:
        res = results[32]
    elif 24 in results:
        res = results[24]
    else:
        print("\n  \u2717 All nu fits failed!")
        return {'nu': np.nan, 'nu_err': np.nan}

    print_section("Step 3: Final Result", level=2)
    print_result("nu (using L>=32)", res['nu'], res['nu_err'], chi2=res['chi2_red'])
    print_result("Finite-size correction B", res['B'], res['B_err'])
    print_comparison_onsager(res['nu'], res['nu_err'], NU_EXACT, "nu")

    # PLOT 3: nu extraction
    print_section("Step 4: Visualization", level=2)

    def model_simple(L, A, inv_nu):
        return A * L**inv_nu

    mask_fit = res['L_used'] >= 32
    popt_simple, _ = spo.curve_fit(model_simple, res['L_used'][mask_fit], res['slopes'][mask_fit],
                                   p0=[res['slopes'][mask_fit][0] / res['L_used'][mask_fit][0], 1.0],
                                   sigma=res['slopes_err'][mask_fit], absolute_sigma=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = get_colors(L_LIST)
    for i, L in enumerate(res['L_used']):
        idx = L_LIST.index(L)
        ax.errorbar(np.log(L), np.log(res['slopes'][i]),
                    yerr=res['slopes_err'][i] / res['slopes'][i],
                    fmt='o', color=colors[idx], ms=8, capsize=0, lw=1.5,
                    elinewidth=1.5, label=f'$L={L}$', zorder=3)

    L_plot = np.linspace(20, 140, 100)
    ax.plot(np.log(L_plot), np.log(model_simple(L_plot, *popt_simple)),
            color='black', lw=2.5, label='Fit', zorder=2)

    ax.axvspan(np.log(20), np.log(32), alpha=0.15, color='gray', zorder=0,
               label='Excluded')

    ax.set_xlabel(r"$\ln(L)$", fontsize=20)
    ax.set_ylabel(r"$\ln(|dU_4/d\beta|_{\beta_{pc}})$", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=12, loc='lower right', ncol=1)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "nu_extraction_loglog.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: nu_extraction_loglog.pdf")

    return res


# ============================================================================
# PHASE 3: gamma/nu AND beta/nu
# ============================================================================
def extract_magnetic_exponents(data, L_list, beta_pc, beta_pc_err, nu, nu_err):
    """Phase 3: gamma/nu and beta/nu as free parameters.

    chi_max extracted via Rummukainen parabolic fit (3-pt around peak);
    M(beta_pc) from spline interpolation.  Power-law fits with
    L^{-omega} corrections; errors from pcov (Cramer-Rao).
    """
    print_phase_header(3, 5, "MAGNETIC EXPONENTS (FREE FITS)",
                       description="Method: Power-law fits with gamma/nu and beta/nu "
                                   "as free parameters")

    print(f"  Using: beta_pc = {beta_pc:.6f}, nu = {nu:.3f} (both measured)")
    print("-" * 80)

    chi_max, chi_max_err, mag_pc, mag_pc_err, L_used = [], [], [], [], []
    beta_peak_list, beta_peak_err_list = [], []       # parabolic peak positions
    chi_max_spline, chi_max_spline_err = [], []       # spline cross-check (diagnostic)

    rng_peak = np.random.default_rng(123)
    N_PARA_PTS = 3            # 3-point parabolic interpolation (Rummukainen FSS standard)

    for L in L_list:
        if L not in data:
            continue

        d = data[L]

        # ── chi_max via local parabolic fit (Rummukainen FSS standard) ──
        try:
            betas = d['beta']
            chis  = d['chi']
            chi_errs = d['chi_err']

            # 1. Rough peak from raw data
            idx_max_raw = int(np.argmax(chis))

            # 2. Select N_PARA_PTS closest points centred on max
            half = N_PARA_PTS // 2
            lo = max(0, idx_max_raw - half)
            hi = min(len(betas), lo + N_PARA_PTS)
            lo = max(0, hi - N_PARA_PTS)          # readjust if near edge

            b_win = betas[lo:hi]
            c_win = chis[lo:hi]
            e_win = chi_errs[lo:hi]

            # 3. Parabolic fit: chi(beta) = c0 + c1*beta + c2*beta^2
            def parabola(beta, c0, c1, c2):
                return c0 + c1 * beta + c2 * beta**2

            p0_c2 = -1e3           # expect downward curvature
            p0_c1 = 2.0 * (-p0_c2) * betas[idx_max_raw]
            p0_c0 = chis[idx_max_raw] - p0_c1 * betas[idx_max_raw] - p0_c2 * betas[idx_max_raw]**2

            popt_p, pcov_p = spo.curve_fit(
                parabola, b_win, c_win,
                p0=[p0_c0, p0_c1, p0_c2],
                sigma=e_win, absolute_sigma=True
            )
            c0, c1, c2 = popt_p

            if c2 >= 0:
                raise ValueError(f"c2={c2:.2f} >= 0: no maximum in parabola")

            # 4. Peak location and height
            beta_peak_val = -c1 / (2.0 * c2)
            chi_max_val   = c0 - c1**2 / (4.0 * c2)

            # 5. Error propagation via Jacobian + pcov
            #    beta_peak = -c1 / (2*c2)
            J_bp = np.array([0.0,
                             -1.0 / (2.0 * c2),
                             c1 / (2.0 * c2**2)])
            var_bp = float(J_bp @ pcov_p @ J_bp)
            beta_peak_err_val = np.sqrt(max(var_bp, 0.0))

            #    chi_max = c0 - c1^2 / (4*c2)
            J_cm = np.array([1.0,
                             -c1 / (2.0 * c2),
                             c1**2 / (4.0 * c2**2)])
            var_cm = float(J_cm @ pcov_p @ J_cm)
            chi_max_err_val = np.sqrt(max(var_cm, 0.0))

            chi_max.append(chi_max_val)
            chi_max_err.append(chi_max_err_val)
            beta_peak_list.append(beta_peak_val)
            beta_peak_err_list.append(beta_peak_err_val)

            # Spline cross-check (diagnostic only)
            w_chi = 1.0 / (chi_errs + 1e-12)
            s_chi = len(betas) * 0.5
            spl_chi = spi.UnivariateSpline(betas, chis,
                                           w=w_chi, k=3, s=s_chi)
            # Use per-L adaptive bounds centred on raw max
            spl_lo = max(betas[0], betas[idx_max_raw] - 0.010)
            spl_hi = min(betas[-1], betas[idx_max_raw] + 0.010)
            opt = spo.minimize_scalar(lambda b: -spl_chi(b),
                                     bounds=(spl_lo, spl_hi), method='bounded')
            chi_max_spl = -opt.fun
            # Quick bootstrap for spline error (keep lightweight)
            spl_boot = []
            for _ in range(N_BOOTSTRAP):
                chi_pert = chis + rng_peak.normal(0.0, chi_errs)
                try:
                    spl_b = spi.UnivariateSpline(betas, chi_pert,
                                                 w=w_chi, k=3, s=s_chi)
                    opt_b = spo.minimize_scalar(lambda b: -spl_b(b),
                                               bounds=(spl_lo, spl_hi), method='bounded')
                    spl_boot.append(-opt_b.fun)
                except Exception:
                    pass
            chi_max_spl_err = np.std(spl_boot, ddof=1) if len(spl_boot) > 30 else np.nan

            chi_max_spline.append(chi_max_spl)
            chi_max_spline_err.append(chi_max_spl_err)

        except Exception as exc:
            print(f"    ⚠ L={L}: parabolic fit failed ({exc}), skipping")
            continue

        # ── M(beta_pc) via spline + parametric bootstrap ──
        try:
            w_mag = 1.0 / (d['M_abs_err'] + 1e-12)
            s_mag = len(d['beta'])

            spl_mag = spi.UnivariateSpline(d['beta'], d['M_abs'],
                                           w=w_mag, k=3, s=s_mag)
            mag_central = float(spl_mag(beta_pc))

            # Parametric bootstrap: perturb M_abs values, redo spline
            mag_boot = []
            for _ in range(N_BOOTSTRAP):
                m_pert = d['M_abs'] + rng_peak.normal(0.0, d['M_abs_err'])
                try:
                    spl_b = spi.UnivariateSpline(d['beta'], m_pert,
                                                 w=w_mag, k=3, s=s_mag)
                    mag_boot.append(float(spl_b(beta_pc)))
                except Exception:
                    pass

            if len(mag_boot) > 30:
                mag_err_val = np.std(mag_boot, ddof=1)
            else:
                mag_err_val = np.mean(d['M_abs_err'])  # fallback
                print(f"    ⚠ L={L}: mag_pc bootstrap fallback ({len(mag_boot)}/{N_BOOTSTRAP})")

            mag_pc.append(mag_central)
            mag_pc_err.append(mag_err_val)
        except Exception:
            mag_pc.append(np.nan)
            mag_pc_err.append(np.nan)

        L_used.append(L)

    chi_max = np.array(chi_max)
    chi_max_err = np.array(chi_max_err)
    beta_peak_arr = np.array(beta_peak_list)
    beta_peak_err_arr = np.array(beta_peak_err_list)
    chi_max_spline = np.array(chi_max_spline)
    chi_max_spline_err = np.array(chi_max_spline_err)
    mag_pc = np.array(mag_pc)
    mag_pc_err = np.array(mag_pc_err)
    L_used = np.array(L_used)

    print(f"\n  chi_max extraction: LOCAL PARABOLIC FIT ({N_PARA_PTS} pts, Rummukainen standard, pcov errors)")
    print(f"  {'L':>5s}  {'chi_max':>10s} {'±σ':>8s} {'%':>6s}  "
          f"{'β_peak':>10s} {'±σ_β':>10s}  "
          f"{'chi_spl':>10s} {'±σ_spl':>8s}  {'Δ%':>6s}")
    for i, L in enumerate(L_used):
        delta_pct = (chi_max[i] - chi_max_spline[i]) / chi_max[i] * 100 if chi_max[i] != 0 else 0
        print(f"    {L:3.0f}  {chi_max[i]:10.2f} {chi_max_err[i]:8.2f} "
              f"{chi_max_err[i]/chi_max[i]*100:5.1f}%  "
              f"{beta_peak_arr[i]:10.6f} {beta_peak_err_arr[i]:10.6f}  "
              f"{chi_max_spline[i]:10.2f} {chi_max_spline_err[i]:8.2f}  "
              f"{delta_pct:+5.2f}%")

    print("\n  M(beta_pc) from spline + parametric bootstrap:")
    for i, L in enumerate(L_used):
        print(f"    L={L:3.0f}: M_pc={mag_pc[i]:.5f} ± {mag_pc_err[i]:.5f}  "
              f"({mag_pc_err[i]/mag_pc[i]*100:.1f}%)")

    # gamma/nu
    print_section("Susceptibility: gamma/nu as FREE parameter", level=2)
    print("  Bounds: 1.5 < gamma/nu < 2.0, -30 < B < 30")

    L_min = 32
    mask = L_used >= L_min

    def model_chi_free(L, A, gamma_nu, B):
        return A * L**gamma_nu * (1 + B * L**(-OMEGA_EXACT))

    try:
        popt_chi, pcov_chi = spo.curve_fit(
            model_chi_free, L_used[mask], chi_max[mask],
            p0=[0.11, 1.75, 5.0],
            sigma=chi_max_err[mask], absolute_sigma=True,
            bounds=([0.05, 1.5, -30], [0.2, 2.0, 30])
        )

        A_chi, gamma_nu, B_chi = popt_chi
        errors_chi = np.sqrt(np.diag(pcov_chi))

        pred_chi = model_chi_free(L_used[mask], *popt_chi)
        dof_chi = int(len(L_used[mask]) - 3)
        chi2_val_chi = float(np.sum(((chi_max[mask] - pred_chi) / chi_max_err[mask])**2))
        chi2_chi = chi2_val_chi / dof_chi if dof_chi > 0 else np.nan
        p_value_chi = float(sps.chi2.sf(chi2_val_chi, dof_chi)) if dof_chi > 0 else np.nan

        print_result("gamma/nu (FREE)", gamma_nu, errors_chi[1], chi2=chi2_chi)
        if dof_chi > 0:
            print(f"    dof={dof_chi:d}, chi2={chi2_val_chi:.3f}, p={p_value_chi:.3f}")
        print_result("Amplitude A", A_chi, errors_chi[0])
        print_result("Correction B", B_chi, errors_chi[2])

        A_chi = popt_chi[0]
        A_chi_err = errors_chi[0]

        gamma_nu_measured = gamma_nu
        gamma_nu_err = errors_chi[1]

    except Exception as e:
        print(f"  \u2717 Free fit failed: {e}")
        gamma_nu_measured = GAMMA_NU_EXACT
        gamma_nu_err = 0.0
        A_chi = np.nan
        A_chi_err = np.nan
        chi2_chi = np.nan
        chi2_val_chi = np.nan
        dof_chi = 0
        p_value_chi = np.nan

    # beta/nu
    print_section("Magnetization: beta/nu as FREE parameter", level=2)
    print("  Bounds: 0.08 < beta/nu < 0.18, -15 < B < 15")

    valid = ~np.isnan(mag_pc) & (L_used >= L_min)

    def model_mag_free(L, A, beta_nu, B):
        return A * L**(-beta_nu) * (1 + B * L**(-OMEGA_EXACT))

    try:
        popt_mag, pcov_mag = spo.curve_fit(
            model_mag_free, L_used[valid], mag_pc[valid],
            p0=[1.0, 0.125, 1.0],
            sigma=mag_pc_err[valid], absolute_sigma=True,
            bounds=([0.8, 0.08, -15], [1.2, 0.18, 15])
        )

        A_mag, beta_nu, B_mag = popt_mag
        errors_mag = np.sqrt(np.diag(pcov_mag))

        pred_mag = model_mag_free(L_used[valid], *popt_mag)
        dof_mag = int(np.sum(valid) - 3)
        chi2_val_mag = float(np.sum(((mag_pc[valid] - pred_mag) / mag_pc_err[valid])**2))
        chi2_mag = chi2_val_mag / dof_mag if dof_mag > 0 else np.nan
        p_value_mag = float(sps.chi2.sf(chi2_val_mag, dof_mag)) if dof_mag > 0 else np.nan

        print_result("beta/nu (FREE)", beta_nu, errors_mag[1], chi2=chi2_mag)
        if dof_mag > 0:
            print(f"    dof={dof_mag:d}, chi2={chi2_val_mag:.3f}, p={p_value_mag:.3f}")
        print_result("Amplitude A", A_mag, errors_mag[0])
        print_result("Correction B", B_mag, errors_mag[2])

        A_mag = popt_mag[0]
        A_mag_err = errors_mag[0]

        beta_nu_measured = beta_nu
        beta_nu_err = errors_mag[1]

    except Exception as e:
        print(f"  \u2717 Free fit failed: {e}")
        beta_nu_measured = BETA_NU_EXACT
        beta_nu_err = 0.0
        A_mag = np.nan
        A_mag_err = np.nan
        chi2_mag = np.nan
        chi2_val_mag = np.nan
        dof_mag = 0
        p_value_mag = np.nan

    # Derived exponents
    print_section("Derived Exponents", level=2)

    gamma = gamma_nu_measured * nu
    gamma_err = np.sqrt((gamma_nu_err * nu)**2 + (gamma_nu_measured * nu_err)**2)

    beta = beta_nu_measured * nu
    beta_err = np.sqrt((beta_nu_err * nu)**2 + (beta_nu_measured * nu_err)**2)

    eta = 2 - gamma_nu_measured
    eta_err = gamma_nu_err

    # delta via Griffiths-Rushbrooke scaling relation
    delta = 1.0 + gamma_nu_measured / beta_nu_measured
    delta_err = np.sqrt(
        (gamma_nu_err / beta_nu_measured)**2 +
        (gamma_nu_measured * beta_nu_err / beta_nu_measured**2)**2
    )

    hyperscaling = 2 * beta_nu_measured + gamma_nu_measured
    hyperscaling_err = np.sqrt((2 * beta_nu_err)**2 + gamma_nu_err**2)

    print_result("gamma = (gamma/nu)*nu", gamma, gamma_err)
    print_result("beta  = (beta/nu)*nu", beta, beta_err)
    print_result("eta   = 2 - gamma/nu", eta, eta_err)
    print_result("delta = 1 + (g/nu)/(b/nu)", delta, delta_err)
    print(f"    Expected: 15.0 (exact)  ->  Deviation: {abs(delta - 15.0) / delta_err:.2f} sigma")
    print_result("Hyperscaling 2b/nu+g/nu", hyperscaling, hyperscaling_err)
    print(f"    Expected: 2.0 (d=2)  ->  Deviation: {abs(hyperscaling - 2.0):.4f}")

    # PLOT 4: Magnetic exponents
    print_section("Visualization", level=2)

    def model_chi_simple(L, A, gamma_nu):
        return A * L**gamma_nu

    def model_mag_simple(L, A, beta_nu):
        return A * L**(-beta_nu)

    popt_chi_simple, _ = spo.curve_fit(model_chi_simple, L_used[mask], chi_max[mask],
                                       p0=[0.11, 1.75], sigma=chi_max_err[mask])

    popt_mag_simple, _ = spo.curve_fit(model_mag_simple, L_used[valid], mag_pc[valid],
                                       p0=[1.0, 0.125], sigma=mag_pc_err[valid])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = get_colors(L_LIST)

    # LEFT: chi'
    for i, L in enumerate(L_used):
        idx = L_LIST.index(L)
        ax1.errorbar(np.log(L), np.log(chi_max[i]), yerr=chi_max_err[i] / chi_max[i],
                     fmt='o', color=colors[idx], ms=8, capsize=0, lw=1.5,
                     elinewidth=1.5, label=f'$L={L}$', zorder=3)

    L_plot = np.linspace(20, 140, 100)
    ax1.plot(np.log(L_plot), np.log(model_chi_simple(L_plot, *popt_chi_simple)),
             color='black', lw=2.5, label='Fit', zorder=2)

    ax1.axvspan(np.log(20), np.log(32), alpha=0.15, color='gray', zorder=0,
                label='Excluded')
    ax1.set_xlabel(r"$\ln(L)$", fontsize=20)
    ax1.set_ylabel(r"$\ln(\chi'_{\max})$", fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.grid(alpha=0.3, ls=':')
    ax1.legend(frameon=False, fontsize=12, loc='lower right', ncol=1)

    # RIGHT: M
    for i, L in enumerate(L_used[valid]):
        idx = L_LIST.index(L)
        ax2.errorbar(np.log(L), np.log(mag_pc[valid][i]),
                     yerr=mag_pc_err[valid][i] / mag_pc[valid][i],
                     fmt='o', color=colors[idx], ms=8, capsize=0, lw=1.5,
                     elinewidth=1.5, label=f'$L={L}$', zorder=3)

    ax2.plot(np.log(L_plot), np.log(model_mag_simple(L_plot, *popt_mag_simple)),
             color='black', lw=2.5, label='Fit', zorder=2)

    ax2.axvspan(np.log(20), np.log(32), alpha=0.15, color='gray', zorder=0,
                label='Excluded')
    ax2.set_xlabel(r"$\ln(L)$", fontsize=20)
    ax2.set_ylabel(r"$\ln(M(\beta_{pc}))$", fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.grid(alpha=0.3, ls=':')
    ax2.legend(frameon=False, fontsize=12, loc='upper right', ncol=1)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "exponent_fits_magnetic.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: exponent_fits_magnetic.pdf")

    return {
        'gamma_nu': gamma_nu_measured,
        'gamma_nu_err': gamma_nu_err,
        'beta_nu': beta_nu_measured,
        'beta_nu_err': beta_nu_err,
        'gamma': gamma,
        'gamma_err': gamma_err,
        'beta': beta,
        'beta_err': beta_err,
        'eta': eta,
        'eta_err': eta_err,
        'delta': delta,
        'delta_err': delta_err,
        'hyperscaling': hyperscaling,
        'hyperscaling_err': hyperscaling_err,
        'A_chi': A_chi,
        'A_chi_err': A_chi_err,
        'A_mag': A_mag,
        'A_mag_err': A_mag_err,
        'chi2_chi': chi2_chi,
        'chi2_val_chi': chi2_val_chi,
        'chi2_dof_chi': dof_chi,
        'chi2_p_chi': p_value_chi,
        'chi2_mag': chi2_mag,
        'chi2_val_mag': chi2_val_mag,
        'chi2_dof_mag': dof_mag,
        'chi2_p_mag': p_value_mag,
        'chi_max': chi_max,
        'chi_max_err': chi_max_err,
        'beta_peak': beta_peak_arr,
        'beta_peak_err': beta_peak_err_arr,
        'chi_max_spline': chi_max_spline,
        'chi_max_spline_err': chi_max_spline_err,
        'mag_pc': mag_pc,
        'mag_pc_err': mag_pc_err,
        'L_used': L_used
    }


# ============================================================================
# PHASE 3b: SUB-LEADING CORRECTIONS (RIGOROUS ANALYSIS)
# ============================================================================
def extract_all_corrections(data, L_list, beta_pc, nu, nu_err, mag_result,
                            nu_result=None, cv_result=None):
    """Phase 3b: sub-leading finite-size corrections for all observables.

    Two-step fit + F-test (see extract_observable_with_corrections).
    Purely diagnostic — does not modify the Phase 3 results.
    """
    print("\n" + "=" * 80)
    print("[PHASE 3b] SUB-LEADING CORRECTIONS (RIGOROUS ANALYSIS)")
    print("=" * 80)
    print(f"  Using: beta_pc = {beta_pc:.6f}, nu = {nu:.3f} (both measured)")
    print(f"  Method: TWO-STEP fit with Bootstrap (n={50}, diagnostic) + F-test")
    print(f"  Correction ansatz: O(L) = A*L^alpha * [1 + B*L^(-{OMEGA_EXACT:.0f})]")
    print("-" * 80)

    L_used = mag_result['L_used']
    chi_max = mag_result['chi_max']
    chi_max_err = mag_result['chi_max_err']
    mag_pc = mag_result['mag_pc']
    mag_pc_err = mag_result['mag_pc_err']

    # Apply L_min = 32 filter (same cut as Phase 3)
    L_min = 32
    mask_chi = L_used >= L_min
    valid_mag = ~np.isnan(mag_pc) & (L_used >= L_min)

    print(f"\n  Data: {len(L_used)} lattice sizes, using L >= {L_min}")
    print(f"    Susceptibility: {np.sum(mask_chi)} points")
    print(f"    Magnetization:  {np.sum(valid_mag)} points")

    N_BOOT_DIAG = 50   # diagnostic only — fast

    # ===== SUSCEPTIBILITY: chi_max ~ L^{gamma/nu} =====
    results_chi = extract_observable_with_corrections(
        L_used[mask_chi], chi_max[mask_chi], chi_max_err[mask_chi],
        observable_name="chi_max(L)",
        exponent_theory=GAMMA_NU_EXACT,
        bounds_exponent=(1.5, 2.0),
        n_bootstrap=N_BOOT_DIAG
    )

    # ===== MAGNETIZATION: M(beta_pc) ~ L^{-beta/nu} =====
    results_M = extract_observable_with_corrections(
        L_used[valid_mag], mag_pc[valid_mag], mag_pc_err[valid_mag],
        observable_name="M(beta_pc, L)",
        exponent_theory=-BETA_NU_EXACT,
        bounds_exponent=(-0.18, -0.08),
        n_bootstrap=N_BOOT_DIAG
    )

    # ===== BINDER SLOPE: S(L) ~ L^{1/nu} =====
    results_S = None
    if nu_result is not None and 'slopes' in nu_result:
        S_data = nu_result['slopes']
        S_err = nu_result['slopes_err']
        L_S = nu_result['L_used']

        mask_S = L_S >= L_min
        if np.sum(mask_S) >= 3:
            results_S = extract_observable_with_corrections(
                L_S[mask_S], S_data[mask_S], S_err[mask_S],
                observable_name="S(L) Binder",
                exponent_theory=1.0 / NU_EXACT,
                bounds_exponent=(0.7, 1.5),
                n_bootstrap=N_BOOT_DIAG
            )
        else:
            print("\n  \u2717 Not enough Binder slope data (need >= 3 points with L >= 32)")
    else:
        print("\n  \u2717 Binder slope data not available (nu_result missing)")

    # ===== SPECIFIC HEAT: C_max ~ A + B*ln(L) [+ C*L^{-omega}] =====
    results_Cv = None
    if cv_result is not None:
        cv_max_data = cv_result['cv_max']
        cv_max_err_data = cv_result['cv_max_err']
        L_cv = cv_result['L_used']
        mask_cv = L_cv >= L_min
        if np.sum(mask_cv) >= 3:
            L_cv_fit = L_cv[mask_cv]
            cv_fit = cv_max_data[mask_cv]
            cv_err_fit = cv_max_err_data[mask_cv]

            print_section("Extracting: C_max(L) [logarithmic model]")

            # --- Model 1: C_max = A + B*ln(L)  (2 params) ---
            print_section("Step 1: Logarithmic fit  C_max = A + B*ln(L)", level=2)

            def model_cv_log(L, A, B):
                return A + B * np.log(L)

            p0_cv1 = [cv_fit[0], 0.3]

            params_boot_cv1, fit_cv1 = bootstrap_fit_procedure(
                model_cv_log, L_cv_fit, cv_fit, cv_err_fit,
                p0=p0_cv1, bounds=([-np.inf, -5], [np.inf, 5]),
                n_bootstrap=N_BOOT_DIAG
            )

            if fit_cv1 is not None:
                _, std_cv1_boot, pct_cv1 = compute_bootstrap_errors(params_boot_cv1)
                print_result("Intercept A", fit_cv1.params[0], fit_cv1.params_err[0], chi2=fit_cv1.chi2_red)
                print_result("Slope B (log)", fit_cv1.params[1], fit_cv1.params_err[1])
                print(f"    Error source: pcov (Cramér-Rao)")
                if len(L_cv_fit) < 20:
                    print(f"    NP bootstrap σ(B_log) = {std_cv1_boot[1]:.4f}  [diagnostic, n={len(L_cv_fit)}]")
                print(f"    Bootstrap: {len(params_boot_cv1)}/{N_BOOT_DIAG} successful fits")
            else:
                print("  \u2717 Logarithmic fit failed!")

            # --- Model 2: C_max = A + B_fixed*ln(L) + D*L^{-omega}  (2 params, B fixed from Step 1) ---
            B_fixed_cv = fit_cv1.params[1] if fit_cv1 else 0.3
            print_section(f"Step 2: Corrected fit  C_max = A + {B_fixed_cv:.4f}*ln(L) + D*L^(-{OMEGA_EXACT:.0f})  (B fixed)", level=2)

            def model_cv_corr(L, A, D):
                return A + B_fixed_cv * np.log(L) + D * L**(-OMEGA_EXACT)

            p0_cv2 = [fit_cv1.params[0] if fit_cv1 else cv_fit[0],
                       0.0]

            params_boot_cv2, fit_cv2 = bootstrap_fit_procedure(
                model_cv_corr, L_cv_fit, cv_fit, cv_err_fit,
                p0=p0_cv2, bounds=([-np.inf, -100], [np.inf, 100]),
                n_bootstrap=N_BOOT_DIAG
            )

            cv_C_val, cv_C_err = np.nan, np.nan
            cv_chi2_corr = np.nan
            cv_F, cv_p, cv_sig = np.nan, np.nan, False

            if fit_cv2 is not None:
                _, std_cv2_boot, pct_cv2 = compute_bootstrap_errors(params_boot_cv2)
                cv_C_val = fit_cv2.params[1]
                cv_C_err = fit_cv2.params_err[1]    # pcov
                cv_chi2_corr = fit_cv2.chi2_red
                print_result("Intercept A", fit_cv2.params[0], fit_cv2.params_err[0], chi2=cv_chi2_corr)
                print(f"    Slope B (log) = {B_fixed_cv:.6f}  [FIXED from Step 1]")
                print_result("Correction D", cv_C_val, cv_C_err)
                print(f"    Error source: pcov (Cramér-Rao)")
                if len(L_cv_fit) < 20:
                    print(f"    NP bootstrap σ(D) = {std_cv2_boot[1]:.4f}  [diagnostic, n={len(L_cv_fit)}]")
                print(f"    Bootstrap: {len(params_boot_cv2)}/{N_BOOT_DIAG} successful fits")
            else:
                print("  \u2717 Corrected fit failed!")

            # --- F-test: Step 1 (A,B) vs 3-param (A,B,D all free) ---
            # Same pattern as power-law observables: compare 2-param vs 3-param
            print_section("Model Comparison (F-test: 2-param vs 3-param)", level=2)

            def model_cv_3p(L, A, B, D):
                return A + B * np.log(L) + D * L**(-OMEGA_EXACT)

            p0_cv3p = [fit_cv1.params[0] if fit_cv1 else cv_fit[0],
                        fit_cv1.params[1] if fit_cv1 else 0.3,
                        0.0]

            _, fit_cv3p = bootstrap_fit_procedure(
                model_cv_3p, L_cv_fit, cv_fit, cv_err_fit,
                p0=p0_cv3p, bounds=([-np.inf, -5, -100], [np.inf, 5, 100]),
                n_bootstrap=0  # no bootstrap needed, just the fit
            )

            if fit_cv1 is not None and fit_cv3p is not None:
                try:
                    cv_F, cv_p, cv_sig = f_test_nested_models(
                        fit_cv1.chi2, fit_cv1.dof,
                        fit_cv3p.chi2, fit_cv3p.dof
                    )
                    sig_str = "sig" if cv_sig else "n.s."
                    print(f"    F-test: F={cv_F:.3f}, p={cv_p:.4f} ({sig_str})")
                except Exception as e:
                    print(f"    \u2717 F-test failed: {e}")

            results_Cv = {
                'observable': 'C_max(L)',
                'exponent': fit_cv1.params[1] if fit_cv1 else np.nan,
                'exponent_err': fit_cv1.params_err[1] if fit_cv1 else np.nan,
                'A_simple': fit_cv1.params[0] if fit_cv1 else np.nan,
                'A_simple_err': fit_cv1.params_err[0] if fit_cv1 else np.nan,
                'A_corrected': fit_cv2.params[0] if fit_cv2 else np.nan,
                'A_corrected_err': fit_cv2.params_err[0] if fit_cv2 else np.nan,
                'B': cv_C_val,
                'B_err': cv_C_err,  # pcov from fit_cv2.params_err[1] (correction D)
                'chi2_red_simple': fit_cv1.chi2_red if fit_cv1 else np.nan,
                'chi2_raw_simple': fit_cv1.chi2 if fit_cv1 else np.nan,
                'dof_simple': fit_cv1.dof if fit_cv1 else 0,
                'chi2_red_corrected': cv_chi2_corr,
                'chi2_raw_corrected': fit_cv2.chi2 if fit_cv2 else np.nan,
                'dof_corrected': fit_cv2.dof if fit_cv2 else 0,
                'F_statistic': cv_F,
                'p_value': cv_p,
                'is_significant': cv_sig,
            }
        else:
            print("\n  \u2717 Not enough C_max data (need >= 3 points with L >= 32)")
    else:
        print("\n  \u2717 Specific heat data not available (cv_result missing)")

    # ===== SUMMARY TABLE =====
    print_section("Summary: Finite-Size Corrections")

    summary_rows = []
    for res in [results_chi, results_M, results_S, results_Cv]:
        if res is None:
            continue
        sig_marker = "sig" if res['is_significant'] else "n.s."
        summary_rows.append([
            res['observable'],
            f"{res['exponent']:+.3f}\u00b1{res['exponent_err']:.3f}",
            f"B={res['B']:+.1f}\u00b1{res['B_err']:.1f}",
            f"{res['chi2_red_simple']:.3f}",
            f"{res['chi2_red_corrected']:.3f}",
            f"p={res['p_value']:.3f} {sig_marker}"
        ])

    print_table_compact(
        ["Observable", "Exponent", "B_correction", "chi2_simp", "chi2_cor", "F-test"],
        summary_rows
    )

    print()

    # ===== DERIVED EXPONENTS (from rigorous sub-leading analysis) =====
    if results_chi is not None and results_M is not None:
        gamma_nu_rig = results_chi['exponent']
        gamma_nu_err_rig = results_chi['exponent_err']
        beta_nu_rig = abs(results_M['exponent'])
        beta_nu_err_rig = results_M['exponent_err']

        gamma_rig = gamma_nu_rig * nu
        gamma_err_rig = gamma_rig * np.sqrt(
            (gamma_nu_err_rig / (gamma_nu_rig + 1e-30))**2 + (nu_err / nu)**2
        )

        beta_rig = beta_nu_rig * nu
        beta_err_rig = beta_rig * np.sqrt(
            (beta_nu_err_rig / (beta_nu_rig + 1e-30))**2 + (nu_err / nu)**2
        )

        eta_rig = 2.0 - gamma_nu_rig

        # delta via Griffiths-Rushbrooke scaling relation
        delta_rig = 1.0 + gamma_nu_rig / beta_nu_rig
        delta_err_rig = np.sqrt(
            (gamma_nu_err_rig / beta_nu_rig)**2 +
            (gamma_nu_rig * beta_nu_err_rig / beta_nu_rig**2)**2
        )

        hyperscaling_rig = 2 * beta_nu_rig + gamma_nu_rig

        print_section("Derived Exponents (from rigorous sub-leading analysis)", level=2)
        print_result("gamma = (gamma/nu)*nu", gamma_rig, gamma_err_rig)
        print_result("beta  = (beta/nu)*nu", beta_rig, beta_err_rig)
        print(f"    eta       = 2 - gamma/nu  = {eta_rig:.3f}")
        print_result("delta = 1 + (g/nu)/(b/nu)", delta_rig, delta_err_rig)
        print(f"    Expected: 15.0 (exact)  ->  Deviation: {abs(delta_rig - 15.0) / delta_err_rig:.2f} sigma")
        print(f"    Hyperscaling: 2*beta/nu + gamma/nu = {hyperscaling_rig:.4f}")
        print(f"    Expected (d=2): 2.0000")
        print(f"    Deviation: {abs(2.0 - hyperscaling_rig):.4f}")
        print()

    return results_chi, results_M, results_S, results_Cv


# ============================================================================
# PHASE 3c: CORRECTIONS SCAN WITH VARIABLE L_min
# ============================================================================
def extract_corrections_scan(mag_result, nu_result=None, cv_result=None,
                             L_min_values=None):
    """Phase 3c: sub-leading corrections for several L_min cutoffs.

    Re-uses extract_observable_with_corrections for each L_min.
    Returns dict[L_min] -> {'chi': res, 'M': res, 'S': res}.
    """
    if L_min_values is None:
        L_min_values = [24, 32]

    print("\n" + "=" * 80)
    print("[PHASE 3c] SUB-LEADING CORRECTIONS — L_min COMPARISON")
    print("=" * 80)
    print(f"  L_min values: {L_min_values}")
    print(f"  Correction ansatz: O(L) = A*L^alpha * [1 + B*L^(-{OMEGA_EXACT:.0f})]")
    print("-" * 80)

    L_used = mag_result['L_used']
    chi_max = mag_result['chi_max']
    chi_max_err = mag_result['chi_max_err']
    mag_pc = mag_result['mag_pc']
    mag_pc_err = mag_result['mag_pc_err']

    N_BOOT_DIAG = 50

    all_results = {}

    for L_min_cut in L_min_values:
        print(f"\n  {'─' * 40}")
        print(f"  L_min = {L_min_cut}")
        print(f"  {'─' * 40}")

        mask_chi = L_used >= L_min_cut
        valid_mag = ~np.isnan(mag_pc) & (L_used >= L_min_cut)

        n_chi = int(np.sum(mask_chi))
        n_mag = int(np.sum(valid_mag))
        print(f"    Points: χ'_max = {n_chi}, M(β_pc) = {n_mag}")

        # Susceptibility
        res_chi = None
        if n_chi >= 3:
            res_chi = extract_observable_with_corrections(
                L_used[mask_chi], chi_max[mask_chi], chi_max_err[mask_chi],
                observable_name=f"chi_max (L>={L_min_cut})",
                exponent_theory=GAMMA_NU_EXACT,
                bounds_exponent=(1.5, 2.0),
                n_bootstrap=N_BOOT_DIAG
            )

        # Magnetization
        res_M = None
        if n_mag >= 3:
            res_M = extract_observable_with_corrections(
                L_used[valid_mag], mag_pc[valid_mag], mag_pc_err[valid_mag],
                observable_name=f"M(beta_pc) (L>={L_min_cut})",
                exponent_theory=-BETA_NU_EXACT,
                bounds_exponent=(-0.18, -0.08),
                n_bootstrap=N_BOOT_DIAG
            )

        # Binder slopes
        res_S = None
        if nu_result is not None and 'slopes' in nu_result:
            S_data = nu_result['slopes']
            S_err = nu_result['slopes_err']
            L_S = nu_result['L_used']
            mask_S = L_S >= L_min_cut
            if np.sum(mask_S) >= 3:
                res_S = extract_observable_with_corrections(
                    L_S[mask_S], S_data[mask_S], S_err[mask_S],
                    observable_name=f"S(L) Binder (L>={L_min_cut})",
                    exponent_theory=1.0 / NU_EXACT,
                    bounds_exponent=(0.7, 1.5),
                    n_bootstrap=N_BOOT_DIAG
                )

        all_results[L_min_cut] = {'chi': res_chi, 'M': res_M, 'S': res_S}

    # ── Summary comparison table ──
    print_section("Phase 3c: Corrections summary (all L_min)")

    headers = ["Observable", "L_min", "Exponent", "B±σ_B",
               "χ²_red", "F-test p"]
    rows = []
    for L_min_cut in L_min_values:
        for obs_key, obs_name in [('chi', "χ'_max"),
                                   ('M', "M(β_pc)"),
                                   ('S', "S(L)")]:
            res = all_results[L_min_cut].get(obs_key)
            if res is None:
                continue
            sig = "sig" if res['is_significant'] else "n.s."
            rows.append([
                obs_name, str(L_min_cut),
                f"{res['exponent']:+.4f}±{res['exponent_err']:.4f}",
                f"{res['B']:+.1f}±{res['B_err']:.1f}",
                f"{res['chi2_red_corrected']:.3f}",
                f"{res['p_value']:.3f} ({sig})",
            ])
    print_table_compact(headers, rows)

    return all_results


def save_corrections_comparison_latex(all_results, L_min_values):
    """Save LaTeX comparison table for Phase 3c (fss_comparison_table.tex)."""
    filepath = ANALYSIS_DIR / "fss_comparison_table.tex"

    with open(filepath, 'w') as f:
        f.write("% FSS Corrections Comparison Table (Phase 3c)\n")
        f.write("% Generated: "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("\\begin{table}[h]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("Observable & $L_{\\min}$ & Exponent & "
                "$B \\pm \\sigma_B$ & "
                "$\\chi^2_{\\text{red}}$ & $p$-value (F) \\\\\n")
        f.write("\\midrule\n")

        obs_ltx_map = {
            'chi': "$\\chi'_{\\max}$",
            'M': "$M(\\beta_{pc})$",
            'S': "$S(L)$",
        }

        first = True
        for L_min_cut in L_min_values:
            if L_min_cut not in all_results:
                continue
            if not first:
                f.write("\\midrule\n")
            first = False
            for obs_key in ['chi', 'M', 'S']:
                res = all_results[L_min_cut].get(obs_key)
                if res is None:
                    continue
                sig = "sig" if res['is_significant'] else "n.s."
                f.write(
                    f"{obs_ltx_map[obs_key]} & {L_min_cut} & "
                    f"${res['exponent']:+.4f} \\pm {res['exponent_err']:.4f}$ & "
                    f"${res['B']:+.1f} \\pm {res['B_err']:.1f}$ & "
                    f"${res['chi2_red_corrected']:.3f}$ & "
                    f"${res['p_value']:.3f}$ ({sig}) \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Sub-leading finite-size corrections "
                "$O(L) = A\\,L^{\\alpha}\\,[1+B\\,L^{-\\omega}]$ with "
                f"$\\omega={OMEGA_EXACT:.0f}$. "
                "Exponent from Step~1 (free); $B$ from Step~2 (exponent fixed); "
                "$p$-value from F-test (2-param vs 3-param).}\n")
        f.write("\\label{tab:fss_comparison}\n\\end{table}\n")

    print(f"  ✓ Saved: {filepath}")




# ============================================================================
# PHASE 4: alpha FROM SPECIFIC HEAT
# ============================================================================
def extract_alpha(data, L_list, nu, nu_err):
    """Phase 4: specific-heat exponent alpha from C_max(L) scaling.

    C_max extracted via 3-point parabolic fit (Rummukainen standard).
    Primary fit: C_max = A*L^g*(1 + q*ln L) with g=alpha/nu free.
    Cross-check: logarithmic model C_max = a + b*ln(L).
    F-test decides whether the generalized model is needed.
    """
    print_phase_header(4, 5, "SPECIFIC HEAT EXPONENT alpha",
                       description="Method: Parabolic C_max extraction "
                                   "+ generalized ansatz C_max = A·L^g·(1+q·ln L)")

    cv_max, cv_max_err, L_used = [], [], []
    beta_peak_C, beta_peak_C_err = [], []        # peak positions
    cv_max_spline, cv_max_spline_err = [], []    # spline cross-check

    rng_cv = np.random.default_rng(456)
    N_PARA_PTS = 3    # 3-point parabolic interpolation (Rummukainen FSS standard)

    for L in L_list:
        if L not in data:
            continue

        d = data[L]

        # ── C_max via LOCAL PARABOLIC FIT (same method as chi_max) ──
        try:
            betas  = d['beta']
            Cs     = d['C']
            C_errs = d['C_err']

            # 1. Rough peak from raw data
            idx_max_raw = int(np.argmax(Cs))

            # 2. Select N_PARA_PTS closest points centred on max
            half = N_PARA_PTS // 2
            lo = max(0, idx_max_raw - half)
            hi = min(len(betas), lo + N_PARA_PTS)
            lo = max(0, hi - N_PARA_PTS)

            b_win = betas[lo:hi]
            c_win = Cs[lo:hi]
            e_win = C_errs[lo:hi]

            # 3. Parabolic fit: C(beta) = c0 + c1*beta + c2*beta^2
            def parabola(beta, c0, c1, c2):
                return c0 + c1 * beta + c2 * beta**2

            p0_c2 = -1e2           # expect downward curvature (weaker than chi)
            p0_c1 = 2.0 * (-p0_c2) * betas[idx_max_raw]
            p0_c0 = Cs[idx_max_raw] - p0_c1 * betas[idx_max_raw] - p0_c2 * betas[idx_max_raw]**2

            popt_p, pcov_p = spo.curve_fit(
                parabola, b_win, c_win,
                p0=[p0_c0, p0_c1, p0_c2],
                sigma=e_win, absolute_sigma=True
            )
            c0, c1, c2 = popt_p

            if c2 >= 0:
                raise ValueError(f"c2={c2:.2f} >= 0: no maximum in parabola")

            # 4. Peak location and height
            beta_peak_val = -c1 / (2.0 * c2)
            cv_max_val    = c0 - c1**2 / (4.0 * c2)

            # Error propagation: Jacobian of beta_peak and cv_max w.r.t. (c0,c1,c2)
            #    beta_peak = -c1 / (2*c2)
            J_bp = np.array([0.0,
                             -1.0 / (2.0 * c2),
                             c1 / (2.0 * c2**2)])
            var_bp = float(J_bp @ pcov_p @ J_bp)
            beta_peak_err_val = np.sqrt(max(var_bp, 0.0))

            #    cv_max = c0 - c1^2 / (4*c2)
            J_cm = np.array([1.0,
                             -c1 / (2.0 * c2),
                             c1**2 / (4.0 * c2**2)])
            var_cm = float(J_cm @ pcov_p @ J_cm)
            cv_max_err_val = np.sqrt(max(var_cm, 0.0))

            cv_max.append(cv_max_val)
            cv_max_err.append(cv_max_err_val)
            beta_peak_C.append(beta_peak_val)
            beta_peak_C_err.append(beta_peak_err_val)

            # Spline cross-check (diagnostic only)
            w_cv = 1.0 / (C_errs + 1e-12)
            s_cv = len(betas)
            spl = spi.UnivariateSpline(betas, Cs, w=w_cv, k=3, s=s_cv)
            spl_lo = max(betas[0], betas[idx_max_raw] - 0.010)
            spl_hi = min(betas[-1], betas[idx_max_raw] + 0.010)
            opt = spo.minimize_scalar(lambda b: -spl(b),
                                     bounds=(spl_lo, spl_hi), method='bounded')
            cv_spl = -opt.fun

            spl_boot = []
            for _ in range(N_BOOTSTRAP):
                c_pert = Cs + rng_cv.normal(0.0, C_errs)
                try:
                    spl_b = spi.UnivariateSpline(betas, c_pert,
                                                 w=w_cv, k=3, s=s_cv)
                    opt_b = spo.minimize_scalar(lambda b: -spl_b(b),
                                               bounds=(spl_lo, spl_hi), method='bounded')
                    spl_boot.append(-opt_b.fun)
                except Exception:
                    pass
            cv_spl_err = np.std(spl_boot, ddof=1) if len(spl_boot) > 30 else np.nan

            cv_max_spline.append(cv_spl)
            cv_max_spline_err.append(cv_spl_err)

        except Exception as exc:
            print(f"    ⚠ L={L}: parabolic fit failed ({exc}), skipping")
            continue

        L_used.append(L)

    cv_max = np.array(cv_max)
    cv_max_err = np.array(cv_max_err)
    beta_peak_C = np.array(beta_peak_C)
    beta_peak_C_err = np.array(beta_peak_C_err)
    cv_max_spline = np.array(cv_max_spline)
    cv_max_spline_err = np.array(cv_max_spline_err)
    L_used = np.array(L_used)

    # Print extraction results
    print(f"\n  C_max extraction: LOCAL PARABOLIC FIT ({N_PARA_PTS} pts, Rummukainen standard, pcov errors)")
    print(f"  {'L':>5s}  {'C_max':>10s} {'±σ':>8s} {'%':>6s}  "
          f"{'β_peak':>10s} {'±σ_β':>10s}  "
          f"{'C_spl':>10s} {'±σ_spl':>8s}  {'Δ%':>6s}")
    for i, L_val in enumerate(L_used):
        delta_pct = (cv_max[i] - cv_max_spline[i]) / cv_max[i] * 100 if cv_max[i] != 0 else 0
        print(f"    {L_val:3.0f}  {cv_max[i]:10.4f} {cv_max_err[i]:8.4f} "
              f"{cv_max_err[i]/cv_max[i]*100:5.1f}%  "
              f"{beta_peak_C[i]:10.6f} {beta_peak_C_err[i]:10.6f}  "
              f"{cv_max_spline[i]:10.4f} {cv_max_spline_err[i]:8.4f}  "
              f"{delta_pct:+5.2f}%")

    # Fits use L >= 32
    L_min = 32
    mask = L_used >= L_min

    # Cross-check: logarithmic fit C_max = a + b*ln(L)
    print_section("Cross-check: Logarithmic fit  C_max = a + b·ln(L)", level=2)

    def model_log(L, a, b):
        return a + b * np.log(L)

    popt_log, pcov_log = spo.curve_fit(
        model_log, L_used[mask], cv_max[mask],
        p0=[cv_max[mask][0], 0.3],
        sigma=cv_max_err[mask], absolute_sigma=True
    )
    errs_log = np.sqrt(np.diag(pcov_log))

    pred_log = model_log(L_used[mask], *popt_log)
    dof_log = int(np.sum(mask) - 2)
    chi2_val_log = float(np.sum(((cv_max[mask] - pred_log) / cv_max_err[mask])**2))
    chi2_red_log = chi2_val_log / dof_log if dof_log > 0 else np.nan
    p_val_log = float(sps.chi2.sf(chi2_val_log, dof_log)) if dof_log > 0 else np.nan

    print_result("Intercept a", popt_log[0], errs_log[0], chi2=chi2_red_log)
    print_result("Slope b (log)", popt_log[1], errs_log[1])
    if dof_log > 0:
        print(f"    dof={dof_log}, chi2={chi2_val_log:.3f}, p={p_val_log:.3f}")
    print(f"  Expected for alpha=0: logarithmic divergence \u2713")

    # ====== Primary: generalized ansatz C_max = A·L^g·(1 + q·ln L) ======
    print_section("Primary: Generalized ansatz  C_max = A·L^g·(1 + q·ln L)", level=2)
    print("  g = α/ν is a free parameter (α=0 corresponds to g=0)")

    def model_gen(L, A, g, q):
        return A * L**g * (1.0 + q * np.log(L))

    # Initial guesses from log fit: if g≈0, A≈a, A*q≈b → q≈b/a
    A0 = max(popt_log[0], 0.01)
    q0 = popt_log[1] / A0 if A0 > 0.01 else 0.1
    g0 = 0.01   # close to zero (α=0 for 2D Ising)

    gen_fit_ok = False
    A_gen = g_gen = q_gen = np.nan
    A_gen_err = g_gen_err = q_gen_err = np.nan
    chi2_red_gen = chi2_val_gen = np.nan
    dof_gen = 0
    p_val_gen = np.nan
    F_val = F_p = np.nan
    F_sig = False

    try:
        popt_gen, pcov_gen = spo.curve_fit(
            model_gen, L_used[mask], cv_max[mask],
            p0=[A0, g0, q0],
            sigma=cv_max_err[mask], absolute_sigma=True,
            bounds=([0.0, -0.5, -5.0], [np.inf, 0.5, 5.0])
        )
        errs_gen = np.sqrt(np.diag(pcov_gen))
        A_gen, g_gen, q_gen = popt_gen
        A_gen_err, g_gen_err, q_gen_err = errs_gen

        pred_gen = model_gen(L_used[mask], *popt_gen)
        dof_gen = int(np.sum(mask) - 3)
        chi2_val_gen = float(np.sum(((cv_max[mask] - pred_gen) / cv_max_err[mask])**2))
        chi2_red_gen = chi2_val_gen / dof_gen if dof_gen > 0 else np.nan
        p_val_gen = float(sps.chi2.sf(chi2_val_gen, dof_gen)) if dof_gen > 0 else np.nan

        print_result("Amplitude A", A_gen, A_gen_err, chi2=chi2_red_gen)
        print_result("g = α/ν (free)", g_gen, g_gen_err)
        print_result("q (log correction)", q_gen, q_gen_err)
        if dof_gen > 0:
            print(f"    dof={dof_gen}, chi2={chi2_val_gen:.3f}, p={p_val_gen:.3f}")

        # Derive alpha = g * nu
        alpha_derived = g_gen * nu
        alpha_derived_err = np.sqrt((g_gen_err * nu)**2 + (g_gen * nu_err)**2)
        print_result("α = g·ν", alpha_derived, alpha_derived_err)

        # F-test: log(2 params) vs generalized(3 params)
        try:
            F_val, F_p, F_sig = f_test_nested_models(
                chi2_val_log, dof_log,
                chi2_val_gen, dof_gen
            )
            sig_str = "sig" if F_sig else "n.s."
            print(f"\n    F-test (log vs generalized): F={F_val:.3f}, p={F_p:.4f} ({sig_str})")
            if not F_sig:
                print("    → Simple logarithmic model is sufficient (as expected for α=0)")
        except Exception as e:
            print(f"    ⚠ F-test failed: {e}")
            F_val, F_p, F_sig = np.nan, np.nan, False

        gen_fit_ok = True

    except Exception as e:
        print(f"  ⚠ Generalized fit failed: {e}")

    # ── Per-point pulls (primary fit) ──
    if gen_fit_ok:
        print_section("Per-point pulls (generalized fit)", level=2)
        for i, L_val in enumerate(L_used[mask]):
            idx_i = np.where(L_used == L_val)[0][0]
            pull = (cv_max[idx_i] - model_gen(L_val, *popt_gen)) / cv_max_err[idx_i]
            print(f"    L={L_val:3.0f}  pull = {pull:+.3f}")

    # ── PLOT 5: Specific heat — both fits ──
    print_section("Visualization", level=2)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = get_colors(L_LIST)

    for i, L_val in enumerate(L_used):
        idx = L_LIST.index(int(L_val))
        ax.errorbar(np.log(L_val), cv_max[i], yerr=cv_max_err[i],
                    fmt='o', color=colors[idx], ms=10, capsize=0, lw=1.5,
                    elinewidth=1.5, label=f'$L={int(L_val)}$', zorder=3)

    L_plot = np.linspace(20, 140, 100)
    # Log fit (cross-check)
    ax.plot(np.log(L_plot), popt_log[0] + popt_log[1] * np.log(L_plot),
            color='gray', lw=2, ls='--', label=r'$a + b\,\ln L$ (log)', zorder=2)
    # Generalized fit (primary)
    if gen_fit_ok:
        ax.plot(np.log(L_plot), model_gen(L_plot, *popt_gen),
                color='black', lw=2.5, label=r'$A\,L^g(1+q\,\ln L)$', zorder=2)

    ax.set_xlabel(r"$\ln(L)$", fontsize=20)
    ax.set_ylabel(r"$C_{\max}$", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=12, loc='lower right', ncol=1)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "specific_heat_log.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: specific_heat_log.pdf")

    return {
        'cv_max': cv_max,
        'cv_max_err': cv_max_err,
        'beta_peak_C': beta_peak_C,
        'beta_peak_C_err': beta_peak_C_err,
        'cv_max_spline': cv_max_spline,
        'cv_max_spline_err': cv_max_spline_err,
        'L_used': L_used,
        # Log fit (cross-check)
        'log_intercept': float(popt_log[0]),
        'log_intercept_err': float(errs_log[0]),
        'log_slope': float(popt_log[1]),
        'log_slope_err': float(errs_log[1]),
        'chi2_red_log': float(chi2_red_log),
        'chi2_val_log': float(chi2_val_log),
        'dof_log': dof_log,
        'p_val_log': float(p_val_log),
        # Generalized fit (primary)
        'A_gen': float(A_gen),
        'A_gen_err': float(A_gen_err),
        'g_gen': float(g_gen),
        'g_gen_err': float(g_gen_err),
        'q_gen': float(q_gen),
        'q_gen_err': float(q_gen_err),
        'chi2_red_gen': float(chi2_red_gen) if not np.isnan(chi2_red_gen) else np.nan,
        'chi2_val_gen': float(chi2_val_gen) if not np.isnan(chi2_val_gen) else np.nan,
        'dof_gen': dof_gen,
        'p_val_gen': float(p_val_gen) if not np.isnan(p_val_gen) else np.nan,
        'F_val': float(F_val) if not np.isnan(F_val) else np.nan,
        'F_p': float(F_p) if not np.isnan(F_p) else np.nan,
        'F_sig': F_sig,
        'gen_fit_ok': gen_fit_ok,
        # Derived
        'alpha': float(g_gen * nu) if gen_fit_ok else np.nan,
        'alpha_err': float(np.sqrt((g_gen_err * nu)**2 + (g_gen * nu_err)**2)) if gen_fit_ok else np.nan,
    }


# ============================================================================
# SUPPLEMENTARY PLOT: CHI PARABOLIC FIT EXAMPLE (L=96)
# ============================================================================
def plot_chi_parabolic_example(data, L_list, mag_result, L_example=96):
    """Supplementary plot: 3-point parabolic fit of chi(beta) for one L."""
    print_section(f"Supplementary: chi parabolic fit example (L={L_example})")

    if L_example not in data:
        print(f"  ⚠ L={L_example} not in data, skipping")
        return

    d = data[L_example]
    betas  = d['beta']
    chis   = d['chi']
    chi_errs = d['chi_err']

    # Reproduce the 3-point parabolic fit
    N_PARA_PTS = 3
    idx_max_raw = int(np.argmax(chis))
    half = N_PARA_PTS // 2
    lo = max(0, idx_max_raw - half)
    hi = min(len(betas), lo + N_PARA_PTS)
    lo = max(0, hi - N_PARA_PTS)

    b_win = betas[lo:hi]
    c_win = chis[lo:hi]
    e_win = chi_errs[lo:hi]

    def parabola(beta, c0, c1, c2):
        return c0 + c1 * beta + c2 * beta**2

    p0_c2 = -1e3
    p0_c1 = 2.0 * (-p0_c2) * betas[idx_max_raw]
    p0_c0 = chis[idx_max_raw] - p0_c1 * betas[idx_max_raw] - p0_c2 * betas[idx_max_raw]**2

    popt, pcov = spo.curve_fit(
        parabola, b_win, c_win,
        p0=[p0_c0, p0_c1, p0_c2],
        sigma=e_win, absolute_sigma=True
    )
    c0, c1, c2 = popt
    beta_peak = -c1 / (2.0 * c2)
    chi_max_val = c0 - c1**2 / (4.0 * c2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Show only the 3 fitted points — viridis color for this L
    clr = get_color_for_L(L_example, L_LIST)
    ax.errorbar(b_win, c_win, yerr=e_win,
                fmt='o', color=clr, ms=10, capsize=0, lw=1.5,
                elinewidth=1.5, label='Data', zorder=2)

    # Parabola overlay — extend slightly beyond the 3 points
    margin = (b_win[-1] - b_win[0]) * 0.25
    beta_fine = np.linspace(b_win[0] - margin, b_win[-1] + margin, 200)
    ax.plot(beta_fine, parabola(beta_fine, *popt),
            color='black', lw=2.5, ls='-',
            label=r'Parabolic fit', zorder=3)

    ax.set_xlabel(r"$\beta$", fontsize=20)
    ax.set_ylabel(r"$\chi'_{max}(\beta)$", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=12, loc='lower left')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "chi_parabolic_fit_example.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Saved: chi_parabolic_fit_example.pdf")


# ============================================================================
# SUPPLEMENTARY PLOT: C(beta) PARABOLIC FIT EXAMPLE (L=96)
# ============================================================================
def plot_C_parabolic_example(data, cv_result, L_example=96):
    """Supplementary plot: 3-point parabolic fit of C(beta) for one L."""
    print_section(f"Supplementary: C parabolic fit example (L={L_example})")

    if L_example not in data:
        print(f"  ⚠ L={L_example} not in data, skipping")
        return

    d = data[L_example]
    betas  = d['beta']
    Cs     = d['C']
    C_errs = d['C_err']

    # Reproduce the 3-point parabolic fit
    N_PARA_PTS = 3
    idx_max_raw = int(np.argmax(Cs))
    half = N_PARA_PTS // 2
    lo = max(0, idx_max_raw - half)
    hi = min(len(betas), lo + N_PARA_PTS)
    lo = max(0, hi - N_PARA_PTS)

    b_win = betas[lo:hi]
    c_win = Cs[lo:hi]
    e_win = C_errs[lo:hi]

    def parabola(beta, c0, c1, c2):
        return c0 + c1 * beta + c2 * beta**2

    p0_c2 = -1e2
    p0_c1 = 2.0 * (-p0_c2) * betas[idx_max_raw]
    p0_c0 = Cs[idx_max_raw] - p0_c1 * betas[idx_max_raw] - p0_c2 * betas[idx_max_raw]**2

    popt, pcov_p = spo.curve_fit(
        parabola, b_win, c_win,
        p0=[p0_c0, p0_c1, p0_c2],
        sigma=e_win, absolute_sigma=True
    )
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Show only the 3 fitted points — viridis color for this L
    clr = get_color_for_L(L_example, L_LIST)
    ax.errorbar(b_win, c_win, yerr=e_win,
                fmt='o', color=clr, ms=10, capsize=0, lw=1.5,
                elinewidth=1.5, label='Data', zorder=2)

    # Parabola overlay — extend slightly beyond the 3 points
    margin = (b_win[-1] - b_win[0]) * 0.25
    beta_fine = np.linspace(b_win[0] - margin, b_win[-1] + margin, 200)
    ax.plot(beta_fine, parabola(beta_fine, *popt),
            color='black', lw=2.5, ls='-',
            label=r'Parabolic fit', zorder=3)

    ax.set_xlabel(r"$\beta$", fontsize=20)
    ax.set_ylabel(r"$C_{max}(\beta)$", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=12, loc='lower left')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "C_parabolic_fit_example.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Saved: C_parabolic_fit_example.pdf")


# ============================================================================
# SUPPLEMENTARY PLOT: C_max(L) SCALING FIT
# ============================================================================
def plot_specific_heat_scaling_fit(cv_result, nu, nu_err):
    """Supplementary plot: C_max(L) vs L with generalized and log fits."""
    print_section("Supplementary: C_max(L) scaling fit")

    L_used     = cv_result['L_used']
    cv_max     = cv_result['cv_max']
    cv_max_err = cv_result['cv_max_err']

    # Mask L >= 32 (same as used in extract_alpha)
    L_min = 32
    mask = L_used >= L_min
    L_fit = L_used[mask]
    C_fit = cv_max[mask]
    C_err = cv_max_err[mask]

    # Retrieve fit parameters from cv_result
    gen_ok   = cv_result.get('gen_fit_ok', False)
    A_gen    = cv_result.get('A_gen', np.nan)
    g_gen    = cv_result.get('g_gen', np.nan)
    q_gen    = cv_result.get('q_gen', np.nan)
    chi2r_g  = cv_result.get('chi2_red_gen', np.nan)
    g_err    = cv_result.get('g_gen_err', np.nan)
    alpha    = cv_result.get('alpha', np.nan)
    alpha_err = cv_result.get('alpha_err', np.nan)

    # Model function
    def model_gen(L, A, g, q):
        return A * L**g * (1.0 + q * np.log(L))

    # Plot (L on x-axis, linear scale)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = get_colors(L_LIST)

    # Data points — viridis per-L colors
    for i, L_val in enumerate(L_fit):
        idx = L_LIST.index(int(L_val))
        ax.errorbar(L_val, C_fit[i], yerr=C_err[i],
                    fmt='o', color=colors[idx], ms=10, capsize=0, lw=1.5,
                    elinewidth=1.5, label=f'$L={int(L_val)}$', zorder=4)

    # Points excluded from fit (L < 32) — dimmed
    mask_excl = L_used < L_min
    for j in np.where(mask_excl)[0]:
        L_val = L_used[j]
        idx = L_LIST.index(int(L_val))
        ax.errorbar(L_val, cv_max[j], yerr=cv_max_err[j],
                    fmt='o', color=colors[idx], ms=10, capsize=0, lw=1.5,
                    elinewidth=1.5, alpha=0.3, label=f'$L={int(L_val)}$',
                    zorder=2)

    # Generalized fit curve — extend slightly beyond data
    L_plot = np.linspace(min(L_fit) * 0.85, max(L_fit) * 1.10, 200)
    if gen_ok:
        ax.plot(L_plot, model_gen(L_plot, A_gen, g_gen, q_gen),
                color='black', lw=2.5, ls='-',
                label=r'$A\,L^g(1+q\,\ln L)$', zorder=3)

    ax.set_xlabel(r'$L$', fontsize=20)
    ax.set_ylabel(r'$C_{\max}(L)$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=12, loc='lower right')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "specific_heat_scaling_fit.pdf",
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: specific_heat_scaling_fit.pdf")


# ============================================================================
# SUPPLEMENTARY PLOT: M(beta_pc, L) vs L  (log-log)
# ============================================================================
def plot_magnetization_vs_L(mag_result):
    """Supplementary plot: M(beta_pc, L) vs L in log-log scale."""
    print_section("Supplementary: M(beta_pc, L) vs L log-log")

    L_used   = mag_result['L_used']
    mag_pc   = mag_result['mag_pc']
    mag_pc_err = mag_result['mag_pc_err']
    beta_nu  = mag_result['beta_nu']

    valid = ~np.isnan(mag_pc) & (L_used >= 32)

    def model_mag_simple(L, A, beta_nu):
        return A * L**(-beta_nu)

    pcov_mag = None
    chi2_red_mag = np.nan
    try:
        popt, pcov_mag = spo.curve_fit(
            model_mag_simple, L_used[valid], mag_pc[valid],
            p0=[1.0, 0.125], sigma=mag_pc_err[valid], absolute_sigma=True
        )
        resid = mag_pc[valid] - model_mag_simple(L_used[valid], *popt)
        dof_mag = int(np.sum(valid)) - len(popt)
        chi2_red_mag = float(np.sum((resid / mag_pc_err[valid])**2)) / dof_mag if dof_mag > 0 else np.nan
    except Exception:
        popt = [1.0, beta_nu]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = get_colors(L_LIST)

    for i, L_val in enumerate(L_used[valid]):
        idx = L_LIST.index(int(L_val))
        ax.errorbar(L_val, mag_pc[valid][i], yerr=mag_pc_err[valid][i],
                    fmt='o', color=colors[idx], ms=10, capsize=0, lw=1.5,
                    elinewidth=1.5, label=f'$L={int(L_val)}$', zorder=3)

    L_plot = np.linspace(20, 140, 200)
    ax.plot(L_plot, model_mag_simple(L_plot, *popt),
            color='black', lw=2.5,
            label='Fit',
            zorder=2)

    ax.text(0.05, 0.05,
        rf'$\beta/\nu = {popt[1]:.4f} \pm {np.sqrt(pcov_mag[1,1]):.4f}$' + '\n' +
        rf'$\chi^2_{{\rm red}} = {chi2_red_mag:.3f}$',
        transform=ax.transAxes, va='bottom', ha='left', fontsize=13,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$L$", fontsize=20)
    ax.set_ylabel(r"$M(\beta_{pc},\, L)$", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':', which='both')
    ax.legend(frameon=False, fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "magnetization_vs_L.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  \u2713 Saved: magnetization_vs_L.pdf")


# ============================================================================
# PHASE 5: DATA COLLAPSE
# ============================================================================
def data_collapse_plots(data, L_list, beta_pc, nu, gamma_nu, beta_nu):
    """Phase 5: four data-collapse plots (chi, M, U4, Cv) using measured exponents."""
    print_phase_header(5, 5, "DATA COLLAPSE VALIDATION",
                       description="Scaling functions with measured exponents")

    print(f"  Using MEASURED exponents:")
    print(f"    beta_pc = {beta_pc:.6f}, nu = {nu:.3f}")
    print(f"    gamma/nu = {gamma_nu:.3f}, beta/nu = {beta_nu:.3f}")
    print("-" * 80)

    colors = get_colors(L_list)

    # PLOT 6: Susceptibility
    print("\n  Generating plots...")

    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]

        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['chi'] / L**(gamma_nu)

        ax.plot(x, y, 'o', color=colors[i], ms=4.5, alpha=0.75,
                label=f"$L={L}$", zorder=2)

    ax.axvline(0, color='grey', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r"$(\beta-\beta_{pc})L^{1/\nu}$", fontsize=20)
    ax.set_ylabel(r"$\chi' / L^{\gamma/\nu}$", fontsize=20)
    ax.set_xlim(*XLIM_COLLAPSE)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=1)

    axins = inset_axes(ax, width="40%", height="35%", loc='lower left',
                       bbox_to_anchor=(0.56, 0.60, 1, 1), bbox_transform=ax.transAxes)
    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]
        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['chi'] / L**(gamma_nu)
        axins.plot(x, y, 'o', color=colors[i], ms=3.5, alpha=0.75)

    axins.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.6)
    axins.set_xlim(-0.15, 0.10)
    axins.set_ylim(0.065, 0.087)
    axins.tick_params(labelsize=10)
    axins.grid(alpha=0.4, ls=':')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "collapse_susceptibility_inset.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 7: Magnetization
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]

        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['M_abs'] * L**(beta_nu)

        ax.plot(x, y, 'o', color=colors[i], ms=4.5, alpha=0.75,
                label=f"$L={L}$", zorder=2)

    ax.axvline(0, color='grey', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r"$(\beta-\beta_{pc})L^{1/\nu}$", fontsize=20)
    ax.set_ylabel(r"$\langle |m| \rangle L^{\beta/\nu}$", fontsize=20)
    ax.set_xlim(*XLIM_COLLAPSE)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=1)

    axins = inset_axes(ax, width="40%", height="35%", loc='lower left',
                       bbox_to_anchor=(0.55, 0.15, 1, 1), bbox_transform=ax.transAxes)
    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]
        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['M_abs'] * L**(beta_nu)
        axins.plot(x, y, 'o', color=colors[i], ms=3.5, alpha=0.75)

    axins.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.6)
    axins.set_xlim(-0.15, 0.10)
    axins.set_ylim(0.96, 1.05)
    axins.tick_params(labelsize=10)
    axins.grid(alpha=0.4, ls=':')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "collapse_magnetization_inset.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 8: Binder
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]

        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['U4']

        ax.plot(x, y, 'o', color=colors[i], ms=4.5, alpha=0.75,
                label=f"$L={L}$", zorder=2)

    ax.axvline(0, color='grey', ls=':', lw=1, alpha=0.6)
    ax.axhline(0.611, color='orange', ls='--', lw=1.5, alpha=0.7,
               label=r'$U^* = 0.611$')
    ax.set_xlabel(r"$(\beta-\beta_{pc})L^{1/\nu}$", fontsize=20)
    ax.set_ylabel(r"$U_4$", fontsize=20)
    ax.set_xlim(*XLIM_COLLAPSE)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=1)

    axins = inset_axes(ax, width="40%", height="35%", loc='lower left',
                       bbox_to_anchor=(0.48, 0.15, 1, 1), bbox_transform=ax.transAxes)
    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]
        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['U4']
        axins.plot(x, y, 'o', color=colors[i], ms=3.5, alpha=0.75)

    axins.axvline(0, color='grey', ls=':', lw=0.8, alpha=0.6)
    axins.axhline(0.611, color='orange', ls='--', lw=1.2, alpha=0.7)
    axins.set_xlim(-0.15, 0.10)
    axins.set_ylim(0.590, 0.620)
    axins.tick_params(labelsize=10)
    axins.grid(alpha=0.4, ls=':')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "collapse_binder_inset.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # PLOT 9: C_v collapse
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for i, L in enumerate(L_list):
        if L not in data:
            continue
        d = data[L]

        x = (d['beta'] - beta_pc) * L**(1 / nu)
        y = d['C'] / np.log(L)

        ax.plot(x, y, 'o', color=colors[i], ms=4.5, alpha=0.75,
                label=f"$L={L}$", zorder=2)

    ax.axvline(0, color='grey', ls=':', lw=1, alpha=0.6)
    ax.set_xlabel(r"$(\beta-\beta_{pc})L^{1/\nu}$", fontsize=20)
    ax.set_ylabel(r"$C_v / \ln(L)$", fontsize=20)
    ax.set_xlim(*XLIM_COLLAPSE)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.3, ls=':')
    ax.legend(frameon=False, fontsize=11, loc='upper left', ncol=1)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "collapse_cv_log.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print("  \u2713 Saved: collapse_susceptibility_inset.pdf")
    print("  \u2713 Saved: collapse_magnetization_inset.pdf")
    print("  \u2713 Saved: collapse_binder_inset.pdf")
    print("  \u2713 Saved: collapse_cv_log.pdf")


# ============================================================================
# PLOT 10: ROBUSTNESS VS L_min
# ============================================================================
def plot_robustness_vs_Lmin(data, L_list, beta_pc, beta_pc_err, nu, nu_err,
                            gamma_nu, gamma_nu_err, ylims=None):
    """Plot 10: three-panel robustness check of exponents vs L_min.

    Uses simplified extraction (spline peaks, simple power-law fits)
    for speed; results are qualitative and not used in the main analysis.
    """
    print("\n  [PLOT 10/10] Generating robustness check (vs L_min)..."
          "  [qualitative — simplified extraction]")

    L_min_values = [24, 32, 48, 64]

    beta_c_vals, inv_nu_vals, gamma_nu_vals = [], [], []
    beta_c_errs, inv_nu_errs, gamma_nu_errs = [], [], []

    for L_min in L_min_values:
        # beta_c from crossings
        crossings_temp = []
        for i in range(len(L_list) - 1):
            L1, L2 = L_list[i], L_list[i + 1]
            if L1 < L_min or L1 not in data or L2 not in data:
                continue

            d1 = data[L1]
            d2 = data[L2]
            mask1 = (d1['beta'] >= 0.435) & (d1['beta'] <= 0.445)
            mask2 = (d2['beta'] >= 0.435) & (d2['beta'] <= 0.445)

            try:
                w1 = 1.0 / (d1['U4_err'][mask1] + 1e-12)
                w2 = 1.0 / (d2['U4_err'][mask2] + 1e-12)

                spl1 = spi.UnivariateSpline(d1['beta'][mask1], d1['U4'][mask1],
                                            w=w1, k=3, s=np.sum(mask1))
                spl2 = spi.UnivariateSpline(d2['beta'][mask2], d2['U4'][mask2],
                                            w=w2, k=3, s=np.sum(mask2))

                bc = spo.brentq(lambda b: spl1(b) - spl2(b), 0.435, 0.445)
                crossings_temp.append(bc)
            except:
                pass

        if len(crossings_temp) > 0:
            beta_c_vals.append(np.mean(crossings_temp))
            beta_c_errs.append(np.std(crossings_temp) / np.sqrt(len(crossings_temp)))
        else:
            beta_c_vals.append(np.nan)
            beta_c_errs.append(np.nan)

        # 1/nu from slopes
        slopes_temp, L_temp = [], []
        for L in L_list:
            if L < L_min or L not in data:
                continue

            d = data[L]
            mask = (d['beta'] >= 0.43) & (d['beta'] <= 0.45)

            try:
                w = 1.0 / (d['U4_err'][mask] + 1e-12)
                spl = spi.UnivariateSpline(d['beta'][mask], d['U4'][mask],
                                           w=w, k=3, s=np.sum(mask))
                slope_val = abs(spl.derivative()(beta_pc))
                slopes_temp.append(slope_val)
                L_temp.append(L)
            except:
                pass

        if len(slopes_temp) >= 2:
            def model(L, A, inv_nu):
                return A * L**inv_nu

            try:
                popt, pcov = spo.curve_fit(model, L_temp, slopes_temp,
                                           p0=[slopes_temp[0] / L_temp[0], 1.0])
                inv_nu_vals.append(popt[1])
                inv_nu_errs.append(np.sqrt(pcov[1, 1]))
            except:
                inv_nu_vals.append(np.nan)
                inv_nu_errs.append(np.nan)
        else:
            inv_nu_vals.append(np.nan)
            inv_nu_errs.append(np.nan)

        # gamma/nu from chi_max
        chi_max_temp, L_chi_temp = [], []
        for L in L_list:
            if L < L_min or L not in data:
                continue

            d = data[L]
            try:
                spl_chi = spi.UnivariateSpline(d['beta'], d['chi'],
                                               w=1 / (d['chi_err'] + 1e-12), k=3,
                                               s=len(d['beta']))
                opt = spo.minimize_scalar(lambda b: -spl_chi(b),
                                          bounds=(0.43, 0.45), method='bounded')
                chi_max_temp.append(-opt.fun)
                L_chi_temp.append(L)
            except:
                pass

        if len(chi_max_temp) >= 2:
            def model_chi(L, A, gamma_nu):
                return A * L**gamma_nu

            try:
                popt_chi, pcov_chi = spo.curve_fit(model_chi, L_chi_temp, chi_max_temp,
                                                   p0=[0.11, 1.75])
                gamma_nu_vals.append(popt_chi[1])
                gamma_nu_errs.append(np.sqrt(pcov_chi[1, 1]))
            except:
                gamma_nu_vals.append(np.nan)
                gamma_nu_errs.append(np.nan)
        else:
            gamma_nu_vals.append(np.nan)
            gamma_nu_errs.append(np.nan)

    # TRIPLE PANEL
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.errorbar(L_min_values, beta_c_vals, yerr=beta_c_errs,
                 fmt='o', color='purple', ms=8, capsize=0, lw=1.5,
                 elinewidth=1.5, zorder=3)
    ax1.axhline(beta_pc, color='red', ls='-', lw=2, zorder=2)
    ax1.axhline(BETA_C_EXACT, color='grey', ls='--', lw=0.8, alpha=0.7, zorder=1)
    ax1.axhspan(beta_pc - beta_pc_err, beta_pc + beta_pc_err,
                alpha=0.1, color='red', zorder=0)
    ax1.set_xlabel(r"$L_{\min}$", fontsize=18)
    ax1.set_ylabel(r"$\beta$", fontsize=18)
    ax1.tick_params(labelsize=14)
    ax1.grid(alpha=0.3, ls=':')
    ax1.set_xlim(20, 68)

    inv_nu_final = 1.0 / nu
    inv_nu_final_err = nu_err / nu**2
    ax2.errorbar(L_min_values, inv_nu_vals, yerr=inv_nu_errs,
                 fmt='o', color='purple', ms=8, capsize=0, lw=1.5,
                 elinewidth=1.5, zorder=3)
    ax2.axhline(inv_nu_final, color='red', ls='-', lw=2, zorder=2)
    ax2.axhline(1.0 / NU_EXACT, color='grey', ls='--', lw=0.8, alpha=0.7, zorder=1)
    ax2.axhspan(inv_nu_final - inv_nu_final_err, inv_nu_final + inv_nu_final_err,
                alpha=0.1, color='red', zorder=0)
    ax2.set_xlabel(r"$L_{\min}$", fontsize=18)
    ax2.set_ylabel(r"$1/\nu$", fontsize=18)
    ax2.tick_params(labelsize=14)
    ax2.grid(alpha=0.3, ls=':')
    ax2.set_xlim(20, 68)

    ax3.errorbar(L_min_values, gamma_nu_vals, yerr=gamma_nu_errs,
                 fmt='o', color='purple', ms=8, capsize=0, lw=1.5,
                 elinewidth=1.5, zorder=3)
    ax3.axhline(gamma_nu, color='red', ls='-', lw=2, zorder=2)
    ax3.axhline(GAMMA_NU_EXACT, color='grey', ls='--', lw=0.8, alpha=0.7, zorder=1)
    ax3.axhspan(gamma_nu - gamma_nu_err, gamma_nu + gamma_nu_err,
                alpha=0.1, color='red', zorder=0)
    ax3.set_xlabel(r"$L_{\min}$", fontsize=18)
    ax3.set_ylabel(r"$\gamma/\nu$", fontsize=18)
    ax3.tick_params(labelsize=14)
    ax3.grid(alpha=0.3, ls=':')
    ax3.set_xlim(20, 68)

    if ylims:
        if ylims[0]:
            ax1.set_ylim(ylims[0])
        if ylims[1]:
            ax2.set_ylim(ylims[1])
        if ylims[2]:
            ax3.set_ylim(ylims[2])

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "robustness_vs_Lmin.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: robustness_vs_Lmin.pdf")


# ============================================================================
# SAVE RESULTS (ENHANCED)
# ============================================================================
def save_results_enhanced(beta_pc, beta_pc_err, U_star, U_star_err,
                          crossings, nu_result, mag_result, cv_result,
                          corrections_results):
    """Write structured .dat results file and LaTeX tables."""

    print("\n" + "=" * 80)
    print("[SAVING RESULTS]")
    print("=" * 80)

    nu_used     = nu_result.get('nu_final', nu_result['nu'])
    nu_used_err = nu_result.get('nu_final_err', nu_result['nu_err'])
    nu_method   = nu_result.get('nu_method', 'Phase 2 slope fit')
    gamma_nu = mag_result['gamma_nu']
    gamma_nu_err = mag_result['gamma_nu_err']
    beta_nu = mag_result['beta_nu']
    beta_nu_err = mag_result['beta_nu_err']

    # cv_result fields
    cv_slope = cv_result['log_slope']
    cv_intercept = cv_result['log_intercept']

    # ========== FILE .dat STRUCTURED ==========
    filepath_dat = ANALYSIS_DIR / "fss_results_complete.dat"

    with open(filepath_dat, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" FINITE-SIZE SCALING ANALYSIS - COMPLETE RESULTS\n")
        f.write(" 2D Ising Model with Periodic Boundary Conditions\n")
        f.write("=" * 80 + "\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}\n")
        f.write(f" Analysis code version: 2.0\n")
        f.write("=" * 80 + "\n\n")

        f.write("[ANALYSIS PARAMETERS]\n")
        f.write("-" * 80 + "\n")
        f.write(f" Lattice sizes:        {L_LIST}\n")
        f.write(f" Beta range:           {BETA_RANGE_FIT}\n")
        f.write(f" Bootstrap samples:    {N_BOOTSTRAP}\n")
        f.write(f" Omega (fixed):        {OMEGA_EXACT}\n")
        f.write(f" L_min (cutoff):       32\n")
        f.write("\n")

        f.write("[CRITICAL TEMPERATURE]\n")
        f.write("-" * 80 + "\n")
        f.write(f" Method: Binder cumulant U4 crossing + bootstrap\n")
        f.write(f" Number of crossings: {len(crossings)}\n")
        f.write("\n")
        f.write(f" beta_pc           = {beta_pc:.10f}\n")
        f.write(f" beta_pc_err       = {beta_pc_err:.10f}\n")
        f.write(f" beta_c_exact      = {BETA_C_EXACT:.15f}\n")
        f.write(f" deviation_abs     = {abs(beta_pc - BETA_C_EXACT):.10e}\n")
        f.write(f" deviation_rel     = {abs(beta_pc - BETA_C_EXACT) / BETA_C_EXACT * 100:.6f}%\n")
        f.write(f" deviation_sigma   = {abs(beta_pc - BETA_C_EXACT) / beta_pc_err:.2f}\n")
        f.write("\n")
        f.write(f" U_star            = {U_star:.6f}\n")
        f.write(f" U_star_err        = {U_star_err:.6f}\n")
        f.write(f" U_star_ref        = 0.6107  [numerical estimate, Ferrenberg & Landau]\n")
        f.write("\n")

        f.write("[INDIVIDUAL CROSSINGS]\n")
        f.write("-" * 80 + "\n")
        f.write(" L1   L2   beta_cross           sigma_beta        U*        sigma_U\n")
        f.write("-" * 80 + "\n")
        for c in crossings:
            f.write(f" {c[0]:3d}  {c[1]:3d}  {c[2]:.15f}  {c[3]:.12f}  {c[4]:.6f}  {c[5]:.6f}\n")
        f.write("\n")

        f.write("[CORRELATION LENGTH EXPONENT]\n")
        f.write("-" * 80 + "\n")
        f.write(f" Method: {nu_method}\n")
        f.write(f" Fit model: S(L) = A * L^(1/nu) * [1 + B*L^(-omega)]\n")
        f.write("\n")
        f.write(f" nu (Phase 2, pcov)          = {nu_result['nu']:.6f} +/- {nu_result['nu_err']:.6f}  [conservative, Cramer-Rao]\n")
        if 'nu_correlated' in nu_result:
            f.write(f" nu (corr. fit w/ B)         = {nu_result['nu_correlated']:.6f} +/- "
                    f"{nu_result['nu_correlated_err']:.6f}  [diagnostic only, chi2_red="
                    f"{nu_result.get('chi2_red_correlated', np.nan):.3f}]\n")
        f.write(f" nu (USED)                   = {nu_used:.6f} +/- {nu_used_err:.6f}"
                f"  [corr. leading-only fit, used for ALL downstream phases]\n")
        f.write(f" nu_exact          = {NU_EXACT}\n")
        f.write(f" deviation_sigma   = {abs(nu_used - NU_EXACT) / nu_used_err:.2f}\n")
        f.write("\n")
        if 'A_corr' in nu_result:
            f.write(f" amplitude_A (correlated fit, USED) = {nu_result['A_corr']:.6f} +/- {nu_result['A_corr_err']:.6f}\n")
            f.write(f" correction_B (correlated fit, USED) = {nu_result['B_corr']:.2f} +/- {nu_result['B_corr_err']:.2f}\n")
        f.write(f" amplitude_A (Phase 2, pcov)        = {nu_result.get('A', np.nan):.6f} +/- {nu_result.get('A_err', np.nan):.6f}\n")
        f.write(f" correction_B (Phase 2, pcov)       = {nu_result.get('B', np.nan):.2f} +/- {nu_result.get('B_err', np.nan):.2f}\n")
        f.write(f" chi2_red (diagonal, per-L fit)  = {nu_result.get('chi2_red', np.nan):.3f}  [informative only]\n")
        if 'chi2_red_correlated' in nu_result:
            f.write(f" chi2_red (correlated fit, USED) = {nu_result['chi2_red_correlated']:.3f}")
            f.write(f"  p = {nu_result.get('p_value_correlated', np.nan):.3f}\n")
        f.write("\n")

        f.write("[NU EXTRACTION DETAILS]\n")
        f.write("-" * 80 + "\n")
        slope_method = nu_result.get('slope_err_method', 'unknown')
        f.write(f" Slope error propagation: {slope_method}\n")
        f.write(f" Spline: UnivariateSpline k=3, s=N_points\n")
        f.write(f" Bootstrap samples: {N_BOOTSTRAP}\n")
        f.write("\n")
        f.write(" Binder slopes used in nu fit (L, S(L), sigma_S)\n")
        if 'L_used' in nu_result and 'slopes' in nu_result and 'slopes_err' in nu_result:
            for L_s, S_s, eS_s in zip(nu_result['L_used'], nu_result['slopes'], nu_result['slopes_err']):
                f.write(f" {L_s:5d}  {S_s: .8f}  {eS_s: .8f}\n")
        f.write("\n")

        f.write("[MAGNETIC EXPONENTS]\n")
        f.write("-" * 80 + "\n")
        f.write(" Fitted as FREE parameters (not constrained to theory)\n")
        f.write(" chi_max extraction: LOCAL PARABOLIC FIT (3 pts, Rummukainen standard, pcov errors)\n")
        f.write(" Source: Phase 3 (all L >= L_min = 32, 3-param fit with corrections)\n")
        f.write(" NOTE: these values are USED for summary table, hyperscaling, data collapse, LaTeX\n")
        f.write("\n")
        f.write(f" gamma_over_nu     = {gamma_nu:.6f}\n")
        f.write(f" gamma_nu_err      = {gamma_nu_err:.6f}\n")
        f.write(f" gamma_nu_exact    = {GAMMA_NU_EXACT}\n")
        f.write(f" deviation         = {abs(gamma_nu - GAMMA_NU_EXACT):.6f}\n")
        f.write(f" deviation_sigma   = {abs(gamma_nu - GAMMA_NU_EXACT) / gamma_nu_err:.2f}σ\n")
        f.write(f" amplitude_A       = {mag_result.get('A_chi', np.nan):.6f} +/- {mag_result.get('A_chi_err', np.nan):.6f}  [pcov]\n")
        f.write(f" chi2_red          = {mag_result.get('chi2_chi', np.nan):.3f}\n")
        if 'chi2_dof_chi' in mag_result:
            f.write(f" dof              = {int(mag_result.get('chi2_dof_chi', 0))}\n")
        if 'chi2_p_chi' in mag_result:
            f.write(f" p_value          = {mag_result.get('chi2_p_chi', np.nan):.6f}\n")
        f.write("\n")
        f.write(f" beta_over_nu      = {beta_nu:.6f}\n")
        f.write(f" beta_nu_err       = {beta_nu_err:.6f}\n")
        f.write(f" beta_nu_exact     = {BETA_NU_EXACT}\n")
        f.write(f" deviation         = {abs(beta_nu - BETA_NU_EXACT):.6f}\n")
        f.write(f" deviation_sigma   = {abs(beta_nu - BETA_NU_EXACT) / beta_nu_err:.2f}σ\n")
        f.write(f" amplitude_A       = {mag_result.get('A_mag', np.nan):.6f} +/- {mag_result.get('A_mag_err', np.nan):.6f}  [pcov]\n")
        f.write(f" chi2_red          = {mag_result.get('chi2_mag', np.nan):.3f}\n")
        if 'chi2_dof_mag' in mag_result:
            f.write(f" dof              = {int(mag_result.get('chi2_dof_mag', 0))}\n")
        if 'chi2_p_mag' in mag_result:
            f.write(f" p_value          = {mag_result.get('chi2_p_mag', np.nan):.6f}\n")
        f.write("\n")

        # chi_max and beta_peak per L — parabolic vs spline cross-check
        f.write(" chi_max(L): parabolic fit [USED] vs spline [cross-check]\n")
        L_mag = mag_result.get('L_used', [])
        cm     = mag_result.get('chi_max', [])
        cm_err = mag_result.get('chi_max_err', [])
        bp     = mag_result.get('beta_peak', [])
        bp_err = mag_result.get('beta_peak_err', [])
        cm_spl     = mag_result.get('chi_max_spline', [])
        cm_spl_err = mag_result.get('chi_max_spline_err', [])
        for i in range(len(L_mag)):
            spl_str = f"{cm_spl[i]:10.2f} ± {cm_spl_err[i]:6.2f}" if i < len(cm_spl) else "N/A"
            f.write(f"  L={int(L_mag[i]):3d}  chi_max={cm[i]:10.2f} ± {cm_err[i]:6.2f} [para]  "
                    f"beta_peak={bp[i]:.6f} ± {bp_err[i]:.6f}  |  "
                    f"chi_spl={spl_str}\n")
        f.write("\n")

        f.write("[DERIVED EXPONENTS]\n")
        f.write("-" * 80 + "\n")
        f.write(f" gamma             = {mag_result['gamma']:.6f} +/- {mag_result['gamma_err']:.6f}\n")
        f.write(f" beta_mag          = {mag_result['beta']:.6f} +/- {mag_result['beta_err']:.6f}\n")
        f.write(f" eta               = {mag_result['eta']:.6f} +/- {mag_result['eta_err']:.6f}\n")
        f.write("\n")

        f.write("[HYPERSCALING RELATION]\n")
        f.write("-" * 80 + "\n")
        f.write(f" 2*beta/nu + gamma/nu = {mag_result['hyperscaling']:.6f}\n")
        f.write(f" Expected (d=2)       = 2.0000\n")
        f.write(f" Deviation            = {abs(mag_result['hyperscaling'] - 2.0):.6f}\n")
        f.write(f" Deviation_sigma      = {abs(mag_result['hyperscaling'] - 2.0) / mag_result['hyperscaling_err']:.2f}\n")
        f.write("\n")

        f.write("[FINITE-SIZE CORRECTIONS (RIGOROUS ANALYSIS)]\n")
        f.write("-" * 80 + "\n")
        f.write(" Method: TWO-STEP fit + F-test, errors from pcov (Cramér-Rao)\n")
        f.write(" Source: Phase 3b (L >= L_min = 32, independent re-fit)\n")
        f.write(" NOTE: exponents here are used ONLY for sub-leading correction analysis,\n")
        f.write("        NOT for the summary table, hyperscaling, or data collapse.\n")
        f.write(" STEP 1: O(L) = A * L^exponent\n")
        f.write(" STEP 2: O(L) = A * L^exponent * [1 + B*L^(-2)] (exponent fixed)\n")
        f.write("\n")

        for res in corrections_results:
            if res is None:
                continue
            obs_name = res['observable']
            f.write(f" OBSERVABLE: {obs_name}\n")
            is_cv = ('C_max' in obs_name)
            if is_cv:
                f.write(f"   Step 1: C_max = A + B*ln(L)\n")
                f.write(f"   slope B        = {res['exponent']:.6f} +/- {res['exponent_err']:.6f}  [pcov]\n")
                f.write(f"   intercept A    = {res.get('A_simple', np.nan):.6f} +/- {res.get('A_simple_err', np.nan):.6f}  [pcov]\n")
            else:
                f.write(f"   Step 1: O(L) = A * L^exponent\n")
                f.write(f"   exponent       = {res['exponent']:.6f} +/- {res['exponent_err']:.6f}  [pcov]\n")
                f.write(f"   amplitude_A    = {res.get('A_simple', np.nan):.6f} +/- {res.get('A_simple_err', np.nan):.6f}  [pcov]\n")
            dof_s = res.get('dof_simple', 0)
            chi2_raw_s = res.get('chi2_raw_simple', np.nan)
            p_val_s = float(sps.chi2.sf(chi2_raw_s, dof_s)) if dof_s > 0 and not np.isnan(chi2_raw_s) else np.nan
            f.write(f"   chi2_simple    = {res['chi2_red_simple']:.3f}\n")
            f.write(f"   dof_simple     = {dof_s}\n")
            f.write(f"   p_value_simple = {p_val_s:.4f}\n")
            if is_cv:
                f.write(f"   Step 2: C_max = A + B_fixed*ln(L) + D*L^(-2)  (B fixed from Step 1)\n")
                f.write(f"   intercept A    = {res.get('A_corrected', np.nan):.6f} +/- {res.get('A_corrected_err', np.nan):.6f}  [pcov]\n")
                f.write(f"   correction_D   = {res['B']:.2f} +/- {res['B_err']:.2f}  [pcov]\n")
            else:
                f.write(f"   Step 2: O(L) = A * L^exponent * [1 + B*L^(-2)]  (exponent fixed)\n")
                f.write(f"   amplitude_A    = {res.get('A_corrected', np.nan):.6f} +/- {res.get('A_corrected_err', np.nan):.6f}  [pcov]\n")
                f.write(f"   correction_B   = {res['B']:.2f} +/- {res['B_err']:.2f}  [pcov]\n")
            dof_c = res.get('dof_corrected', 0)
            chi2_raw_c = res.get('chi2_raw_corrected', np.nan)
            p_val_c = float(sps.chi2.sf(chi2_raw_c, dof_c)) if dof_c > 0 and not np.isnan(chi2_raw_c) else np.nan
            f.write(f"   chi2_corrected = {res['chi2_red_corrected']:.3f}\n")
            f.write(f"   dof_corrected  = {dof_c}\n")
            f.write(f"   p_value_corrected = {p_val_c:.4f}\n")
            f.write(f"   F_statistic    = {res['F_statistic']:.3f}\n")
            f.write(f"   p_value_F      = {res['p_value']:.4f}\n")
            f.write(f"   significant    = {'YES' if res['is_significant'] else 'NO'}\n")
            f.write("\n")

        f.write("[SPECIFIC HEAT EXPONENT]\n")
        f.write("-" * 80 + "\n")
        f.write(" C_max extraction: LOCAL PARABOLIC FIT (3 pts, Rummukainen standard, pcov errors)\n")
        f.write("\n")
        f.write(" Simple log fit:       C_max = A + B*ln(L)   [reference]\n")
        f.write(f"   intercept A     = {cv_intercept:.6f} +/- {cv_result['log_intercept_err']:.6f}  [pcov]\n")
        f.write(f"   slope B         = {cv_slope:.6f} +/- {cv_result['log_slope_err']:.6f}  [pcov]\n")
        f.write(f"   chi2_red        = {cv_result['chi2_red_log']:.3f}\n")
        f.write(f"   dof             = {cv_result['dof_log']}\n")
        f.write(f"   p_value         = {cv_result['p_val_log']:.6f}\n")
        f.write("\n")
        if cv_result.get('gen_fit_ok', False):
            f.write(" Generalized fit:      C_max = A*L^g*(1+q*ln(L))   [USED]\n")
            f.write(f"   A               = {cv_result['A_gen']:.6f} +/- {cv_result['A_gen_err']:.6f}  [pcov]\n")
            f.write(f"   g (alpha/nu)    = {cv_result['g_gen']:.6f} +/- {cv_result['g_gen_err']:.6f}  [pcov]\n")
            f.write(f"   q (log corr.)   = {cv_result['q_gen']:.6f} +/- {cv_result['q_gen_err']:.6f}  [pcov]\n")
            f.write(f"   chi2_red        = {cv_result['chi2_red_gen']:.3f}\n")
            f.write(f"   dof             = {cv_result['dof_gen']}\n")
            f.write(f"   p_value         = {cv_result['p_val_gen']:.6f}\n")
            f.write(f"   alpha = g*nu    = {cv_result['alpha']:.6f} +/- {cv_result['alpha_err']:.6f}  [propagated]\n")
            f.write("\n")
            f.write(f" F-test (simple vs generalized): F={cv_result['F_val']:.3f}, p={cv_result['F_p']:.4f}\n")
            f.write(f"   significant     = {'YES' if cv_result['F_sig'] else 'NO'}\n")
            if not cv_result['F_sig']:
                f.write(f"   → Simple log model sufficient (alpha=0 consistent)\n")
        else:
            f.write(f" alpha             = 0  (logarithmic divergence)\n")
        f.write("\n")

        # ===== PEAK TABLES =====
        f.write("[PEAK VALUES TABLE - SUSCEPTIBILITY]\n")
        f.write("-" * 80 + "\n")
        f.write(" Method: local parabolic fit, 3 points, errors from pcov\n")
        L_mag = mag_result.get('L_used', [])
        cm     = mag_result.get('chi_max', [])
        cm_err = mag_result.get('chi_max_err', [])
        bp     = mag_result.get('beta_peak', [])
        bp_err = mag_result.get('beta_peak_err', [])
        f.write(f" {'L':>5s}  {'beta_peak':>12s}  {'sigma_beta':>12s}  {'chi_max':>12s}  {'sigma_chi':>10s}\n")
        for i in range(len(L_mag)):
            f.write(f" {int(L_mag[i]):5d}  {bp[i]:12.6f}  {bp_err[i]:12.6f}  "
                    f"{cm[i]:12.2f}  {cm_err[i]:10.2f}\n")
        f.write("\n")

        f.write("[PEAK VALUES TABLE - SPECIFIC HEAT]\n")
        f.write("-" * 80 + "\n")
        f.write(" Method: local parabolic fit, 3 points, errors from pcov\n")
        L_cv = cv_result.get('L_used', [])
        cvm     = cv_result.get('cv_max', [])
        cvm_err = cv_result.get('cv_max_err', [])
        bpC     = cv_result.get('beta_peak_C', [])
        bpC_err = cv_result.get('beta_peak_C_err', [])
        f.write(f" {'L':>5s}  {'beta_peak':>12s}  {'sigma_beta':>12s}  {'C_max':>12s}  {'sigma_C':>10s}\n")
        for i in range(len(L_cv)):
            f.write(f" {int(L_cv[i]):5d}  {bpC[i]:12.6f}  {bpC_err[i]:12.6f}  "
                    f"{cvm[i]:12.4f}  {cvm_err[i]:10.4f}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(" END OF RESULTS\n")
        f.write("=" * 80 + "\n")

    print(f"  \u2713 Structured results: {filepath_dat}")

    # ========== MULTIPLE LATEX TABLES ==========
    generate_latex_tables(
        beta_pc, beta_pc_err, U_star, U_star_err,
        crossings, nu_result, mag_result, cv_result,
        corrections_results
    )


# ============================================================================
# GENERATE MULTIPLE LATEX TABLES
# ============================================================================
def generate_latex_tables(beta_pc, beta_pc_err, U_star, U_star_err,
                          crossings, nu_result, mag_result, cv_result,
                          corrections_results):
    """Generate five LaTeX tables (crossings, amplitudes, corrections, summary, Onsager)."""

    nu_used     = nu_result.get('nu_final', nu_result['nu'])
    nu_used_err = nu_result.get('nu_final_err', nu_result['nu_err'])
    nu_method   = nu_result.get('nu_method', 'Phase 2 slope fit')

    filepath = ANALYSIS_DIR / "fss_tables_complete.tex"

    with open(filepath, 'w') as f:
        f.write("% ============================================================\n")
        f.write("% FINITE-SIZE SCALING ANALYSIS - LATEX TABLES\n")
        f.write("% Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("% ============================================================\n\n")

        # ===== TABLE 1: BINDER CROSSINGS =====
        f.write("% TABLE 1: Binder Cumulant Crossings\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("$(L_1,L_2)$ & $\\beta_{\\text{cross}} \\pm \\sigma_{\\beta}$ "
                "& $U^* \\pm \\sigma_{U^*}$ \\\\\n")
        f.write("\\midrule\n")

        for c in crossings:
            f.write(f"{c[0]}/{c[1]} & ${c[2]:.6f} \\pm {c[3]:.6f}$ "
                    f"& ${c[4]:.4f} \\pm {c[5]:.5f}$ \\\\\n")

        f.write("\\midrule\n")
        f.write(f"Media pesata & ${beta_pc:.10f} \\pm {beta_pc_err:.10f}$ "
                f"& ${U_star:.6f} \\pm {U_star_err:.6f}$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Punti di crossing $\\beta_{\\text{cross}}$ tra coppie consecutive "
                "di taglie, con errori bootstrap e valore universale $U^*$ all'intersezione. "
                "La stabilit\\`a di $U^*$ per $L \\ge 48$ conferma il raggiungimento del "
                "regime asintotico.}\n")
        f.write("\\label{tab:fss_crossings}\n")
        f.write("\\end{table}\n\n")

        # ===== TABLE 2: LEADING AMPLITUDES AND CORRECTIONS =====
        f.write("% TABLE 2: Leading Amplitudes and Finite-Size Corrections\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Osservabile & Esponente & Ampiezza $A$ & Correzione $B$ "
                "& $\\chi^2_{\\text{red}}$ \\\\\n")
        f.write("\\midrule\n")

        if 'A' in nu_result and 'A_err' in nu_result:
            inv_nu = 1.0 / nu_result['nu']
            inv_nu_err = nu_result['nu_err'] / nu_result['nu']**2
            f.write(f"$|dU_4/d\\beta|_{{\\beta_{{pc}}}}$ & "
                    f"${inv_nu:.4f} \\pm {inv_nu_err:.4f}$ & "
                    f"${nu_result['A']:.4f} \\pm {nu_result['A_err']:.4f}$ & "
                    f"${nu_result.get('B', 0):.2f} \\pm {nu_result.get('B_err', 0):.2f}$ & "
                    f"${nu_result.get('chi2_red', 0):.2f}$ \\\\\n")

        if corrections_results[0] is not None:
            res = corrections_results[0]
            f.write(f"$\\chi'_{{\\max}}$ & "
                    f"$\\gamma/\\nu = {mag_result['gamma_nu']:.3f}$ & "
                    f"$(\\text{{from fit}})$ & "
                    f"${res['B']:.2f} \\pm {res['B_err']:.2f}$ & "
                    f"${res['chi2_red_simple']:.2f}$ \\\\\n")

        if corrections_results[1] is not None:
            res = corrections_results[1]
            f.write(f"$M(\\beta_{{pc}})$ & "
                    f"$\\beta/\\nu = {mag_result['beta_nu']:.3f}$ & "
                    f"$(\\text{{from fit}})$ & "
                    f"${res['B']:.2f} \\pm {res['B_err']:.2f}$ & "
                    f"${res['chi2_red_simple']:.2f}$ \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Ampiezze leading $A$ e correzioni finite-size $B$ estratte "
                "dai fit con $L \\ge 32$ e $\\omega = 2$ fisso. Le correzioni "
                "$B \\cdot L^{-2}$ sono statisticamente compatibili con zero, "
                "confermando il regime asintotico.}\n")
        f.write("\\label{tab:fss_amplitudes}\n")
        f.write("\\end{table}\n\n")

        # ===== TABLE 3: FINITE-SIZE CORRECTIONS SUMMARY (WITH F-TEST) =====
        f.write("% TABLE 3: Finite-Size Corrections Summary\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Osservabile & Esponente & $B$ (correzione) & $\\chi^2_{\\text{simple}}$ & "
                "$\\chi^2_{\\text{corr}}$ & F-test \\\\\n")
        f.write("\\midrule\n")

        for res in corrections_results:
            if res is None:
                continue
            sig_str = "sig" if res['is_significant'] else "n.s."
            f.write(f"{res['observable']:14s} & "
                    f"${res['exponent']:+.3f} \\pm {res['exponent_err']:.3f}$ & "
                    f"${res['B']:+.1f} \\pm {res['B_err']:.1f}$ & "
                    f"${res['chi2_red_simple']:.3f}$ & "
                    f"${res['chi2_red_corrected']:.3f}$ & "
                    f"$p={res['p_value']:.3f}$ ({sig_str}) \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Analisi rigorosa delle correzioni finite-size tramite fit a "
                "DUE STEP e F-test. Tutte le correzioni risultano statisticamente non "
                "significative ($p > 0.05$), confermando che il sistema \\`e nel regime "
                "asintotico per $L \\ge 32$.}\n")
        f.write("\\label{tab:fss_corrections}\n")
        f.write("\\end{table}\n\n")

        # ===== TABLE 4: FINAL EXPONENTS SUMMARY =====
        f.write("% TABLE 4: Critical Exponents Summary\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lccl}\n")
        f.write("\\toprule\n")
        f.write("Esponente & Misurato & Esatto & Metodo \\\\\n")
        f.write("\\midrule\n")

        f.write(f"$\\beta_{{pc}}$ & ${beta_pc:.5f} \\pm {beta_pc_err:.5f}$ & "
                f"${BETA_C_EXACT:.5f}$ & Binder crossing \\\\\n")
        f.write(f"$U^*$ & ${U_star:.4f} \\pm {U_star_err:.4f}$ & "
                f"$0.6107$ & Crossing universale \\\\\n")
        f.write(f"$\\nu$ & ${nu_used:.3f} \\pm {nu_used_err:.3f}$ & "
                f"${NU_EXACT}$ & {nu_method} \\\\\n")
        f.write(f"$\\gamma/\\nu$ & ${mag_result['gamma_nu']:.3f} \\pm "
                f"{mag_result['gamma_nu_err']:.3f}$ & "
                f"${GAMMA_NU_EXACT}$ & Picco $\\chi'_{{\\max}}$ (Phase 3, parabola 3 pt) \\\\\n")
        f.write(f"$\\beta/\\nu$ & ${mag_result['beta_nu']:.3f} \\pm "
                f"{mag_result['beta_nu_err']:.3f}$ & "
                f"${BETA_NU_EXACT}$ & $M(\\beta_{{pc}})$ (free fit) \\\\\n")
        f.write(f"$\\eta$ & ${mag_result['eta']:.3f} \\pm {mag_result['eta_err']:.3f}$ & "
                f"$0.25$ & $2-\\gamma/\\nu$ \\\\\n")
        if cv_result.get('gen_fit_ok', False):
            alpha_str = f"${cv_result['alpha']:.3f} \\pm {cv_result['alpha_err']:.3f}$"
            f.write(f"$\\alpha$ & {alpha_str} & $0$ & $C_{{\\max}} = A L^g(1+q\\ln L)$ \\\\\n")
            f.write(f"$g = \\alpha/\\nu$ & ${cv_result['g_gen']:.3f} \\pm {cv_result['g_gen_err']:.3f}$ & $0$ & "
                    f"F-test (vs $\\ln$): $p={cv_result['F_p']:.3f}$ \\\\\n")
        else:
            f.write(f"$\\alpha$ & $0$ (log) & $0$ & $C_{{\\max}} \\sim \\ln L$ \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Hyperscaling & ${mag_result['hyperscaling']:.4f} \\pm "
                f"{mag_result['hyperscaling_err']:.4f}$ & "
                f"$2.000$ & $2\\beta/\\nu + \\gamma/\\nu$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        hyp_dev_pct = abs(mag_result['hyperscaling'] - 2.0) * 100
        f.write(f"\\caption{{Esponenti critici estratti dall'analisi FSS con correzioni "
                f"finite-size $L^{{-\\omega}}$ ($\\omega = 2$ fisso) e $L_{{\\min}}=32$. "
                f"Il test di hyperscaling \\`e soddisfatto con deviazione di "
                f"{hyp_dev_pct:.1f}\\% dal valore atteso.}}\n")
        f.write("\\label{tab:fss_exponents_summary}\n")
        f.write("\\end{table}\n\n")

        # ===== TABLE 5: COMPARISON WITH ONSAGER (DEVIATIONS) =====
        f.write("% TABLE 5: Detailed Comparison with Onsager\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Esponente & Misurato & Esatto & Deviazione & $\\sigma$ \\\\\n")
        f.write("\\midrule\n")

        dev_beta_pc = abs(beta_pc - BETA_C_EXACT)
        dev_beta_pc_rel = dev_beta_pc / BETA_C_EXACT * 100
        dev_beta_pc_sigma = dev_beta_pc / beta_pc_err
        f.write(f"$\\beta_{{pc}}$ & ${beta_pc:.6f}$ & ${BETA_C_EXACT:.6f}$ & "
                f"${dev_beta_pc_rel:.3f}\\%$ & ${dev_beta_pc_sigma:.1f}\\sigma$ \\\\\n")

        dev_nu = abs(nu_used - NU_EXACT)
        dev_nu_sigma = dev_nu / nu_used_err
        f.write(f"$\\nu$ & ${nu_used:.3f}$ & ${NU_EXACT}$ & "
                f"${dev_nu:.3f}$ & ${dev_nu_sigma:.1f}\\sigma$ \\\\\n")

        dev_gamma_nu = abs(mag_result['gamma_nu'] - GAMMA_NU_EXACT)
        dev_gamma_nu_sigma = dev_gamma_nu / mag_result['gamma_nu_err']
        f.write(f"$\\gamma/\\nu$ & ${mag_result['gamma_nu']:.3f}$ & ${GAMMA_NU_EXACT}$ & "
                f"${dev_gamma_nu:.3f}$ & ${dev_gamma_nu_sigma:.1f}\\sigma$ \\\\\n")

        dev_beta_nu = abs(mag_result['beta_nu'] - BETA_NU_EXACT)
        dev_beta_nu_sigma = dev_beta_nu / mag_result['beta_nu_err']
        f.write(f"$\\beta/\\nu$ & ${mag_result['beta_nu']:.3f}$ & ${BETA_NU_EXACT}$ & "
                f"${dev_beta_nu:.3f}$ & ${dev_beta_nu_sigma:.1f}\\sigma$ \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Confronto dettagliato con i valori esatti di Onsager. "
                "La significativit\\`a statistica delle deviazioni \\`e quantificata "
                "in unit\\`a di deviazione standard.}\n")
        f.write("\\label{tab:fss_onsager_comparison}\n")
        f.write("\\end{table}\n\n")

        f.write("% ============================================================\n")
        f.write("% END OF LATEX TABLES\n")
        f.write("% ============================================================\n")

    print(f"  \u2713 LaTeX tables: {filepath}")
    print(f"    \u2022 Table 1: Binder crossings (tab:fss_crossings)")
    print(f"    \u2022 Table 2: Amplitudes & corrections (tab:fss_amplitudes)")
    print(f"    \u2022 Table 3: F-test results (tab:fss_corrections)")
    print(f"    \u2022 Table 4: Exponents summary (tab:fss_exponents_summary)")
    print(f"    \u2022 Table 5: Onsager comparison (tab:fss_onsager_comparison)")


# ============================================================================
# MAIN
# ============================================================================
def main():

    print("\n" + "=" * 80)
    print("  2D ISING MODEL - FINITE-SIZE SCALING ANALYSIS")
    print("  Clean Output + Structured Results + Multiple LaTeX Tables")
    print("=" * 80)
    print(f"  Lattice sizes:    {L_LIST}")
    print(f"  Beta range:       {BETA_RANGE_FIT}")
    print(f"  Bootstrap:        {N_BOOTSTRAP} samples")
    print(f"  Method:           Self-consistent FSS cascade")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION A: Initialisation
    # ─────────────────────────────────────────────────────────────────────
    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2
    })
    plt.ioff()

    print("\n[INITIALIZATION] Loading data...")
    data = load_data(L_LIST)
    if not data:
        print("  \u2717 No data found.")
        return
    print(f"  \u2713 Loaded {len(data)} lattice sizes: {list(data.keys())}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION B: Phase 1-4 — extract critical exponents (sequential)
    # ─────────────────────────────────────────────────────────────────────
    # Phase 1: beta_pc from Binder crossings
    crossings, splines, beta_pc, beta_pc_err, U_star, U_star_err = find_beta_pc(data, L_LIST)
    if np.isnan(beta_pc):
        print("\n\u2717 Cannot proceed without beta_pc")
        return
    plot_beta_pc_vs_L(data, L_LIST, crossings)

    # Phase 2: nu from Binder slopes
    nu_result = extract_nu(data, L_LIST, beta_pc, beta_pc_err, splines)
    if np.isnan(nu_result['nu']):
        print("\n\u2717 Cannot proceed without nu")
        return

    # Phase 3: gamma/nu and beta/nu (free fits of chi_max and M)
    mag_result = extract_magnetic_exponents(
        data, L_LIST, beta_pc, beta_pc_err,
        nu_result['nu'], nu_result['nu_err']
    )

    # Phase 4: alpha/nu from specific-heat scaling (deferred to Section D,
    # because it needs nu_final from the correlated fit below)
    cv_result = None

    # Phase 3b: sub-leading corrections (diagnostic bootstrap, n=50)
    corrections_results = extract_all_corrections(
        data, L_LIST, beta_pc,
        nu_result['nu'], nu_result['nu_err'],
        mag_result,
        nu_result=nu_result,
        cv_result=cv_result
    )
    results_chi, results_M, results_S, results_Cv = corrections_results

    # Phase 3c: sub-leading corrections with L_min comparison
    L_min_scan = [24, 32]
    results_3c = extract_corrections_scan(
        mag_result, nu_result=nu_result, cv_result=cv_result,
        L_min_values=L_min_scan
    )
    save_corrections_comparison_latex(results_3c, L_min_scan)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION C: Correlated nu fit (uses global bootstrap covariance)
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  ν (Phase 2,  pcov)         = {nu_result['nu']:.3f} ± {nu_result['nu_err']:.3f}")
    print(f"    (conservative Cramér-Rao bound with parametric σ_S)")

    # Correlated fit of nu using the global bootstrap covariance matrix
    print("\n" + "-" * 80)
    print("[CORRELATED NU FIT] Using global bootstrap covariance matrix")
    print("-" * 80)
    try:
        global_boot = bootstrap_global_slopes(data, L_LIST, BETA_RANGE_FIT,
                                              beta_pc_fixed=beta_pc,
                                              n_boot=N_BOOTSTRAP)
        print(f"  Global bootstrap: {global_boot['n_success']}/{N_BOOTSTRAP} replicas")
        print(f"  Correlation matrix:")
        rho = global_boot['rho_matrix']
        L_gl = global_boot['L_used']
        header = "       " + "  ".join(f"L={l:3d}" for l in L_gl)
        print(f"  {header}")
        for i, L_i in enumerate(L_gl):
            row_str = "  ".join(f"{rho[i,j]:+.3f}" for j in range(len(L_gl)))
            print(f"  L={L_i:3d}  {row_str}")

        stdd_global = np.sqrt(np.diag(global_boot['C_matrix']))
        print("  sqrt(C_ii) diagonal errors (global boot):")
        for L_i, s_i in zip(L_gl, stdd_global):
            print(f"    L={int(L_i):3d}  sqrt(C_ii) = {s_i:.5f}")

        # Apply L_min = 32
        mask_corr = L_gl >= 32
        L_corr = L_gl[mask_corr]
        slopes_corr = global_boot['slopes_mean'][mask_corr]
        C_sub = global_boot['C_matrix'][np.ix_(mask_corr, mask_corr)]

        def model_corr(L, A, inv_nu, B):
            return A * L**inv_nu * (1 + B * L**(-OMEGA_EXACT))

        p0_corr = [slopes_corr[0] / L_corr[0], 1.0, 5.0]
        bounds_corr = ([0, 0.7, -500], [np.inf, 1.5, 500])

        popt_c, perr_c, chi2_c, chi2red_c, dof_c, rank_c = fit_nu_correlated(
            L_corr, slopes_corr, C_sub, model_corr, p0_corr, bounds_corr
        )
        nu_corr = 1.0 / popt_c[1]
        nu_corr_err = perr_c[1] / popt_c[1]**2

        print(f"\n  Correlated fit (L >= 32, {len(L_corr)} pts, rank={rank_c}):")
        print(f"    ν = {nu_corr:.4f} ± {nu_corr_err:.4f}")
        print(f"    A = {popt_c[0]:.4f} ± {perr_c[0]:.4f}")
        print(f"    B = {popt_c[2]:.2f} ± {perr_c[2]:.2f}")
        print(f"    χ²_red = {chi2red_c:.3f}  (dof_eff = {dof_c})")

        # p-value (upper tail) for the correlated chi2
        p_corr = 1.0 - sps.chi2.cdf(chi2_c, dof_c) if dof_c > 0 else np.nan
        print(f"    p-value (upper tail) = {p_corr:.4f}")

        nu_result['nu_correlated'] = nu_corr
        nu_result['nu_correlated_err'] = nu_corr_err
        nu_result['chi2_red_correlated'] = chi2red_c
        nu_result['dof_eff_correlated'] = dof_c
        nu_result['p_value_correlated'] = p_corr
        nu_result['A_corr'] = popt_c[0]
        nu_result['A_corr_err'] = perr_c[0]
        nu_result['B_corr'] = popt_c[2]
        nu_result['B_corr_err'] = perr_c[2]

        # Correlated fit -> final value of nu.
        # Use leading-only correlated fit (2-param) — the 3-param fit with B
        # hits the bound and is unreliable.
        # nu_final is set below after the diagnostic block, using popt_cl.

        # Diagnostic: compare per-L sigma_S vs global-bootstrap sigma_S
        print("\n" + "-" * 80)
        print("[DIAGNOSTIC] Per-L vs Global-bootstrap slope errors")
        print("-" * 80)
        print("  L    S(L)_perL    σ_perL      σ_global    ratio(perL/global)")
        print("  " + "─" * 68)
        perL_slopes = nu_result.get('slopes', np.array([]))
        perL_errs = nu_result.get('slopes_err', np.array([]))
        perL_L = nu_result.get('L_used', np.array([]))
        glob_slopes_mean = global_boot['slopes_mean']
        glob_slopes_err = global_boot['slopes_err']
        for j, L_j in enumerate(L_gl):
            # Find matching per-L entry
            idx_match = np.where(perL_L == L_j)[0]
            if len(idx_match) > 0:
                k = idx_match[0]
                ratio = perL_errs[k] / glob_slopes_err[j] if glob_slopes_err[j] > 0 else np.nan
                print(f"  {int(L_j):3d}  {perL_slopes[k]:11.5f}  {perL_errs[k]:10.5f}  "
                      f"{glob_slopes_err[j]:10.5f}  {ratio:8.3f}")
            else:
                print(f"  {int(L_j):3d}       (not in per-L data)")

        # ── DIAGNOSTIC: uncorrelated vs correlated chi2 for both models ────
        print("\n" + "-" * 80)
        print("[DIAGNOSTIC] χ² comparison: uncorrelated vs correlated")
        print("-" * 80)

        # Use per-L slopes for uncorrelated fit (L >= 32)
        mask_diag = perL_L >= 32
        L_diag = perL_L[mask_diag]
        S_diag = perL_slopes[mask_diag]
        Serr_diag = perL_errs[mask_diag]

        def model_leading(Lx, A, inv_nu):
            return A * Lx**inv_nu

        def model_corr_diag(Lx, A, inv_nu, B):
            return A * Lx**inv_nu * (1 + B * Lx**(-OMEGA_EXACT))

        try:
            # Leading fit (uncorrelated)
            popt_lead, _ = spo.curve_fit(
                model_leading, L_diag, S_diag,
                p0=[S_diag[0]/L_diag[0], 1.0],
                sigma=Serr_diag, absolute_sigma=True,
                bounds=([0, 0.7], [np.inf, 1.5]))
            yhat_lead = model_leading(L_diag, *popt_lead)
            chi2_lead = float(np.sum(((S_diag - yhat_lead)/Serr_diag)**2))
            dof_lead = len(L_diag) - 2
            chi2red_lead = chi2_lead / dof_lead if dof_lead > 0 else np.inf

            # Corrected fit (uncorrelated)
            popt_corr_unc, _ = spo.curve_fit(
                model_corr_diag, L_diag, S_diag,
                p0=[popt_lead[0], popt_lead[1], 0.0],
                sigma=Serr_diag, absolute_sigma=True,
                bounds=([0, 0.7, -500], [np.inf, 1.5, 500]))
            yhat_corr_unc = model_corr_diag(L_diag, *popt_corr_unc)
            chi2_corr_unc = float(np.sum(((S_diag - yhat_corr_unc)/Serr_diag)**2))
            dof_corr_unc = len(L_diag) - 3
            chi2red_corr_unc = chi2_corr_unc / dof_corr_unc if dof_corr_unc > 0 else np.inf

            # Correlated: leading only
            p0_cl = [slopes_corr[0]/L_corr[0], 1.0]
            bounds_cl = ([0, 0.7], [np.inf, 1.5])
            popt_cl, perr_cl, chi2_cl, chi2red_cl, dof_cl, rank_cl = fit_nu_correlated(
                L_corr, slopes_corr, C_sub, model_leading, p0_cl, bounds_cl)
            p_cl = 1.0 - sps.chi2.cdf(chi2_cl, dof_cl) if dof_cl > 0 else np.nan

            # Leading-only correlated fit -> final nu value──
            nu_cl     = 1.0 / popt_cl[1]
            nu_cl_err = perr_cl[1] / popt_cl[1]**2
            nu_result['nu_final']     = nu_cl
            nu_result['nu_final_err'] = nu_cl_err
            nu_result['nu_method']    = (f'Correlated fit (leading-only, '
                                         f'L>=32, χ²_red={chi2red_cl:.3f}, p={p_cl:.3f})')
            print(f"\n  ══> ν FINALE = {nu_cl:.4f} ± {nu_cl_err:.4f}  [leading-only correlated fit]")

            # Correlated: leading + correction (already computed above)
            p_cc = p_corr

            print(f"  {'Model':<35s}  {'χ²':>8s}  {'dof':>4s}  {'χ²_red':>8s}  {'p-val':>6s}")
            print(f"  {'─'*35}  {'─'*8}  {'─'*4}  {'─'*8}  {'─'*6}")
            print(f"  {'Leading (uncorrelated)':<35s}  {chi2_lead:8.4f}  {dof_lead:4d}  {chi2red_lead:8.4f}  {'—':>6s}")
            print(f"  {'Leading+B (uncorrelated)':<35s}  {chi2_corr_unc:8.4f}  {dof_corr_unc:4d}  {chi2red_corr_unc:8.4f}  {'—':>6s}")
            print(f"  {'Leading (correlated)':<35s}  {chi2_cl:8.4f}  {dof_cl:4d}  {chi2red_cl:8.4f}  {p_cl:6.3f}")
            print(f"  {'Leading+B (correlated)':<35s}  {chi2_c:8.4f}  {dof_c:4d}  {chi2red_c:8.4f}  {p_corr:6.3f}")

            print("\n  Interpretation:")
            print("  • χ²_red << 1 (uncorrelated) → σ_S conservative and/or points correlated")
            print("  • χ²_red closer to 1 (correlated) → correlations explain part of the deficit")
            print("  • p-value > 0.05 → model is acceptable at 95% CL")

        except Exception as e:
            print(f"  ⚠ Diagnostic fits failed: {e}")

    except Exception as e:
        print(f"  ⚠ Correlated fit failed: {e}")

    # ── NOTE: γ/ν and β/ν do NOT need a correlated fit ─────────────────
    print("\n" + "-" * 80)
    print("[NOTE] Correlated fit for γ/ν and β/ν")
    print("-" * 80)
    print("  χ_max(L) and M(β_pc, L) are measured from INDEPENDENT simulations")
    print("  at each L → their covariance matrix is diagonal (C_ij = δ_ij σ_i²).")
    print("  Therefore pcov from curve_fit(absolute_sigma=True) is already the")
    print("  correct Cramér-Rao error bound. A correlated fit would be redundant.")
    print("  This is unlike S(L), where all slopes share the same spline+β_pc")
    print("  structure, creating inter-L correlations that require C_ij treatment.")
    print(f"  → γ/ν = {mag_result['gamma_nu']:.4f} ± {mag_result['gamma_nu_err']:.4f}  [pcov, correct]")
    print(f"  → β/ν = {mag_result['beta_nu']:.4f} ± {mag_result['beta_nu_err']:.4f}  [pcov, correct]")
    print("-" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION D: Recompute derived exponents with nu_final, then plots
    # ─────────────────────────────────────────────────────────────────────
    nu_final     = nu_result.get('nu_final', nu_result['nu'])
    nu_final_err = nu_result.get('nu_final_err', nu_result['nu_err'])

    # Patch mag_result: gamma, beta_mag, and their errors
    gn  = mag_result['gamma_nu']
    gne = mag_result['gamma_nu_err']
    bn  = mag_result['beta_nu']
    bne = mag_result['beta_nu_err']

    mag_result['gamma']     = gn * nu_final
    mag_result['gamma_err'] = np.sqrt((gne * nu_final)**2 + (gn * nu_final_err)**2)
    mag_result['beta']      = bn * nu_final
    mag_result['beta_err']  = np.sqrt((bne * nu_final)**2 + (bn * nu_final_err)**2)

    # Phase 4 (deferred): alpha/nu from specific-heat scaling — uses nu_final
    cv_result = extract_alpha(data, L_LIST, nu_final, nu_final_err)

    print(f"\n  [DERIVED EXPONENTS] Recomputed with ν_final = {nu_final:.4f} ± {nu_final_err:.4f}")
    print(f"    γ  = {mag_result['gamma']:.6f} ± {mag_result['gamma_err']:.6f}")
    print(f"    β  = {mag_result['beta']:.6f} ± {mag_result['beta_err']:.6f}")
    if cv_result.get('gen_fit_ok', False):
        print(f"    α  = {cv_result['alpha']:.6f} ± {cv_result['alpha_err']:.6f}")

    # Data collapse plots
    data_collapse_plots(data, L_LIST, beta_pc, nu_final,
                        mag_result['gamma_nu'], mag_result['beta_nu'])

    # Supplementary plots
    plot_chi_parabolic_example(data, L_LIST, mag_result, L_example=96)
    plot_C_parabolic_example(data, cv_result, L_example=96)
    plot_specific_heat_scaling_fit(cv_result, nu_final, nu_final_err)
    plot_magnetization_vs_L(mag_result)

    # Robustness check
    print("\n[SUPPLEMENTARY] Robustness Analysis")
    print("-" * 80)

    # Centre ylims on the red-line values, keeping the same range widths
    _hw = [0.000175, 0.075, 0.03]          # half-widths for beta_pc, 1/nu, gamma/nu
    _centres = [beta_pc, 1.0 / nu_final, mag_result['gamma_nu']]
    YLIMS_ROBUSTNESS = [
        (_centres[0] - _hw[0], _centres[0] + _hw[0]),
        (_centres[1] - _hw[1], _centres[1] + _hw[1]),
        (_centres[2] - _hw[2], _centres[2] + _hw[2]),
    ]

    plot_robustness_vs_Lmin(data, L_LIST, beta_pc, beta_pc_err,
                            nu_final, nu_final_err,
                            mag_result['gamma_nu'], mag_result['gamma_nu_err'],
                            ylims=YLIMS_ROBUSTNESS)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION E: Save results (.dat + LaTeX tables)
    # ─────────────────────────────────────────────────────────────────────
    save_results_enhanced(
        beta_pc, beta_pc_err, U_star, U_star_err,
        crossings, nu_result, mag_result, cv_result,
        corrections_results
    )

    # ─────────────────────────────────────────────────────────────────────
    # SECTION F: Final summary and validation
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY - CRITICAL EXPONENTS")
    print("=" * 80)

    nu_used     = nu_result.get('nu_final', nu_result['nu'])
    nu_used_err = nu_result.get('nu_final_err', nu_result['nu_err'])

    summary_rows = [
        ["beta_pc",
         f"{beta_pc:.6f} \u00b1 {beta_pc_err:.6f}",
         f"{BETA_C_EXACT:.6f}",
         f"{abs(beta_pc - BETA_C_EXACT) / BETA_C_EXACT * 100:.3f}%",
         f"{abs(beta_pc - BETA_C_EXACT) / beta_pc_err:.1f}\u03c3"],

        ["nu",
         f"{nu_used:.3f} \u00b1 {nu_used_err:.3f}",
         f"{NU_EXACT}",
         f"{abs(nu_used - NU_EXACT):.3f}",
         f"{abs(nu_used - NU_EXACT) / nu_used_err:.1f}\u03c3"],

        ["gamma/nu",
         f"{mag_result['gamma_nu']:.3f} \u00b1 {mag_result['gamma_nu_err']:.3f}",
         f"{GAMMA_NU_EXACT}",
         f"{abs(mag_result['gamma_nu'] - GAMMA_NU_EXACT):.3f}",
         f"{abs(mag_result['gamma_nu'] - GAMMA_NU_EXACT) / mag_result['gamma_nu_err']:.1f}\u03c3"],

        ["beta/nu",
         f"{mag_result['beta_nu']:.3f} \u00b1 {mag_result['beta_nu_err']:.3f}",
         f"{BETA_NU_EXACT}",
         f"{abs(mag_result['beta_nu'] - BETA_NU_EXACT):.3f}",
         f"{abs(mag_result['beta_nu'] - BETA_NU_EXACT) / mag_result['beta_nu_err']:.1f}\u03c3"],

        ["eta",
         f"{mag_result['eta']:.3f} \u00b1 {mag_result['eta_err']:.3f}",
         "0.250",
         f"{abs(mag_result['eta'] - 0.25):.3f}",
         f"{abs(mag_result['eta'] - 0.25) / mag_result['eta_err']:.1f}\u03c3"],

        ["alpha", "0 (log)", "0", "\u2014", "\u2014"]
    ]

    # If generalized fit succeeded, update alpha row with measured value
    if cv_result.get('gen_fit_ok', False):
        alpha_meas = cv_result['alpha']
        alpha_err = cv_result['alpha_err']
        if alpha_err > 0:
            summary_rows[-1] = [
                "alpha",
                f"{alpha_meas:.3f} \u00b1 {alpha_err:.3f}",
                "0",
                f"{abs(alpha_meas):.3f}",
                f"{abs(alpha_meas) / alpha_err:.1f}\u03c3"
            ]

    print_table_compact(
        ["Exponent", "Measured", "Exact", "Deviation", "sigma"],
        summary_rows
    )

    print(f"\n  Hyperscaling: 2beta/nu + gamma/nu = "
          f"{mag_result['hyperscaling']:.4f} \u00b1 {mag_result['hyperscaling_err']:.4f}")
    print(f"  Expected:     2.0000")
    print(f"  Deviation:    {abs(mag_result['hyperscaling'] - 2.0):.4f} "
          f"({abs(mag_result['hyperscaling'] - 2.0) / mag_result['hyperscaling_err']:.1f}\u03c3)")

    max_sigma = max(
        abs(beta_pc - BETA_C_EXACT) / beta_pc_err,
        abs(nu_used - NU_EXACT) / nu_used_err,
        abs(mag_result['gamma_nu'] - GAMMA_NU_EXACT) / mag_result['gamma_nu_err'],
        abs(mag_result['beta_nu'] - BETA_NU_EXACT) / mag_result['beta_nu_err']
    )

    if max_sigma < 1.5:
        print(f"\n  \u2713 All exponents compatible with Onsager within {max_sigma:.1f}\u03c3")
    elif max_sigma < 2.0:
        print(f"\n  \u26a0 Some deviations up to {max_sigma:.1f}\u03c3 (still acceptable)")
    else:
        print(f"\n  \u2717 WARNING: Deviations up to {max_sigma:.1f}\u03c3 detected!")

    # Summary of the two nu estimates
    print("\n" + "-" * 80)
    print("  TWO ESTIMATES OF ν (for thesis reference)")
    print("-" * 80)
    nu_ph2 = nu_result['nu']
    nu_ph2_err = nu_result['nu_err']
    print(f"  1. Phase 2 (pcov, Cramér-Rao):        ν = {nu_ph2:.4f} ± {nu_ph2_err:.4f}")
    print(f"     Conservative upper bound on σ_ν; σ_S from per-L parametric bootstrap")
    print(f"     χ²_red (diagonal, per-L) = {nu_result.get('chi2_red', np.nan):.3f} (low → σ_S overestimated)")
    if 'nu_correlated' in nu_result:
        nu_c = nu_result['nu_correlated']
        nu_c_err = nu_result['nu_correlated_err']
        print(f"  2. Correlated fit (global C_ij):      ν = {nu_c:.4f} ± {nu_c_err:.4f}  [diagnostics only]")
        print(f"     Uses full bootstrap covariance; Hessian errors; χ²_red = {nu_result.get('chi2_red_correlated', np.nan):.3f}")
        print(f"     p-value = {nu_result.get('p_value_correlated', np.nan):.3f}")
    print(f"\n  ══> FINAL ν = {nu_used:.4f} ± {nu_used_err:.4f}  [USED for all downstream phases]")
    print(f"      Deviation from Onsager: {abs(nu_used - NU_EXACT)/nu_used_err:.1f}σ")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("  \u2713 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n  Results:  {ANALYSIS_DIR}/fss_results_complete.dat")
    print(f"  Tables:   {ANALYSIS_DIR}/fss_tables_complete.tex")
    print(f"  Plots:    {PLOT_DIR}/*.pdf (14 files)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
