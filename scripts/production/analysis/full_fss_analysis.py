#!/usr/bin/env python3
"""
Finite-size scaling (FSS) analysis for the 2D Ising model (PBC).

Pipeline: beta_pc (Binder crossing) -> nu (slope fit) -> gamma/nu, beta/nu
(peak scaling) -> alpha (specific-heat) -> data collapse.  Errors from
bootstrap + pcov; correlated chi^2 for nu.  Outputs .dat results
and 14 publication-quality PDF plots.
"""

import argparse
import json
import platform
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.stats as sps
import scipy
import seaborn as sns
from pathlib import Path
from datetime import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from fss_joint_correlations import (
    JOINT_REPLICAS,
    JOINT_SEED,
    ensure_official_cache,
    run_joint_bootstrap,
    save_joint_replicas,
)

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
L_MIN_OFFICIAL = 32
EXPECTED_PLOTS = {
    "binder_crossing_with_inset.pdf",
    "beta_pc_vs_L_convergence.pdf",
    "nu_extraction_loglog.pdf",
    "exponent_fits_magnetic.pdf",
    "specific_heat_log.pdf",
    "collapse_susceptibility_inset.pdf",
    "collapse_magnetization_inset.pdf",
    "collapse_binder_inset.pdf",
    "collapse_cv_log.pdf",
    "robustness_vs_Lmin.pdf",
    "chi_parabolic_fit_example.pdf",
    "C_parabolic_fit_example.pdf",
    "specific_heat_scaling_fit.pdf",
    "magnetization_vs_L.pdf",
}


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
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError("primary Binder crossing root not found") from exc

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
        except (ValueError, RuntimeError):
            # A resampled grid may have no root; failed bootstrap replicas are
            # discarded by the original validated estimator.
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
            except (ValueError, RuntimeError):
                ok = False
                break
        if not ok:
            continue

        row = []
        for L in L_used:
            try:
                row.append(abs(splines_b[L].derivative()(beta_pc_fixed)))
            except (ValueError, RuntimeError):
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
    if not res.success or not np.isfinite(res.x).all():
        raise RuntimeError(f"correlated fit failed: {res.message}")

    # Numerical Hessian for parameter errors
    eps = 1e-5
    H = np.array([[approx_fprime(res.x,
                    lambda p: approx_fprime(p, chi2c, eps)[j], eps)[i]
                   for j in range(len(p0))] for i in range(len(p0))])
    try:
        perr = np.sqrt(np.abs(np.diag(np.linalg.inv(H / 2.0))))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("singular Hessian in correlated fit") from exc
    if not np.isfinite(perr).all():
        raise RuntimeError("non-finite correlated-fit uncertainties")

    chi2_red_eff = res.fun / dof_eff if dof_eff > 0 else np.inf
    return res.x, perr, res.fun, chi2_red_eff, dof_eff, rank


# ============================================================================
# NESTED-MODEL DIAGNOSTIC
# ============================================================================
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


def extract_nu_official(data, beta_pc):
    """Official correlated leading-only extraction of 1/nu and nu."""
    print_phase_header(2, 5, "CORRELATION LENGTH EXPONENT nu",
                       description="Official correlated leading-only fit of Binder slopes")
    global_boot = bootstrap_global_slopes(
        data, L_LIST, BETA_RANGE_FIT, beta_pc_fixed=beta_pc,
        n_boot=N_BOOTSTRAP, seed=42,
    )
    if global_boot['n_success'] != N_BOOTSTRAP:
        raise RuntimeError(
            f"Binder-slope bootstrap incomplete: {global_boot['n_success']}/{N_BOOTSTRAP}"
        )
    selected = global_boot['L_used'] >= L_MIN_OFFICIAL
    L_fit = global_boot['L_used'][selected]
    slopes = global_boot['slopes_mean'][selected]
    covariance = global_boot['C_matrix'][np.ix_(selected, selected)]

    def leading(size, amplitude, inv_nu):
        return amplitude * size**inv_nu

    lead = fit_nu_correlated(
        L_fit, slopes, covariance, leading,
        [slopes[0] / L_fit[0], 1.0], ([0.0, 0.7], [np.inf, 1.5]),
    )
    popt, perr, chi2_value, chi2_red, dof, rank = lead
    inv_nu = float(popt[1])
    inv_nu_err = float(perr[1])
    nu = 1.0 / inv_nu
    nu_err = inv_nu_err / inv_nu**2
    p_value = float(sps.chi2.sf(chi2_value, dof))

    def subleading(size, amplitude, inverse_nu, correction):
        return amplitude * size**inverse_nu * (1.0 + correction * size**(-OMEGA_EXACT))

    complex_fit = fit_nu_correlated(
        L_fit, slopes, covariance, subleading,
        [popt[0], popt[1], 0.0], ([0.0, 0.7, -500.0], [np.inf, 1.5, 500.0]),
    )
    popt_sub, perr_sub, chi2_sub, chi2_red_sub, dof_sub, _rank_sub = complex_fit
    F_value, F_p, F_significant = f_test_nested_models(
        chi2_value, dof, chi2_sub, dof_sub
    )

    print_result("1/nu (official)", inv_nu, inv_nu_err, chi2=chi2_red)
    print_result("nu = 1/(1/nu)", nu, nu_err)
    print(f"    chi2={chi2_value:.9f}, dof={dof}, chi2_red={chi2_red:.6f}, p={p_value:.6f}")
    print(f"    diagnostic leading+B: B={popt_sub[2]:.6g} +/- {perr_sub[2]:.6g}, "
          f"F={F_value:.6f}, p={F_p:.6f}, significant={F_significant}")

    # Preserve the existing plot style, feeding it the official fit and data.
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = get_colors(L_LIST)
    for index, L in enumerate(global_boot['L_used']):
        color_index = L_LIST.index(int(L))
        ax.errorbar(np.log(L), np.log(global_boot['slopes_mean'][index]),
                    yerr=global_boot['slopes_err'][index] / global_boot['slopes_mean'][index],
                    fmt='o', color=colors[color_index], ms=8, capsize=0, lw=1.5,
                    elinewidth=1.5, label=f'$L={L}$', zorder=3)
    L_plot = np.linspace(20, 140, 100)
    ax.plot(np.log(L_plot), np.log(leading(L_plot, *popt)),
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

    return {
        'inv_nu': inv_nu, 'inv_nu_err': inv_nu_err,
        'nu': nu, 'nu_err': nu_err,
        'A': float(popt[0]), 'A_err': float(perr[0]),
        'chi2': float(chi2_value), 'chi2_red': float(chi2_red),
        'dof': int(dof), 'rank': int(rank), 'p_value': p_value,
        'L_used': global_boot['L_used'],
        'slopes': global_boot['slopes_mean'],
        'slopes_err': global_boot['slopes_err'],
        'covariance': global_boot['C_matrix'],
        'official_params': popt, 'official_errors': perr,
        'subleading_params': popt_sub, 'subleading_errors': perr_sub,
        'F_test': {'F': F_value, 'p_value': F_p, 'significant': F_significant,
                   'chi2_complex': chi2_sub, 'chi2_red_complex': chi2_red_sub,
                   'dof_complex': dof_sub},
        'n_bootstrap': global_boot['n_success'],
        'slope_err_method': 'validated synchronized point bootstrap covariance, beta_pc fixed',
    }


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
                # Keep the validated operation ordering: pcov is sensitive to
                # roundoff in this ill-conditioned uncentred three-point fit.
                return c0 + c1 * beta + c2 * beta * beta

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
                except (ValueError, RuntimeError):
                    # Diagnostic spline-peak replica; not used by the primary fit.
                    continue
            chi_max_spl_err = np.std(spl_boot, ddof=1) if len(spl_boot) > 30 else np.nan

            chi_max_spline.append(chi_max_spl)
            chi_max_spline_err.append(chi_max_spl_err)

        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"primary chi_max parabolic fit failed for L={L}") from exc

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
                except (ValueError, RuntimeError):
                    continue

            if len(mag_boot) <= 30:
                raise RuntimeError(
                    f"insufficient M(beta_pc) bootstrap replicas: {len(mag_boot)}/{N_BOOTSTRAP}"
                )
            mag_err_val = np.std(mag_boot, ddof=1)

            mag_pc.append(mag_central)
            mag_pc_err.append(mag_err_val)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(f"primary M(beta_pc) spline/bootstrap failed for L={L}") from exc

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
    print_section("Susceptibility: official leading-only gamma/nu", level=2)

    L_min = 32
    mask = L_used >= L_min

    def model_chi_leading(L, A, gamma_nu):
        return A * L**gamma_nu

    popt_chi, pcov_chi = spo.curve_fit(
        model_chi_leading, L_used[mask], chi_max[mask], p0=[0.11, 1.75],
        sigma=chi_max_err[mask], absolute_sigma=True,
        bounds=([0.05, 1.5], [0.2, 2.0]),
    )
    A_chi, gamma_nu_measured = popt_chi
    errors_chi = np.sqrt(np.diag(pcov_chi))
    A_chi_err, gamma_nu_err = errors_chi
    pred_chi = model_chi_leading(L_used[mask], *popt_chi)
    dof_chi = int(np.sum(mask) - 2)
    chi2_val_chi = float(np.sum(((chi_max[mask] - pred_chi) / chi_max_err[mask])**2))
    chi2_chi = chi2_val_chi / dof_chi
    p_value_chi = float(sps.chi2.sf(chi2_val_chi, dof_chi))

    def model_chi_subleading(L, A, gamma_nu, B):
        return A * L**gamma_nu * (1.0 + B * L**(-OMEGA_EXACT))

    popt_chi_sub, _ = spo.curve_fit(
        model_chi_subleading, L_used[mask], chi_max[mask],
        p0=[A_chi, gamma_nu_measured, 0.0], sigma=chi_max_err[mask],
        absolute_sigma=True, bounds=([0.05, 1.5, -30], [0.2, 2.0, 30]),
    )
    chi2_chi_sub = float(np.sum(((chi_max[mask] - model_chi_subleading(
        L_used[mask], *popt_chi_sub)) / chi_max_err[mask])**2))
    F_chi, F_p_chi, F_sig_chi = f_test_nested_models(
        chi2_val_chi, dof_chi, chi2_chi_sub, int(np.sum(mask) - 3)
    )
    print_result("gamma/nu (official)", gamma_nu_measured, gamma_nu_err, chi2=chi2_chi)
    print(f"    dof={dof_chi}, chi2={chi2_val_chi:.6f}, p={p_value_chi:.6f}")
    print(f"    diagnostic leading+B: F={F_chi:.6f}, p={F_p_chi:.6f}, significant={F_sig_chi}")

    # beta/nu
    print_section("Magnetization: official leading-only beta/nu", level=2)

    valid = ~np.isnan(mag_pc) & (L_used >= L_min)

    def model_mag_leading(L, A, beta_nu):
        return A * L**(-beta_nu)

    popt_mag, pcov_mag = spo.curve_fit(
        model_mag_leading, L_used[valid], mag_pc[valid], p0=[1.0, 0.125],
        sigma=mag_pc_err[valid], absolute_sigma=True,
        bounds=([0.8, 0.08], [1.2, 0.18]),
    )
    A_mag, beta_nu_measured = popt_mag
    errors_mag = np.sqrt(np.diag(pcov_mag))
    A_mag_err, beta_nu_err = errors_mag
    pred_mag = model_mag_leading(L_used[valid], *popt_mag)
    dof_mag = int(np.sum(valid) - 2)
    chi2_val_mag = float(np.sum(((mag_pc[valid] - pred_mag) / mag_pc_err[valid])**2))
    chi2_mag = chi2_val_mag / dof_mag
    p_value_mag = float(sps.chi2.sf(chi2_val_mag, dof_mag))

    def model_mag_subleading(L, A, beta_nu, B):
        return A * L**(-beta_nu) * (1.0 + B * L**(-OMEGA_EXACT))

    popt_mag_sub, _ = spo.curve_fit(
        model_mag_subleading, L_used[valid], mag_pc[valid],
        p0=[A_mag, beta_nu_measured, 0.0], sigma=mag_pc_err[valid],
        absolute_sigma=True, bounds=([0.8, 0.08, -15], [1.2, 0.18, 15]),
    )
    chi2_mag_sub = float(np.sum(((mag_pc[valid] - model_mag_subleading(
        L_used[valid], *popt_mag_sub)) / mag_pc_err[valid])**2))
    F_mag, F_p_mag, F_sig_mag = f_test_nested_models(
        chi2_val_mag, dof_mag, chi2_mag_sub, int(np.sum(valid) - 3)
    )
    print_result("beta/nu (official)", beta_nu_measured, beta_nu_err, chi2=chi2_mag)
    print(f"    dof={dof_mag}, chi2={chi2_val_mag:.6f}, p={p_value_mag:.6f}")
    print(f"    diagnostic leading+B: F={F_mag:.6f}, p={F_p_mag:.6f}, significant={F_sig_mag}")

    # PLOT 4: Magnetic exponents
    print_section("Visualization", level=2)

    def model_chi_simple(L, A, gamma_nu):
        return A * L**gamma_nu

    def model_mag_simple(L, A, beta_nu):
        return A * L**(-beta_nu)

    # Plot exactly the official leading-only fits; do not refit for graphics.
    popt_chi_simple = popt_chi
    popt_mag_simple = popt_mag

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
        'official_chi_params': popt_chi,
        'official_chi_cov': pcov_chi,
        'official_mag_params': popt_mag,
        'official_mag_cov': pcov_mag,
        'subleading_chi_params': popt_chi_sub,
        'subleading_mag_params': popt_mag_sub,
        'F_test_chi': {'F': F_chi, 'p_value': F_p_chi, 'significant': F_sig_chi,
                       'chi2_complex': chi2_chi_sub, 'dof_complex': int(np.sum(mask) - 3)},
        'F_test_mag': {'F': F_mag, 'p_value': F_p_mag, 'significant': F_sig_mag,
                       'chi2_complex': chi2_mag_sub, 'dof_complex': int(np.sum(valid) - 3)},
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


# ============================================================================
# PHASE 3c: CORRECTIONS SCAN WITH VARIABLE L_min
# ============================================================================


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
                except (ValueError, RuntimeError):
                    # Diagnostic spline cross-check; primary C_max is parabolic.
                    continue
            cv_spl_err = np.std(spl_boot, ddof=1) if len(spl_boot) > 30 else np.nan

            cv_max_spline.append(cv_spl)
            cv_max_spline_err.append(cv_spl_err)

        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"primary C_max parabolic fit failed for L={L}") from exc

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
        except ValueError as exc:
            raise RuntimeError("specific-heat F-test failed") from exc

        gen_fit_ok = True

    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        raise RuntimeError("primary generalized specific-heat fit failed") from exc

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

    popt = np.asarray(mag_result['official_mag_params'])
    pcov_mag = np.asarray(mag_result['official_mag_cov'])
    chi2_red_mag = float(mag_result['chi2_mag'])

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


def plot_robustness_official(crossings, nu_result, mag_result,
                             beta_pc, beta_pc_err, ylims=None):
    """Diagnostic L_min scan using the same estimators as the primary fits."""
    print("\n  [PLOT 10/10] Generating robustness check (official estimators; diagnostic)")
    L_min_values = [24, 32, 48, 64]
    beta_c_vals, beta_c_errs = [], []
    inv_nu_vals, inv_nu_errs = [], []
    gamma_nu_vals, gamma_nu_errs = [], []

    for L_min in L_min_values:
        chosen = [crossing for crossing in crossings if crossing[0] >= L_min]
        errors = np.asarray([crossing[3] for crossing in chosen])
        weights = 1.0 / (errors**2 + 1e-12)
        beta_c_vals.append(float(weights @ np.asarray([c[2] for c in chosen]) / weights.sum()))
        beta_c_errs.append(float(1.0 / np.sqrt(weights.sum())))

        selected = nu_result['L_used'] >= L_min
        L_nu = nu_result['L_used'][selected]
        slopes = nu_result['slopes'][selected]
        covariance = nu_result['covariance'][np.ix_(selected, selected)]

        def leading_slope(size, amplitude, inverse_nu):
            return amplitude * size**inverse_nu

        fit = fit_nu_correlated(
            L_nu, slopes, covariance, leading_slope,
            [slopes[0] / L_nu[0], 1.0], ([0.0, 0.7], [np.inf, 1.5]),
        )
        inv_nu_vals.append(float(fit[0][1]))
        inv_nu_errs.append(float(fit[1][1]))

        selected_mag = mag_result['L_used'] >= L_min

        def leading_chi(size, amplitude, exponent):
            return amplitude * size**exponent

        params, covariance_chi = spo.curve_fit(
            leading_chi, mag_result['L_used'][selected_mag], mag_result['chi_max'][selected_mag],
            p0=[0.11, 1.75], sigma=mag_result['chi_max_err'][selected_mag],
            absolute_sigma=True, bounds=([0.05, 1.5], [0.2, 2.0]),
        )
        gamma_nu_vals.append(float(params[1]))
        gamma_nu_errs.append(float(np.sqrt(covariance_chi[1, 1])))

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

    inv_nu_final = nu_result['inv_nu']
    inv_nu_final_err = nu_result['inv_nu_err']
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
    ax3.axhline(mag_result['gamma_nu'], color='red', ls='-', lw=2, zorder=2)
    ax3.axhline(GAMMA_NU_EXACT, color='grey', ls='--', lw=0.8, alpha=0.7, zorder=1)
    ax3.axhspan(mag_result['gamma_nu'] - mag_result['gamma_nu_err'],
                mag_result['gamma_nu'] + mag_result['gamma_nu_err'],
                alpha=0.1, color='red', zorder=0)
    ax3.set_xlabel(r"$L_{\min}$", fontsize=18)
    ax3.set_ylabel(r"$\gamma/\nu$", fontsize=18)
    ax3.tick_params(labelsize=14)
    ax3.grid(alpha=0.3, ls=':')
    ax3.set_xlim(20, 68)
    if ylims:
        if ylims[0]: ax1.set_ylim(ylims[0])
        if ylims[1]: ax2.set_ylim(ylims[1])
        if ylims[2]: ax3.set_ylim(ylims[2])
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "robustness_vs_Lmin.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print("  \u2713 Saved: robustness_vs_Lmin.pdf")


# ============================================================================
# SAVE RESULTS (ENHANCED)
# ============================================================================


# ============================================================================
# MAIN
# ============================================================================

def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def report_joint_derived_quantities(joint_result) -> None:
    """Print the unique official derived results from calibrated replicas."""
    print_section("Joint derived exponents", level=2)
    labels = {
        "gamma": r"gamma = (gamma/nu) nu",
        "beta": r"beta = (beta/nu) nu",
        "eta": r"eta = 2 - gamma/nu",
        "delta": r"delta = 1 + (gamma/nu)/(beta/nu)",
        "d_hyper": r"Hyperscaling 2 beta/nu + gamma/nu",
    }
    for name in ("gamma", "beta", "eta", "delta", "d_hyper"):
        result = joint_result.summary["derived"][name]
        print_result(labels[name], result["central"], result["error"])

    hyper = joint_result.summary["hyperscaling_variance_decomposition"]
    significance = abs(
        joint_result.summary["derived"]["d_hyper"]["central"] - 2.0
    ) / joint_result.summary["derived"]["d_hyper"]["error"]
    print(f"    covariance contribution = {hyper['four_cov_gamma_nu_beta_nu']:.9g}")
    print(f"    variance-identity residual = {hyper['identity_residual']:.3e}")
    print(f"    deviation from d=2 = {significance:.2f} sigma")


def save_official_outputs(beta_pc, beta_pc_err, U_star, U_star_err, crossings,
                          nu_result, mag_result, cv_result, joint_result,
                          cache_manifest_path, cache_status):
    """Serialize the unique official result set to DAT, JSON and NPZ."""
    derived = joint_result.summary['derived']
    fit_quality = {
        'nu': {'chi2': nu_result['chi2'], 'chi2_red': nu_result['chi2_red'],
               'dof': nu_result['dof'], 'p_value': nu_result['p_value']},
        'gamma_nu': {'chi2': mag_result['chi2_val_chi'], 'chi2_red': mag_result['chi2_chi'],
                     'dof': mag_result['chi2_dof_chi'], 'p_value': mag_result['chi2_p_chi']},
        'beta_nu': {'chi2': mag_result['chi2_val_mag'], 'chi2_red': mag_result['chi2_mag'],
                    'dof': mag_result['chi2_dof_mag'], 'p_value': mag_result['chi2_p_mag']},
        'alpha': {'chi2': cv_result['chi2_val_gen'], 'chi2_red': cv_result['chi2_red_gen'],
                  'dof': cv_result['dof_gen'], 'p_value': cv_result['p_val_gen']},
    }
    F_tests = {
        'nu': nu_result['F_test'],
        'gamma_nu': mag_result['F_test_chi'],
        'beta_nu': mag_result['F_test_mag'],
        'specific_heat_log_vs_generalized': {
            'F': cv_result['F_val'], 'p_value': cv_result['F_p'],
            'significant': cv_result['F_sig'],
        },
    }
    payload = {
        'metadata': {
            'generated_at': datetime.now().astimezone().isoformat(),
            'analysis_version': '3.0',
            'observable_convention': 'reduced',
            'joint_bootstrap_role': 'correlation structure only',
        },
        'configuration': {
            'L': L_LIST, 'L_min_official': L_MIN_OFFICIAL,
            'beta_window': list(BETA_RANGE_FIT), 'omega_diagnostic': OMEGA_EXACT,
            'crossing_bootstrap_replicas': N_BOOTSTRAP,
            'joint_bootstrap_replicas': JOINT_REPLICAS,
        },
        'beta_pc': {'value': beta_pc, 'error': beta_pc_err},
        'U_star': {'value': U_star, 'error': U_star_err},
        'inv_nu': {'value': nu_result['inv_nu'], 'error': nu_result['inv_nu_err']},
        'nu': {'value': nu_result['nu'], 'error': nu_result['nu_err']},
        'gamma_nu': {'value': mag_result['gamma_nu'], 'error': mag_result['gamma_nu_err']},
        'beta_nu': {'value': mag_result['beta_nu'], 'error': mag_result['beta_nu_err']},
        'alpha': {'value': cv_result['alpha'], 'error': cv_result['alpha_err']},
        'derived': derived,
        'correlation_matrix': joint_result.summary['correlation_matrix'],
        'calibrated_covariance': joint_result.summary['calibrated_covariance'],
        'correlation_validation': {
            key: joint_result.summary[key] for key in (
                'correlation_eigenvalues', 'covariance_eigenvalues', 'condition_number',
                'positive_definite', 'psd_projection_applied', 'calibration_checks',
                'requested', 'valid', 'raw_mean', 'raw_std',
            )
        },
        'hyperscaling_variance_decomposition': joint_result.summary['hyperscaling_variance_decomposition'],
        'fit_quality': fit_quality,
        'F_tests': F_tests,
        'crossings': [
            {'L1': c[0], 'L2': c[1], 'beta': c[2], 'error': c[3],
             'U_star': c[4], 'U_star_error': c[5]} for c in crossings
        ],
        'cache_manifest_reference': str(cache_manifest_path.relative_to(BASE_DIR)),
        'cache_status': cache_status,
        'seeds': {
            'crossings': [42 + index for index in range(len(L_LIST) - 1)],
            'slope_covariance': 42, 'magnetic_bootstrap': 123,
            'specific_heat_bootstrap': 456, 'joint': JOINT_SEED,
        },
        'software_versions': {
            'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'matplotlib': plt.matplotlib.__version__,
        },
        'expected_plots': sorted(EXPECTED_PLOTS),
    }
    payload = _json_ready(payload)
    json_path = ANALYSIS_DIR / 'fss_results_complete.json'
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    save_joint_replicas(ANALYSIS_DIR / 'fss_joint_replicas.npz', joint_result)

    dat_path = ANALYSIS_DIR / 'fss_results_complete.dat'
    with dat_path.open('w') as stream:
        stream.write('FINITE-SIZE SCALING ANALYSIS - OFFICIAL RESULTS\n')
        stream.write('Observable convention: reduced (no beta or beta^2 factors)\n\n')
        stream.write('[A. OFFICIAL FSS MARGINALS]\n')
        stream.write(f'beta_pc = {beta_pc:.11f} +/- {beta_pc_err:.11f}\n')
        stream.write(f'U_star = {U_star:.9f} +/- {U_star_err:.9f}\n')
        stream.write(f'1/nu = {nu_result["inv_nu"]:.9f} +/- {nu_result["inv_nu_err"]:.9f}\n')
        stream.write(f'nu = 1/(1/nu) = {nu_result["nu"]:.9f} +/- {nu_result["nu_err"]:.9f}\n')
        stream.write(f'gamma/nu = {mag_result["gamma_nu"]:.9f} +/- {mag_result["gamma_nu_err"]:.9f}\n')
        stream.write(f'beta/nu = {mag_result["beta_nu"]:.9f} +/- {mag_result["beta_nu_err"]:.9f}\n')
        stream.write(f'alpha = {cv_result["alpha"]:.9f} +/- {cv_result["alpha_err"]:.9f}\n\n')
        stream.write('[FIT QUALITY]\n')
        for name, quality in fit_quality.items():
            stream.write(f'{name}: chi2={quality["chi2"]:.9f}, dof={quality["dof"]}, '
                         f'chi2_red={quality["chi2_red"]:.9f}, p={quality["p_value"]:.9f}\n')
        stream.write('\n[B. SUBLEADING L^-2 DIAGNOSTICS]\n')
        for name, test in F_tests.items():
            stream.write(f'{name}: F={test["F"]:.9f}, p={test["p_value"]:.9f}, '
                         f'significant={test["significant"]}\n')
        stream.write('\n[C. BOOTSTRAP CORRELATION MATRIX: nu, gamma/nu, beta/nu]\n')
        stream.write(np.array2string(np.asarray(payload['correlation_matrix']), precision=9) + '\n')
        stream.write('\n[D. JOINT DERIVED QUANTITIES]\n')
        for name in ('gamma', 'beta', 'eta', 'delta', 'd_hyper'):
            stream.write(f'{name} = {derived[name]["central"]:.9f} +/- {derived[name]["error"]:.9f}\n')
        hyper = payload['hyperscaling_variance_decomposition']
        stream.write('\n[HYPERSCALING VARIANCE DECOMPOSITION]\n')
        for key, value in hyper.items():
            stream.write(f'{key} = {value:.16g}\n')
        stream.write(f'\ncache_manifest = {payload["cache_manifest_reference"]}\n')
        stream.write(f'joint_replicas = {joint_result.summary["valid"]}/{JOINT_REPLICAS}\n')
    print(f'  \u2713 Structured results: {dat_path}')
    print(f'  \u2713 JSON results: {json_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-joint-cache', action='store_true',
                        help='explicitly rebuild only invalid joint-cache runs from raw binaries')
    args = parser.parse_args()

    sns.set_theme(style="ticks", font_scale=1.0)
    plt.rcParams.update({
        'mathtext.fontset': 'cm', 'font.family': 'serif', 'axes.linewidth': 1.2,
        'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    })
    plt.ioff()
    data = load_data(L_LIST)
    if sorted(data) != L_LIST:
        raise RuntimeError(f'missing parsed production data: loaded sizes {sorted(data)}')

    crossings, splines, beta_pc, beta_pc_err, U_star, U_star_err = find_beta_pc(data, L_LIST)
    if len(crossings) != len(L_LIST) - 1 or not np.isfinite(beta_pc):
        raise RuntimeError('primary Binder crossing analysis failed')
    plot_beta_pc_vs_L(data, L_LIST, crossings)

    nu_result = extract_nu_official(data, beta_pc)
    mag_result = extract_magnetic_exponents(
        data, L_LIST, beta_pc, beta_pc_err, nu_result['nu'], nu_result['nu_err']
    )
    cv_result = extract_alpha(data, L_LIST, nu_result['nu'], nu_result['nu_err'])
    if not cv_result.get('gen_fit_ok'):
        raise RuntimeError('primary alpha fit failed')

    cache_dir, cache_manifest_path, cache_manifest = ensure_official_cache(
        BASE_DIR, rebuild=args.rebuild_joint_cache
    )
    crossing_errors = np.asarray([crossing[3] for crossing in crossings])
    centers = np.array([nu_result['nu'], mag_result['gamma_nu'], mag_result['beta_nu']])
    sigmas = np.array([nu_result['nu_err'], mag_result['gamma_nu_err'], mag_result['beta_nu_err']])
    joint_result = run_joint_bootstrap(
        cache_dir, cache_manifest, data, nu_result['covariance'], crossing_errors,
        mag_result['chi_max_err'], mag_result['mag_pc_err'], centers, sigmas,
    )
    for name, values in joint_result.summary['derived'].items():
        key = 'hyperscaling' if name == 'd_hyper' else name
        mag_result[key] = values['central']
        mag_result[f'{key}_err'] = values['error']

    report_joint_derived_quantities(joint_result)

    data_collapse_plots(data, L_LIST, beta_pc, nu_result['nu'],
                        mag_result['gamma_nu'], mag_result['beta_nu'])
    plot_chi_parabolic_example(data, L_LIST, mag_result, L_example=96)
    plot_C_parabolic_example(data, cv_result, L_example=96)
    plot_specific_heat_scaling_fit(cv_result, nu_result['nu'], nu_result['nu_err'])
    plot_magnetization_vs_L(mag_result)
    half_widths = [0.000175, 0.075, 0.03]
    centers_plot = [beta_pc, nu_result['inv_nu'], mag_result['gamma_nu']]
    ylims = [(center - width, center + width) for center, width in zip(centers_plot, half_widths)]
    plot_robustness_official(crossings, nu_result, mag_result,
                             beta_pc, beta_pc_err, ylims=ylims)

    actual_plots = {path.name for path in PLOT_DIR.iterdir() if path.is_file()}
    if actual_plots != EXPECTED_PLOTS:
        raise RuntimeError(
            f'plot inventory mismatch; missing={sorted(EXPECTED_PLOTS - actual_plots)}, '
            f'extra={sorted(actual_plots - EXPECTED_PLOTS)}'
        )
    save_official_outputs(
        beta_pc, beta_pc_err, U_star, U_star_err, crossings,
        nu_result, mag_result, cv_result, joint_result,
        cache_manifest_path, cache_manifest['cache_status'],
    )
    print('\nANALYSIS COMPLETE')
    print(f'Results: {ANALYSIS_DIR / "fss_results_complete.dat"}')
    print(f'JSON: {ANALYSIS_DIR / "fss_results_complete.json"}')
    print(f'Joint replicas: {ANALYSIS_DIR / "fss_joint_replicas.npz"}')
    print(f'Plots: {PLOT_DIR} (exactly 14 PDFs)')


if __name__ == "__main__":
    main()