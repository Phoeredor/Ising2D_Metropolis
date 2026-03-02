#!/usr/bin/env python3
"""
Dynamic critical exponent z from tau_int(m) ~ L^z.

For each L, reads the binary run closest to beta_c, computes tau_int
of the magnetisation via FFT + Madras-Sokal windowing (c=6), then fits
the power law tau = A * L^z with weighted least squares.

Outputs:
  analysis/tau_critical_table.tex    – LaTeX table of tau per L
  analysis/dynamic_exponent_results.txt – z, A, chi2_red
  plots/dynamic_scaling_z.pdf        – log-log fit plot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import seaborn as sns
import re
import os
import gc
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR     = Path(__file__).resolve().parents[3]
PLOT_DIR     = BASE_DIR / "results" / "production" / "plots"
ANALYSIS_DIR = BASE_DIR / "results" / "production" / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

L_LIST    = [10, 16, 24, 32, 48, 64, 96, 128]
BETA_CRIT = 0.44068679350977151262   # exact Onsager value
MAX_LAG   = 400_000                  # max lag for ACF truncation
C_WINDOW  = 6.0                      # Madras-Sokal window constant
SAFE_N_LOAD = 15_000_000             # cap on records loaded per file

# Binary layout: 48-byte header, then packed records of 6 doubles + 1 int64
HEADER_SIZE  = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

# ============================================================================
# DATA I/O
# ============================================================================
def find_critical_file(L):
    """Return (path, beta) for the binary file closest to beta_c.

    Searches production first, then pilot directories.
    """
    search_dirs = [
        BASE_DIR / "results" / "production" / f"L{L}" / "bin",
        BASE_DIR / "results" / "pilot" / f"L{L}" / "bin"
    ]
    best_file, min_diff, best_val = None, float('inf'), None
    regex = re.compile(r"beta(\d+\.\d+)")

    for d in search_dirs:
        if not d.exists(): continue
        for f in d.glob(f"*obs_L{L}*.bin"):
            match = regex.search(f.name)
            if match:
                b = float(match.group(1))
                diff = abs(b - BETA_CRIT)
                if diff < min_diff:
                    min_diff, best_file, best_val = diff, f, b
    return best_file, best_val


def read_magnetization_safe(fpath):
    """Load the m column from a binary file, capped at SAFE_N_LOAD records."""
    try:
        fsize = os.path.getsize(fpath)
        data_size = fsize - HEADER_SIZE
        if data_size <= 0: return np.array([])
        n_records = data_size // RECORD_DTYPE.itemsize
        n_to_read = min(n_records, SAFE_N_LOAD)

        mm = np.memmap(fpath, dtype=RECORD_DTYPE, mode='r',
                       offset=HEADER_SIZE, shape=(n_records,))
        m_data = np.array(mm["m"][:n_to_read], dtype=np.float64)
        del mm
        return m_data
    except Exception as e:
        print(f"    Error reading {fpath}: {e}")
        return np.array([])

# ============================================================================
# AUTOCORRELATION ANALYSIS
# ============================================================================
def compute_tau_madras_sokal(data):
    """Integrated autocorrelation time via FFT + Madras-Sokal windowing.

    Returns (tau_int, W_opt, sigma_tau).
    The window W is the smallest lag satisfying W >= c * tau(W),
    with c = C_WINDOW.  Error follows Eq.(4.10) of Madras & Sokal (1988).
    """
    N = len(data)
    if N < 1000: return np.nan, np.nan, np.nan

    # Normalised ACF via FFT (zero-padded to next power of 2)
    data_c = data - np.mean(data)
    n_fft = 2**(int(np.log2(2*N)) + 1)
    ft = np.fft.rfft(data_c, n=n_fft)
    S = ft * np.conj(ft)
    acf = np.fft.irfft(S)
    acf = acf[:min(N, MAX_LAG)]

    if acf[0] != 0:
        acf /= acf[0]
    else:
        return 0.0, 0.0, 0.0

    # Cumulative sum gives tau(W) = 0.5 + sum_{t=1}^{W} rho(t)
    cumsum = np.cumsum(acf[1:])
    tau_prof = 0.5 + cumsum
    lags = np.arange(1, len(tau_prof) + 1)

    # Automatic truncation: first W where W >= c * tau(W)
    valid = np.where(lags >= C_WINDOW * tau_prof)[0]
    if len(valid) > 0:
        idx = valid[0]
        tau, W = tau_prof[idx], lags[idx]
    else:
        tau, W = tau_prof[-1], lags[-1]

    # Madras-Sokal error: sigma_tau = tau * sqrt(2(2W+1)/N)
    sigma = tau * np.sqrt(2.0 * (2.0 * W + 1) / N)
    return tau, W, sigma

# ============================================================================
# FITTING
# ============================================================================
def fit_power_law(L, A, z):
    """Model: tau = A * L^z."""
    return A * (L**z)


def perform_weighted_fit(L_arr, tau_arr, err_arr):
    """Weighted NLS fit of tau = A * L^z.

    Initial guess from OLS in log-log space; final fit in linear space
    with absolute_sigma=True.  Returns (A, A_err, z, z_err, chi2_red).
    """
    p_init = np.polyfit(np.log(L_arr), np.log(tau_arr), 1)
    z_guess, A_guess = p_init[0], np.exp(p_init[1])
    
    popt, pcov = spo.curve_fit(
        fit_power_law, L_arr, tau_arr,
        p0=[A_guess, z_guess],
        sigma=err_arr,
        absolute_sigma=True
    )
    A_fit, z_fit = popt
    perr      = np.sqrt(np.diag(pcov))
    A_err     = perr[0]
    z_err_fit = perr[1]
    
    residuals = tau_arr - fit_power_law(L_arr, *popt)
    chi2      = np.sum((residuals / err_arr)**2)
    dof       = len(L_arr) - 2
    chi2_red  = chi2 / dof if dof > 0 else 0
    
    return A_fit, A_err, z_fit, z_err_fit, chi2_red

# ============================================================================
# OUTPUT
# ============================================================================
def save_latex_table(data, filepath):
    """Write a booktabs LaTeX table with tau_int, sigma, W per L."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$L$ & $\tau_{int,m}$ & $\sigma_{\tau}$ & $W_{opt}$ \\",
        r"\midrule"
    ]
    
    for row in data:
        L, tau, err, W = row['L'], row['tau'], row['err'], row['W']
        lines.append(f"{L} & {tau:.1f} & {err:.1f} & {W} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Integrated autocorrelation times at $\beta_c$ with Madras-Sokal errors.}",
        r"\label{tab:tau_critical}",
        r"\end{table}"
    ])
    
    with open(filepath, "w") as f:
        f.write("\n".join(lines))

def save_fit_results(z, z_err, chi2_red, A, A_err, filepath):
    """Write z, A and goodness-of-fit summary to a plain-text file."""
    with open(filepath, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("DYNAMIC EXPONENT CALCULATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Beta_c (Onsager): {BETA_CRIT:.17f}\n")
        f.write(f"Madras-Sokal window constant: c = {C_WINDOW}\n\n")
        f.write("FIT RESULTS (tau_int ~ A * L^z):\n")
        f.write("-" * 60 + "\n")
        f.write(f"z          = {z:.3f} +/- {z_err:.3f}\n")
        f.write(f"A          = {A:.2f} +/- {A_err:.2f}  [MCS]\n")
        f.write(f"chi2_red   = {chi2_red:.2f}\n")
        f.write("\nINTERPRETATION:\n")
        f.write(f"  Theoretical value: z ~ 2.17 (Metropolis, 2D Ising)\n")
        agreement = 'EXCELLENT' if abs(z - 2.17) < 2*z_err else 'GOOD'
        f.write(f"  Agreement: {agreement}\n")
        validity  = 'CONFIRMED' if 0.3 < chi2_red < 3 else 'CHECK DATA'
        f.write(f"  Model validity: {validity}\n")
        f.write("=" * 60 + "\n")

def plot_dynamic_scaling(L_arr, tau_arr, err_arr, A_fit, z_fit, z_err,
                         chi2_red, filepath):
    """Log-log plot of tau_int vs L with the power-law fit overlaid."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    
    color_dict = {L: plt.cm.viridis(c) 
                  for L, c in zip(L_LIST, np.linspace(0, 0.9, len(L_LIST)))}
    
    for i, L_val in enumerate(L_arr):
        lbl = 'Data' if i == 0 else None
        ax.errorbar(L_val, tau_arr[i], yerr=err_arr[i], fmt='o',
                   color=color_dict[L_val], ms=7, capsize=0, label=lbl)
    
    l_range = np.linspace(min(L_arr)*0.9, max(L_arr)*1.1, 100)
    ax.plot(l_range, fit_power_law(l_range, A_fit, z_fit), 'k--',
           label=fr"Fit: $z={z_fit:.2f} \pm {z_err:.2f}$")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(L_arr)
    ax.set_xticklabels(L_arr)
    ax.minorticks_off()
    
    ax.set_xlabel(r"Lattice Size $L$", fontsize=16)
    ax.set_ylabel(r"$\tau_{int,m}$ [MCS]", fontsize=16)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    
    plt.savefig(filepath)

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("DYNAMIC EXPONENT CALCULATION (Madras-Sokal c={})".format(C_WINDOW))
    print("=" * 60)
    
    table_data = []
    valid_L, valid_tau, valid_err = [], [], []

    print(f"\n{'L':<4} | {'Beta':<8} | {'Tau':<10} | {'Err':<8} | {'Window':<8}")
    print("-" * 55)
    
    for L in L_LIST:
        gc.collect()
        fpath, beta_val = find_critical_file(L)
        if not fpath:
            print(f"{L:<4} | NOT FOUND")
            continue
        
        m_data = read_magnetization_safe(fpath)
        if len(m_data) == 0: continue

        # Discard first 10 % as thermalisation burn-in
        cut = int(len(m_data)*0.1)
        m_data = m_data[cut:]
        n_meas = len(m_data)
        
        tau, window, err = compute_tau_madras_sokal(m_data)
        del m_data
        
        print(f"{L:<4} | {beta_val:.5f} | {tau:<10.1f} | {err:<8.1f} | {window:<8}")
        
        if not np.isnan(tau) and tau > 0:
            table_data.append({'L': L, 'tau': tau, 'err': err, 'W': window})
            valid_L.append(L)
            valid_tau.append(tau)
            valid_err.append(err)

    save_latex_table(table_data, ANALYSIS_DIR / "tau_critical_table.tex")
    print(f"\n✓ Table saved: {ANALYSIS_DIR / 'tau_critical_table.tex'}")

    # Power-law fit tau = A * L^z (need >= 3 points)
    if len(valid_L) > 2:
        L_arr = np.array(valid_L)
        tau_arr = np.array(valid_tau)
        err_arr = np.array(valid_err)
        
        A_fit, A_err, z_fit, z_err, chi2_red = perform_weighted_fit(
            L_arr, tau_arr, err_arr
        )
        
        print("\n" + "=" * 60)
        print("FIT RESULTS:")
        print(f"  z = {z_fit:.3f} ± {z_err:.3f}")
        print(f"  A = {A_fit:.2f} ± {A_err:.2f}  [MCS]")
        print(f"  χ²_red = {chi2_red:.2f}")
        print("=" * 60)
        
        # Save numerical results
        save_fit_results(z_fit, z_err, chi2_red, A_fit, A_err,
                         ANALYSIS_DIR / "dynamic_exponent_results.txt")
        print(f"✓ Results saved: {ANALYSIS_DIR / 'dynamic_exponent_results.txt'}")
        
        # Generate plot
        plot_dynamic_scaling(L_arr, tau_arr, err_arr, A_fit, z_fit, z_err,
                             chi2_red, PLOT_DIR / "dynamic_scaling_z.pdf")
        print(f"✓ Plot saved: {PLOT_DIR / 'dynamic_scaling_z.pdf'}\n")
    else:
        print("\n✗ Insufficient data for fit.\n")

if __name__ == "__main__":
    main()

