#!/usr/bin/env python3
"""
Data-blocking analysis of statistical errors (Marinari, Sec. 20.3.1).

For a chosen lattice size at beta_c, produces two diagnostic plots:
  1. sigma_mean vs block size k (log-log) — error saturation plateau.
  2. <|m|> vs k — value stability with growing error bars.
Output: PDF files in results/production/plots/blocking/.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "production" / "plots" / "blocking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_L = 64                               # large enough to show correlations
BETA_CRIT = 0.44068679350977151262           # exact Onsager critical point

# Binary file layout (48-byte header, then packed records)
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

FONT_SCALE = 1.6
COMMON_MARKER_SIZE = 4

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def find_critical_file(L):
    """Return (path, beta) of the binary file closest to beta_c."""
    bin_dir = PROJECT_ROOT / "results" / "production" / f"L{L}" / "bin"
    if not bin_dir.exists():  # fall back to pilot data
        bin_dir = PROJECT_ROOT / "results" / "pilot" / f"L{L}" / "bin"
    if not bin_dir.exists(): return None, None

    best_file = None
    min_diff = float('inf')
    best_beta = 0.0

    import re
    regex = re.compile(r"beta(\d+\.\d+)")

    for f in bin_dir.glob("*.bin"):
        match = regex.search(f.name)
        if match:
            beta = float(match.group(1))
            diff = abs(beta - BETA_CRIT)
            if diff < min_diff:
                min_diff = diff
                best_file = f
                best_beta = beta

    return best_file, best_beta

def load_timeseries(fpath):
    """Read |m| and e time-series from a binary production file."""
    try:
        data = np.fromfile(fpath, dtype=RECORD_DTYPE, offset=HEADER_SIZE)
        return np.abs(data["m"]), data["e"]
    except Exception as e:
        print(f"[ERR] Could not read {fpath}: {e}")
        return None, None

def blocking_analysis(data):
    """Blocking transform: return (k_values, block_means, sigma_mean).

    Block sizes are log-spaced from 1 to N/50 (>= 50 blocks per point).
    """
    N = len(data)
    max_k = int(N / 50)
    k_list = np.unique(np.logspace(0, np.log10(max_k), 40).astype(int))

    means = []
    errors = []

    for k in k_list:
        n_blocks = N // k
        truncated = data[:n_blocks * k]
        blocks = truncated.reshape(n_blocks, k).mean(axis=1)

        mu = np.mean(blocks)
        sigma_mean = np.std(blocks, ddof=1) / np.sqrt(n_blocks)

        means.append(mu)
        errors.append(sigma_mean)

    return k_list, np.array(means), np.array(errors)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_error_saturation(k, err_m, err_e, L, beta):
    """Log-log plot of sigma_mean vs block size k (error saturation)."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    c_mag = plt.cm.viridis(0.3)
    c_ene = plt.cm.viridis(0.8)

    # Left axis: magnetisation error
    ln1 = ax.plot(k, err_m, 'o-', color=c_mag, markersize=COMMON_MARKER_SIZE,
                  label=r'$\sigma_{\langle m \rangle}$')
    ax.set_ylabel(r"$\sigma_{\langle m \rangle}$", fontsize=20, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_yscale('log')

    # Right axis: energy error
    ax2 = ax.twinx()
    ln2 = ax2.plot(k, err_e, 's-', color=c_ene, markersize=COMMON_MARKER_SIZE, 
                   label=r'$\sigma_{\langle e \rangle}$')
    
    ax2.set_ylabel(r"$\sigma_{\langle e \rangle}$", fontsize=20, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_yscale('log')

    ax.set_xlabel(r"$k$", fontsize=20)
    ax.set_xscale('log')
    # ax.set_title(fr"Uncertainty Estimation for $\langle e \rangle$ and "
    #              fr"$\langle m \rangle$ - $L={L}$ at $\beta_c$",
    #              fontsize=20, pad=15)

    # Combined legend for both axes
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left', frameon=False, fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.5, which='major')

    save_path = OUTPUT_DIR / "error_saturation.pdf"
    plt.savefig(save_path)
    print(f"[OK] Saved: {save_path}")
    plt.close(fig)

def plot_magnetization_stability(k, means_m, err_m, L, beta):
    """<|m|> vs block size k — value stability with growing error bars."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    
    color = plt.cm.viridis(0.5)
    
    ax.errorbar(k, means_m, yerr=err_m, fmt='o', color=color,
                markersize=COMMON_MARKER_SIZE, capsize=0, elinewidth=1.5,
                label=r'$\langle |m| \rangle_k$')
    
    ax.set_xscale('log')
    ax.set_xlabel(r"$k$", fontsize=20)
    ax.set_ylabel(r"$\langle m \rangle$", fontsize=20)
    # ax.set_title(fr"Mean Magnetization Stability - $L={L}$ at $\beta_c$", fontsize=20, pad=15)
    
    # Centre on the best estimate, zoom to 4x the largest error bar
    mean_val = means_m[-1]
    max_err = err_m[-1]
    ax.set_ylim(mean_val - 4*max_err, mean_val + 4*max_err)
    
    ax.grid(True, linestyle=':', alpha=0.5)
    
    save_path = OUTPUT_DIR / "magnetization_stability.pdf"
    plt.savefig(save_path)
    print(f"[OK] Saved: {save_path}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sns.set_theme(style="ticks", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "axes.spines.right": True, "axes.spines.top": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "cm", "font.family": "serif",
    })

    print("--- Data Blocking Analysis ---")

    fpath, beta = find_critical_file(TARGET_L)
    if not fpath:
        print(f"[ERR] No data found for L={TARGET_L}")
        return
    print(f"Analyzing: {fpath.name}")

    m_series, e_series = load_timeseries(fpath)
    if m_series is None: return

    # Discard first 10 % as thermalisation burn-in
    cut = int(len(m_series) * 0.1)
    m_series = m_series[cut:]
    e_series = e_series[cut:]
    print(f"Data points: {len(m_series)}")

    k, m_means, m_errs = blocking_analysis(m_series)
    _, _, e_errs = blocking_analysis(e_series)

    plot_error_saturation(k, m_errs, e_errs, TARGET_L, beta)
    plot_magnetization_stability(k, m_means, m_errs, TARGET_L, beta)

if __name__ == "__main__":
    main()
