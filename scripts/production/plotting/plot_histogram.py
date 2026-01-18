#!/usr/bin/env python3
"""
Magnetisation probability-distribution histograms P(m) for the 2D Ising model.

Produces two PDF plots:
  1. Disordered phase (beta ~ 0.35): single Gaussian peak at m = 0.
  2. Ordered phase (beta ~ 0.50): bimodal distribution with broken-axis view.
Output: results/production/plots/histograms/.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import struct
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "production" / "plots" / "histograms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BETA_DISORDERED_TARGET = 0.35           # well above T_c
BETA_ORDERED_TARGET = 0.50              # well below T_c
BETA_TOLERANCE = 0.005                  # file-matching tolerance

L_TARGETS = [64, 96]

# Binary file layout (48-byte header, then packed records)
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

sns.set_theme(style="ticks", context="paper", font_scale=1.6)

PLOT_COLORS = {64: plt.cm.viridis(0.4), 96: plt.cm.viridis(0.8)}
KDE_COLORS = {64: 'darkblue', 96: 'darkgreen'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_all_matching_files(L, target_beta):
    """Return sorted list of binary files within BETA_TOLERANCE of *target_beta*."""
    search_dirs = [
        PROJECT_ROOT / "results" / "production" / f"L{L}" / "bin",
        PROJECT_ROOT / "results" / "pilot" / f"L{L}" / "bin"
    ]
    matching_files = []
    regex = re.compile(r"beta(\d+\.\d+)")

    for d in search_dirs:
        if not d.exists(): continue
        for f in d.glob("*.bin"):
            match = regex.search(f.name)
            if match:
                b = float(match.group(1))
                if abs(b - target_beta) < BETA_TOLERANCE:
                    matching_files.append(f)
    return sorted(matching_files)

def load_magnetization_aggregate(files):
    """Load and concatenate m time-series from *files*, discarding 10 % burn-in."""
    all_data = []
    for fpath in files:
        try:
            data = np.fromfile(fpath, dtype=RECORD_DTYPE, offset=HEADER_SIZE)
            cut = int(len(data) * 0.1)
            if len(data) > cut:
                all_data.append(data["m"][cut:])
        except Exception as e:
            print(f"[ERR] {fpath.name}: {e}")
    if not all_data: return None
    return np.concatenate(all_data)

def symmetrize_data(m_data):
    """Enforce Z2 symmetry by mirroring m -> -m."""
    return np.concatenate([m_data, -m_data])

def setup_plot(xlabel, title):
    """Single-axes figure with standard styling."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(r"$P(m)$", fontsize=20)
    ax.set_title(title, fontsize=20, pad=15)
    ax.grid(True, linestyle=':', alpha=0.5)
    return fig, ax

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_disordered_phase():
    """P(m) in the disordered phase: single Gaussian peak at m = 0."""
    print(f"\n-> Disordered Phase Histogram...")
    fig, ax = setup_plot(r"Magnetization $m$",
                         fr"Disordered Phase ($\beta \approx {BETA_DISORDERED_TARGET}$)")

    for L in L_TARGETS:
        files = find_all_matching_files(L, BETA_DISORDERED_TARGET)
        if not files: continue
        m_data = load_magnetization_aggregate(files)
        if m_data is None: continue
        m_data = symmetrize_data(m_data)

        sns.histplot(m_data, stat="density", element="step", fill=True,
                     ax=ax, color=PLOT_COLORS[L], alpha=0.4, label=f"L={L}",
                     bins=65, linewidth=1.5)
        sns.kdeplot(m_data, ax=ax, color=KDE_COLORS[L], linestyle='--',
                    linewidth=1.5, bw_adjust=0.5, gridsize=2000)

    ax.legend(frameon=False, fontsize=14)
    plt.savefig(OUTPUT_DIR / "hist_disordered_gaussian.pdf")
    print(f"[OK] Saved: hist_disordered_gaussian.pdf")
    plt.close(fig)

def plot_ordered_phase_broken_axis():
    """P(m) in the ordered phase: bimodal peaks shown with a broken x-axis."""
    print(f"\n-> Ordered Phase Histogram (Split View)...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
    fig.subplots_adjust(wspace=0.15)
    fig.suptitle(fr"Ordered Phase ($\beta \approx {BETA_ORDERED_TARGET}$)",
                 fontsize=20, y=0.95)

    for L in L_TARGETS:
        files = find_all_matching_files(L, BETA_ORDERED_TARGET)
        if not files: continue
        m_data = load_magnetization_aggregate(files)
        if m_data is None: continue
        m_data = symmetrize_data(m_data)

        # Left panel (negative-m peak)
        sns.histplot(m_data, stat="density", element="step", fill=True,
                     ax=ax1, color=PLOT_COLORS[L], alpha=0.4,
                     bins=500, linewidth=1.5)
        sns.kdeplot(m_data, ax=ax1, color=KDE_COLORS[L], linestyle='--',
                    linewidth=1.5, bw_adjust=0.05, gridsize=2000)

        # Right panel (positive-m peak)
        sns.histplot(m_data, stat="density", element="step", fill=True,
                     ax=ax2, color=PLOT_COLORS[L], alpha=0.4, label=f"L={L}",
                     bins=500, linewidth=1.5)
        sns.kdeplot(m_data, ax=ax2, color=KDE_COLORS[L], linestyle='--',
                    linewidth=1.5, bw_adjust=0.05, gridsize=2000)

    # -- Broken-axis cosmetics --
    ax1.set_xlim(-1.0, -0.8)
    ax2.set_xlim(0.8, 1.0)

    ax1.set_xticks([-1.0, -0.95, -0.9, -0.85, -0.8])
    ax1.set_xticklabels(["-1.0", "-0.95", "-0.9", "-0.85", "-0.8"])
    ax2.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax2.set_xticklabels(["0.8", "0.85", "0.9", "0.95", "1.0"])

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.set_visible(False)
    ax2.tick_params(left=False)

    # Diagonal break marks
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    ax1.set_ylabel(r"$P(m)$", fontsize=20)
    fig.text(0.5, 0.02, r"Magnetization $m$", ha='center', fontsize=20)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(frameon=False, fontsize=14, loc='upper right')

    plt.savefig(OUTPUT_DIR / "hist_ordered_bimodal.pdf", bbox_inches='tight')
    print(f"[OK] Saved: hist_ordered_bimodal.pdf")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plot_disordered_phase()
    plot_ordered_phase_broken_axis()

if __name__ == "__main__":
    main()
