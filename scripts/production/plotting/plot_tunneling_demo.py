#!/usr/bin/env python3
"""
SCRIPT: plot_tunneling_demo.py
PURPOSE: Visualize magnetization tunneling and energy fluctuations for L=16.
         Replicates Marinari's Fig 20.2 style using Production Data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy import special

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
# Search both production and pilot folders
SEARCH_DIRS = [
    BASE_DIR / "results" / "production" / "L16" / "bin",
    BASE_DIR / "results" / "pilot" / "L16" / "bin"
]
PLOT_DIR = BASE_DIR / "results" / "production" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Target
TARGET_L = 16
TARGET_BETA = 0.50  # We look for something close to T=2.0

# Binary Structure
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

# Zoom Window (Requested: 60k to 85k)
ZOOM_WINDOW = (73500, 77500) 

# =============================================================================
# PHYSICS UTILS
# =============================================================================

def onsager_magnetization(beta):
    """Exact spontaneous magnetization (Yang)."""
    beta_c = np.log(1 + np.sqrt(2)) / 2
    if beta <= beta_c: return 0.0
    return (1 - np.sinh(2 * beta)**(-4))**(1/8)

def onsager_energy(beta):
    """Exact internal energy density (Onsager)."""
    k = 2 * np.sinh(2 * beta) / (np.cosh(2 * beta)**2)
    K = special.ellipk(k**2)
    term1 = -1.0 / np.tanh(2 * beta)
    term2 = 1 + (2 / np.pi) * (2 * np.tanh(2 * beta)**2 - 1) * K
    return term1 * term2

def find_best_file():
    """Finds the binary file closest to TARGET_BETA for TARGET_L using Regex."""
    candidates = []
    print("\n--- DEBUG: SEARCHING FOR FILES ---")
    
    for d in SEARCH_DIRS:
        if d.exists():
            matched = list(d.glob(f"*obs_L{TARGET_L}*.bin"))
            candidates.extend(matched)
    
    if not candidates: 
        print("[ERR] No candidates found.")
        return None, None

    best_file = None
    min_diff = float('inf')
    
    # Robust regex: look for 'beta' followed by a float
    regex_beta = re.compile(r"beta(\d+\.\d+)")

    for f in candidates:
        try:
            match = regex_beta.search(f.name)
            if match:
                beta_val = float(match.group(1))
                diff = abs(beta_val - TARGET_BETA)
                
                if diff < min_diff:
                    min_diff = diff
                    best_file = f
        except Exception as e:
            continue
            
    return best_file, min_diff

def read_data(filepath):
    with open(filepath, "rb") as f:
        f.read(HEADER_SIZE)
        data = np.fromfile(f, dtype=RECORD_DTYPE)
    return data["sweep"], data["m"], data["e"]

# =============================================================================
# PLOTTING
# =============================================================================

def main():
    # Set style exactly as requested in plot_equilibration_dynamics.py
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    
    print(f"--- Searching for L={TARGET_L}, Beta ~ {TARGET_BETA} ---")
    fpath, diff = find_best_file()
    
    if not fpath:
        print("[ERR] File not found.")
        return
        
    print(f"File found: {fpath.name}")
    
    # Re-parse beta robustly for calculation
    regex_beta = re.compile(r"beta(\d+\.\d+)")
    match = regex_beta.search(fpath.name)
    if match:
        actual_beta = float(match.group(1))
    else:
        actual_beta = TARGET_BETA
        
    print(f"Actual Beta: {actual_beta:.6f} (Diff: {diff:.6f})")

    # Load Data
    t, m, e = read_data(fpath)
    
    # Exact Values
    m_ex = onsager_magnetization(actual_beta)
    e_ex = onsager_energy(actual_beta)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # Colors matching palette
    vir_colors = plt.cm.viridis(np.linspace(0, 0.95, 5))
    inf_colors = plt.cm.inferno(np.linspace(0.4, 0.8, 3))
    
    c_m = vir_colors[3] 
    c_e = inf_colors[1] 

    # Plot Traces
    ax.plot(t, m, color=c_m, lw=1.5, alpha=0.9, label=r"$m(t)$")
    ax.plot(t, e, color=c_e, lw=1.5, alpha=0.9, label=r"$e(t)$")

    # Analytical Lines (Onsager)
    ax.axhline(m_ex, color="gray", ls="--", lw=1, alpha=0.4)
    ax.axhline(-m_ex, color="gray", ls="--", lw=1, alpha=0.4)
    ax.axhline(e_ex, color="gray", ls="--", lw=1, alpha=0.4)
    
    # Annotations for Onsager values
    t_start = ZOOM_WINDOW[0] if ZOOM_WINDOW else t[0]
    
    # --- ADDING THE DELTA ---
    t_w0, t_w1 = ZOOM_WINDOW if ZOOM_WINDOW else (t[0], t[-1])
    mask = (t >= t_w0) & (t <= t_w1)
    
    if np.any(mask):
        e_window = e[mask]
        t_window = t[mask]
        
        # Find local maximum (peak) in this window
        idx_max = np.argmax(e_window)
        e_peak = e_window[idx_max]
        t_peak = t_window[idx_max]
        
        # Only annotate if significant peak
        if e_peak > e_ex + 0.1: 
            # Dashed line at the peak (visual guide)
            ax.hlines(e_peak, t_peak - 100, t_peak + 100, color='gray', linestyles=':', lw=1)
            
            # The Vertical Bar Annotation |-|
            # widthA and widthB control the width of the caps
            ax.annotate(
                text='', 
                xy=(t_peak + 80, e_ex ), 
                xytext=(t_peak + 80, e_peak), 
                arrowprops=dict(arrowstyle='|-|,widthA=0.2,widthB=0.2', color='black', lw=1)
            )
            
            # Text Delta
            ax.text(t_peak + 100, (e_peak + e_ex)/2, r"$\Delta$", color='black', fontsize=12, va='center', fontweight='bold')

    # Setup Axes with UNIFIED FONT SIZES
    ax.set_xlabel(r"$t$ [MC sweeps]", fontsize=18)
    ax.set_title(f"Monte Carlo Dynamics: Tunneling - $L={TARGET_L}$", fontsize=20, pad=12)
    
    ax.set_ylim(-2.1, 1.4) 
    
    # Apply Zoom
    if ZOOM_WINDOW:
        ax.set_xlim(ZOOM_WINDOW)
    else:
        ax.set_xlim(t[0], t[-1])

    # Legend (Upper Right, No Frame, FontSize 14)
    ax.legend(loc='upper right', frameon=False, fontsize=14)
    ax.grid(True, which="major", ls=":", alpha=0.4)
    
    # Thicker borders
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Save
    save_path = PLOT_DIR / f"dynamics_L{TARGET_L}_beta{actual_beta:.3f}.pdf"
    plt.savefig(save_path)
    print(f"[OK] Saved: {save_path}")

if __name__ == "__main__":
    main()
