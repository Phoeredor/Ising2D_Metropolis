#!/usr/bin/env python3
"""
Publication-quality plots of thermodynamic observables for the 2D Ising model.

One PDF per observable: <|m|> (with Yang exact solution), chi / chi',
C_v, U4 (with crossing-point inset), <e>, signed <m> (zoomed near beta_c),
and U4 vs L at fixed beta. Input: parsed JSON; output: PDF.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "production" / "plots" / "observables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BETA_C_EXACT = 0.44068679350977151262          # exact Onsager critical point

# Universal Binder-cumulant value at beta_c: U4* = 1 - R/3,
# R = 1.1679227(4) (Salas & Sokal, 2000)
U4_STAR = 1.0 - (1.1679227 / 3.0)

BETA_ZOOM_MIN = 0.42                           # signed-m zoom window
BETA_ZOOM_MAX = 0.46
BETA_FIXED_TARGET = 0.42                       # for U4-vs-L (disordered phase)

COMMON_MARKER_SIZE = 4
FONT_SCALE = 1.6

# Per-observable L exclusions (e.g. noisy small lattices)
PLOT_EXCLUSIONS = {
    "magnetization_abs": [],
    "susceptibility":    [],
    "specific_heat":     [],
    "binder":            [10],
    "energy":            [],
    "mag_zoom":          [],
    "binder_vs_L":       []
}

# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------
def yang_magnetization(beta):
    """Yang (1952) exact magnetisation: M = [1 - sinh(2*beta)^{-4}]^{1/8}."""
    m_theory = np.zeros_like(beta)
    mask = beta > BETA_C_EXACT
    b_ord = beta[mask]

    sinh_term = np.sinh(2 * b_ord)
    term = sinh_term**(-4)
    valid = term < 1.0

    m_theory[mask] = np.where(valid, (1.0 - term[valid])**(1/8), 0.0)
    return m_theory

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(L):
    """Load parsed JSON observables for lattice size *L* (beta in [0.30, 0.60])."""
    json_path = PROJECT_ROOT / "results" / "production" / f"L{L}" / "parsed" / f"observables_L{L}.json"
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    data = sorted([d for d in raw_data if 0.30 <= d["beta"] <= 0.60], key=lambda x: x["beta"])
    return data

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def setup_plot(xlabel, ylabel):
    """Single-axes figure with standard styling (no title)."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-4, 4))
    ax.grid(True, linestyle=':', alpha=0.5)
    return fig, ax

def save_plot(fig, filename):
    """Save figure as PDF and close."""
    path = OUTPUT_DIR / filename
    plt.savefig(path)
    print(f"[OK] Saved: {path}")
    plt.close(fig)

def get_color_dict(L_list):
    """Map lattice sizes to viridis colours."""
    if len(L_list) > 1:
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))
    else:
        colors = [plt.cm.viridis(0.5)]
    return {L: c for L, c in zip(L_list, colors)}

# ---------------------------------------------------------------------------
# Individual observable plots
# ---------------------------------------------------------------------------
def plot_abs_magnetization(L_list, colors):
    """<|m|>(beta) with the Yang exact solution overlaid."""
    fig, ax = setup_plot(r"$\beta$", r"$\langle |m| \rangle$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')

    beta_range = np.linspace(0.30, 0.60, 2000)
    m_yang = yang_magnetization(beta_range)
    ax.plot(beta_range, m_yang, color='purple', linewidth=2, linestyle='-', alpha=0.8, label='Theory')

    for L in L_list:
        if L in PLOT_EXCLUSIONS["magnetization_abs"]: continue
        data = load_data(L)
        if not data: continue
        beta = [d["beta"] for d in data]
        val  = [d["M_abs"] for d in data]
        err  = [d["M_abs_err"] for d in data]
        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE, 
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)

    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "magnetization_abs.pdf")

def plot_susceptibility_variants(L_list, colors):
    """Three susceptibility variants: chi', chi/10^3 (standard), chi'/10^2."""

    # -- Reduced susceptibility chi' --
    fig, ax = setup_plot(r"$\beta$", r"$\chi'$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')
    for L in L_list:
        if L in PLOT_EXCLUSIONS["susceptibility"]: continue
        data = load_data(L)
        if not data: continue
        beta = [d["beta"] for d in data]
        val  = [d["chi"] for d in data]
        err  = [d["chi_err"] for d in data]
        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE, 
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "susceptibility.pdf")

    # -- Standard chi = beta * L^2 * <m^2>, scaled by 10^3 --
    fig, ax = setup_plot(r"$\beta$", r"$\chi / 10^3$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')
    for L in L_list:
        if L in PLOT_EXCLUSIONS["susceptibility"]: continue
        data = load_data(L)
        if not data or "M2_mean" not in data[0]: continue

        beta = np.array([d["beta"] for d in data])
        m2 = np.array([d["M2_mean"] for d in data])
        m2_err = np.array([d["M2_err"] for d in data])

        val = (beta * (L**2) * m2) / 1000.0
        err = (beta * (L**2) * m2_err) / 1000.0

        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE,
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "susceptibility_standard_scaled.pdf")

    # -- Reduced chi' scaled by 10^2 --
    fig, ax = setup_plot(r"$\beta$", r"$\chi' / 10^2$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')
    for L in L_list:
        if L in PLOT_EXCLUSIONS["susceptibility"]: continue
        data = load_data(L)
        if not data: continue
        beta = [d["beta"] for d in data]
        val  = np.array([d["chi"] for d in data]) / 100.0
        err  = np.array([d["chi_err"] for d in data]) / 100.0
        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE, 
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "susceptibility_reduced_scaled.pdf")

def plot_specific_heat(L_list, colors):
    """C_v(beta) for all lattice sizes."""
    fig, ax = setup_plot(r"$\beta$", r"$C_v$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')
    for L in L_list:
        if L in PLOT_EXCLUSIONS["specific_heat"]: continue
        data = load_data(L)
        if not data: continue
        beta = [d["beta"] for d in data]
        val  = [d["C"] for d in data]
        err  = [d["C_err"] for d in data]
        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE, 
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "specific_heat.pdf")

def plot_binder(L_list, colors):
    """U4(beta) with inset zoomed on the crossing region near beta_c."""
    fig, ax = setup_plot(r"$\beta$", r"$U_4$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')

    ax_ins = inset_axes(ax, width="35%", height="35%", loc='lower right',
                        bbox_to_anchor=(0.015, 0.0, 1, 1),
                        bbox_transform=ax.transAxes, borderpad=2)

    for L in L_list:
        if L in PLOT_EXCLUSIONS["binder"]: continue
        data = load_data(L)
        if not data: continue
        beta = np.array([d["beta"] for d in data])
        val  = np.array([d["U4"] for d in data])
        err  = np.array([d["U4_err"] for d in data])

        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE,
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
        # Inset: points connected by lines
        ax_ins.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE,
                        color=colors[L], linestyle='-', linewidth=1.2, capsize=0)

    ax.set_ylim(-0.05, 0.70)
    ax.legend(frameon=False, fontsize=12, loc='upper left')

    # Inset: zoom on crossing region with U4* reference line
    ax_ins.set_xlim(BETA_C_EXACT - 0.005, BETA_C_EXACT + 0.006)
    ax_ins.set_ylim(0.586, 0.624)
    ax_ins.grid(True, linestyle=':', alpha=0.4)
    ax_ins.ticklabel_format(style='sci', axis='both', scilimits=(-4, 4))
    ax_ins.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax_ins.axhline(U4_STAR, color='red', linestyle='-', linewidth=1.5, label=r"$U^*$", zorder=100)

    mark_inset(ax, ax_ins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--", lw=0.5)
    save_plot(fig, "binder_cumulant.pdf")

def plot_energy_density(L_list, colors):
    """Energy density <e>(beta)."""
    fig, ax = setup_plot(r"$\beta$", r"$\langle e \rangle$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')

    for L in L_list:
        if L in PLOT_EXCLUSIONS["energy"]: continue
        data = load_data(L)
        if not data: continue
        if "E_mean" not in data[0]: continue

        beta = np.array([d["beta"] for d in data])
        val  = np.array([d["E_mean"] for d in data])
        err  = np.array([d["E_err"] for d in data])
        
        ax.errorbar(beta, val, yerr=err, fmt='o', markersize=COMMON_MARKER_SIZE, 
                    color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    
    ax.legend(frameon=False, fontsize=12, loc='lower left')
    save_plot(fig, "energy_density.pdf")

def plot_signed_magnetization_zoom(L_list, colors):
    """Signed <m>(beta) x 10^5, zoomed near beta_c."""
    fig, ax = setup_plot(r"$\beta$", r"$\langle m \rangle \times 10^5$")
    ax.axvline(BETA_C_EXACT, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=r'$\beta_c$ exact')
    ax.ticklabel_format(style='plain', axis='y')  # avoid extra sci offset

    target_Ls = [L for L in L_list if L in [32, 64, 128]]
    if len(target_Ls) < 3: target_Ls = L_list
    
    for L in target_Ls:
        if L in PLOT_EXCLUSIONS["mag_zoom"]: continue
        d = load_data(L)
        if not d: continue
        zoom = [x for x in d if BETA_ZOOM_MIN <= x["beta"] <= BETA_ZOOM_MAX]
        if not zoom: continue
        
        ax.errorbar([x["beta"] for x in zoom], np.array([x["M_val"] for x in zoom])*1e5, 
                    yerr=np.array([x["M_err"] for x in zoom])*1e5,
                    fmt='o', markersize=COMMON_MARKER_SIZE, color=colors[L], label=f"L={L}", linestyle='none', capsize=0)
    
    ax.set_xlim(0.435, 0.455)
    ax.axhline(0, color='gray', lw=1, linestyle='-', alpha=0.5)
    ax.legend(frameon=False, fontsize=12)
    save_plot(fig, "magnetization_signed_zoom.pdf")

def plot_binder_vs_L(L_list, colors):
    """U4 vs L at fixed beta (disordered phase) — thermodynamic-limit check."""
    fig, ax = setup_plot(r"$L$", r"$U_4$")

    l_vals = []
    u4_vals = []
    u4_errs = []
    found_L = []
    actual_beta = 0.0

    for L in sorted(L_list):
        if L in PLOT_EXCLUSIONS["binder_vs_L"]: continue
        data = load_data(L)
        if not data: continue
        
        closest_pt = min(data, key=lambda x: abs(x["beta"] - BETA_FIXED_TARGET))
        if abs(closest_pt["beta"] - BETA_FIXED_TARGET) > 0.01:
            print(f"[WARN] No data near beta={BETA_FIXED_TARGET} for L={L}")
            continue

        l_vals.append(L)
        u4_vals.append(closest_pt["U4"])
        u4_errs.append(closest_pt["U4_err"])
        found_L.append(L)
        actual_beta = closest_pt["beta"]

    if not l_vals:
        print("[WARN] No data found for Binder vs L plot.")
        plt.close(fig)
        return

    for i, L in enumerate(found_L):
        ax.errorbar(l_vals[i], u4_vals[i], yerr=u4_errs[i],
                    fmt='o', markersize=6, color=colors[L],
                    capsize=0, elinewidth=1.5, label=f"L={L}")

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label=r"Limit $L \to \infty$")
    ax.set_title(rf"$\beta \sim {actual_beta:.2f}$", fontsize=22, pad=15)
    ax.set_xlim(0, max(l_vals)*1.1)
    ax.set_ylim(bottom=-0.02)
    ax.legend(frameon=False, fontsize=12, loc='upper right')

    save_plot(fig, "binder_vs_L.pdf")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_observables.py L1 [L2 L3 ...]")
        sys.exit(1)

    L_list = sorted([int(x) for x in sys.argv[1:]])
    
    sns.set_theme(style="ticks", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "axes.spines.right": True, "axes.spines.top": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "mathtext.fontset": "cm", "font.family": "serif",
    })

    colors = get_color_dict(L_list)

    plot_abs_magnetization(L_list, colors)
    plot_susceptibility_variants(L_list, colors)
    plot_specific_heat(L_list, colors)
    plot_binder(L_list, colors)
    plot_energy_density(L_list, colors)
    plot_signed_magnetization_zoom(L_list, colors)
    plot_binder_vs_L(L_list, colors)

if __name__ == "__main__":
    main()
