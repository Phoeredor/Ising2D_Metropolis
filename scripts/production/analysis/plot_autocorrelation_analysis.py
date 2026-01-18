#!/usr/bin/env python3
"""
Autocorrelation analysis for 2D Ising production runs.

Computes integrated autocorrelation times (tau_int) via FFT + Madras-Sokal
automatic windowing for multiple observables (M, M^2, M^4, E) at beta_c.
Produces temperature-dependence, critical-slowing-down, tau convergence,
and multi-observable comparison plots. Output: PDF + .dat files.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
import os
from pathlib import Path
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
PLOT_DIR = BASE_DIR / "results" / "production" / "plots" / "autocorrelation"
ANALYSIS_DIR = BASE_DIR / "results" / "production" / "analysis"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

L_ALL = [24, 32, 48, 64, 96, 128]
L_TEMP_COMPARE = [24, 64]                  # sizes for temperature comparison
TARGET_TEMPS = [2.3, 2.4, 2.5]             # temperatures for T-dependence plot
BETA_CRIT = 0.44068679350977151262          # exact Onsager critical point

MAX_LAG = 300000                            # maximum lag for autocorrelation
C_WINDOW = 6.0                              # Madras-Sokal window constant
SAFE_N_LOAD = 20_000_000                    # memmap safety cap (records)

# Binary file layout (48-byte header, then packed records)
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

FONT_SCALE = 1.6
MARKER_SIZE = 6

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_color_dict(L_list):
    """Map lattice sizes to viridis colours."""
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_list)))
    return {L: c for L, c in zip(L_list, colors)}

def find_file(L, target_beta):
    """Return (path, beta) of the binary file closest to *target_beta*."""
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
                diff = abs(b - target_beta)
                if diff < min_diff:
                    min_diff, best_file, best_val = diff, f, b
    return best_file, best_val

def read_magnetization_safe(fpath):
    """Load magnetisation time-series from binary, capped at SAFE_N_LOAD."""
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
    except:
        return np.array([])

# ---------------------------------------------------------------------------
# Multi-observable loading
# ---------------------------------------------------------------------------
def read_observable_safe(fpath, obs_name, L=None):
    """Load a time-series for *obs_name* from binary (memmap, capped).

    Supported: 'm', 'm2', 'm4', 'e', 'e2', 'U4', 'chi' (needs L).
    """
    try:
        fsize = os.path.getsize(fpath)
        data_size = fsize - HEADER_SIZE
        if data_size <= 0: 
            return np.array([])
        
        n_records = data_size // RECORD_DTYPE.itemsize
        n_to_read = min(n_records, SAFE_N_LOAD)
        
        mm = np.memmap(fpath, dtype=RECORD_DTYPE, mode='r', 
                      offset=HEADER_SIZE, shape=(n_records,))
        
        # Direct per-sweep fields
        if obs_name == 'm':
            data = np.array(mm["m"][:n_to_read], dtype=np.float64)
        elif obs_name == 'm2':
            data = np.array(mm["m2"][:n_to_read], dtype=np.float64)
        elif obs_name == 'm4':
            data = np.array(mm["m4"][:n_to_read], dtype=np.float64)
        elif obs_name == 'e':
            data = np.array(mm["e"][:n_to_read], dtype=np.float64)
        elif obs_name == 'e2':
            data = np.array(mm["e2"][:n_to_read], dtype=np.float64)
        # Derived observables
        elif obs_name == 'U4':
            # Binder cumulant: U4 = 1 - <m^4> / (3 <m^2>^2)
            m2 = np.array(mm["m2"][:n_to_read], dtype=np.float64)
            m4 = np.array(mm["m4"][:n_to_read], dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                data = 1.0 - m4 / (3.0 * m2**2)
                data = np.where(np.isfinite(data), data, 0.0)
        elif obs_name == 'chi' and L is not None:
            # Susceptibility: chi = V * (<m^2> - <|m|>^2)
            m = np.array(mm["m"][:n_to_read], dtype=np.float64)
            m2 = np.array(mm["m2"][:n_to_read], dtype=np.float64)
            V = L**2
            data = V * (m2 - np.abs(m)**2)
        else:
            raise ValueError(f"Unknown observable: {obs_name}")
        
        del mm
        return data
    except Exception as e:
        print(f"[ERROR] Failed to read {obs_name} from {fpath.name}: {e}")
        return np.array([])

def compute_autocorr_fft(data, max_lag):
    """Normalised autocorrelation function via FFT (O(N log N))."""
    N = len(data)
    if N == 0:
        return np.arange(max_lag), np.zeros(max_lag)

    data_c = data - np.mean(data)
    n_fft = 2**(int(np.log2(2*N)) + 1)   # zero-pad to next power of 2
    ft = np.fft.rfft(data_c, n=n_fft)
    acf = np.fft.irfft(ft * np.conj(ft))  # Wiener-Khinchin
    acf = acf[:min(N, max_lag)]

    if acf[0] != 0:
        acf /= acf[0]
    return np.arange(len(acf)), np.real(acf)

def compute_madras_sokal(lags, C_t, c=6.0):
    """Madras-Sokal automatic windowing: W >= c * tau_int(W).

    Returns (tau_profile, W_opt, tau_int, converged).
    """
    cumsum = np.cumsum(C_t[1:])
    tau_profile = 0.5 + cumsum              # running tau_int estimate
    valid_W = np.where(lags[1:] >= c * tau_profile)[0]

    if len(valid_W) > 0:
        idx = valid_W[0]
        return tau_profile, lags[1:][idx], tau_profile[idx], True
    return tau_profile, lags[-1], tau_profile[-1], False

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def setup_plot(xlabel, ylabel):
    """Create a single-axes figure with standard styling."""
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-4, 4))
    ax.grid(True, linestyle=':', alpha=0.5)
    return fig, ax

def plot_temp_dependence():
    """C(dt) at several temperatures for L=24 vs L=64."""
    print("\n[1/4] Plotting temperature dependence...")
    fig, ax = setup_plot(r"$\Delta t$ [MCS]", r"$C(\Delta t)$")

    colors_temp = plt.cm.viridis(np.linspace(0.1, 0.9, len(TARGET_TEMPS)))
    styles = {64: "-", 24: "--"}
    found_betas = [0.0] * len(TARGET_TEMPS)

    for i, T in enumerate(TARGET_TEMPS):
        target_beta = 1.0/T
        for L in L_TEMP_COMPARE:
            gc.collect()
            fpath, actual_beta = find_file(L, target_beta)
            if not fpath: continue
            if found_betas[i] == 0.0: found_betas[i] = actual_beta

            m_raw = read_magnetization_safe(fpath)
            m_data = m_raw[int(len(m_raw)*0.1):]   # discard 10 % burn-in
            del m_raw

            lags, Cm = compute_autocorr_fft(m_data, 1800)
            cut = min(1800, len(lags))
            ax.plot(lags[:cut], Cm[:cut], color=colors_temp[i],
                   ls=styles[L], lw=2, alpha=0.9)

    legend_el = [
        Line2D([0],[0], color='k', ls='-', label=r"$L=64$"),
        Line2D([0],[0], color='k', ls='--', label=r"$L=24$"),
        Line2D([0],[0], color='w', label=' '),
        *[Line2D([0],[0], color=colors_temp[i], lw=2, 
                 label=rf"$\beta \approx {found_betas[i]:.2f}$") 
          for i in range(3)]
    ]
    ax.legend(handles=legend_el, loc='upper right', frameon=False, 
             fontsize=12, ncol=2)
    ax.set_xlim(0, 1800)
    ax.axhline(0, color='k', ls=':', alpha=0.5)
    
    plt.savefig(PLOT_DIR / "autocorr_temp_dependence.pdf")
    print(f"    Saved: {PLOT_DIR / 'autocorr_temp_dependence.pdf'}")
    plt.close()

def plot_critical_slowing_down():
    """C(dt) and tau_int convergence at beta_c for all lattice sizes."""
    print("\n[2/4] Plotting critical slowing down...")
    colors_L = get_color_dict(L_ALL)
    plot_data = []

    for L in L_ALL:
        gc.collect()
        fpath, actual_beta = find_file(L, BETA_CRIT)
        if not fpath: continue

        m_raw = read_magnetization_safe(fpath)
        m_data = m_raw[int(len(m_raw)*0.1):]   # discard 10 % burn-in
        del m_raw
        
        lags, Cm = compute_autocorr_fft(m_data, MAX_LAG)
        tau_vals, W_opt, tau_final, _ = compute_madras_sokal(lags, Cm, C_WINDOW)
        
        cut_idx = int(W_opt) + 1
        if cut_idx >= len(lags): cut_idx = len(lags) - 1

        plot_data.append({
            'L': L, 'lags': lags[:cut_idx], 'Cm': Cm[:cut_idx],
            'tau_lags': lags[1:cut_idx], 'tau_vals': tau_vals[:cut_idx-1],
            'W_opt': W_opt, 'tau_final': tau_final, 'color': colors_L[L],
            'val_at_cut': Cm[W_opt] if W_opt < len(Cm) else 0
        })

    # -- Plot 1: C(dt) at beta_c with Madras-Sokal cut-off markers
    fig1, ax1 = setup_plot(r"$\Delta t$ [MCS]", r"$C(\Delta t)$")
    max_x = 0
    for d in plot_data:
        ax1.plot(d['lags'], d['Cm'], color=d['color'], lw=2, 
                alpha=0.9, label=f"L={d['L']}")
        ax1.plot(d['W_opt'], d['val_at_cut'], 'o', color=d['color'], 
                markersize=4, zorder=10)
        max_x = max(max_x, d['W_opt'])

    ax1.legend(loc='upper right', frameon=False, fontsize=12, ncol=2)
    ax1.set_xlim(0, max_x * 1.05)
    ax1.axhline(0, color='k', ls=':', alpha=0.5)
    plt.savefig(PLOT_DIR / "autocorr_critical_all.pdf")
    print(f"    Saved: {PLOT_DIR / 'autocorr_critical_all.pdf'}")
    plt.close()

    # -- Plot 2: running tau_int(W) with W = c*tau line
    print("\n[3/4] Plotting tau convergence...")
    fig2, ax2 = setup_plot(r"Window $W$ [MCS]", r"$\tau_{int}(W)$")
    max_tau_lag = 0
    for d in plot_data:
        ax2.plot(d['tau_lags'], d['tau_vals'], color=d['color'], 
                lw=2, alpha=0.9, label=f"L={d['L']}")
        ax2.plot(d['W_opt'], d['tau_final'], 'o', color=d['color'], 
                markersize=MARKER_SIZE, zorder=10)
        max_tau_lag = max(max_tau_lag, d['W_opt'])

    w_range = np.linspace(10, max_tau_lag, 1000)
    ax2.plot(w_range, w_range/C_WINDOW, 'k--', lw=1.5, 
            alpha=0.6, label=r"$W = 6\tau$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(100, max_tau_lag * 1.5)
    ax2.set_ylim(bottom=50)
    ax2.legend(loc='lower right', frameon=False, fontsize=12, ncol=2)
    
    plt.savefig(PLOT_DIR / "tau_convergence_critical_all.pdf")
    print(f"    Saved: {PLOT_DIR / 'tau_convergence_critical_all.pdf'}")
    plt.close()

# ---------------------------------------------------------------------------
# Multi-observable tau_int comparison
# ---------------------------------------------------------------------------
def analyze_tau_comparison():
    """Compare tau_int(M), tau_int(M^2), tau_int(M^4), tau_int(E) at beta_c.

    Prints a summary table with cross-observable ratios relevant for FSS
    error estimation and saves numerical results to a .dat file.
    """
    print("\n" + "="*70)
    print("  [4/4] TAU_INT COMPARISON: Different Observables at β_c")
    print("="*70)
    
    observables = ['m', 'm2', 'm4', 'e']
    obs_labels = {
        'm': 'M (magnetization)',
        'm2': 'M² (second moment)',
        'm4': 'M⁴ (fourth moment)',
        'e': 'E (energy)'
    }
    
    results = []
    
    for L in L_ALL:
        gc.collect()
        fpath, actual_beta = find_file(L, BETA_CRIT)
        if not fpath:
            print(f"[SKIP] L={L}: No file found at beta_c")
            continue
        
        print(f"\n--- L = {L} (β = {actual_beta:.6f}) ---")
        tau_dict = {'L': L, 'beta': actual_beta}
        
        for obs in observables:
            data_raw = read_observable_safe(fpath, obs, L=L)
            if len(data_raw) == 0:
                print(f"  [WARN] {obs}: Failed to load")
                continue
            
            # Discard 10 % burn-in
            data = data_raw[int(len(data_raw)*0.1):]
            del data_raw

            # Integrated autocorrelation time via Madras-Sokal windowing
            lags, C = compute_autocorr_fft(data, MAX_LAG)
            tau_vals, W_opt, tau_final, converged = compute_madras_sokal(
                lags, C, C_WINDOW
            )
            
            tau_dict[obs] = tau_final
            tau_dict[f'{obs}_W'] = W_opt
            
            status = "✓" if converged else "⚠"
            print(f"  {status} {obs_labels[obs]:<25} τ_int = {tau_final:>8.0f}  "
                  f"(W={W_opt:>8.0f})")
        
        # Cross-observable tau ratios (relevant for blocking / error bars)
        if 'm' in tau_dict and 'm2' in tau_dict:
            ratio_m_m2 = tau_dict['m'] / tau_dict['m2']
            tau_dict['ratio_m_m2'] = ratio_m_m2
            print(f"\n  ➤ Ratio τ(M)/τ(M²)  = {ratio_m_m2:.2f}")
        
        if 'm' in tau_dict and 'e' in tau_dict:
            ratio_m_e = tau_dict['m'] / tau_dict['e']
            tau_dict['ratio_m_e'] = ratio_m_e
            print(f"  ➤ Ratio τ(M)/τ(E)   = {ratio_m_e:.2f}")
        
        if 'm2' in tau_dict and 'm4' in tau_dict:
            ratio_m2_m4 = tau_dict['m2'] / tau_dict['m4']
            tau_dict['ratio_m2_m4'] = ratio_m2_m4
            print(f"  ➤ Ratio τ(M²)/τ(M⁴) = {ratio_m2_m4:.2f}")
        
        results.append(tau_dict)
    
    # Summary table
    print("\n" + "="*78)
    print("  SUMMARY TABLE: τ_int at β_c for All Observables")
    print("="*78)
    print(f"{'L':<6} {'τ(M)':<10} {'τ(M²)':<10} {'τ(M⁴)':<10} {'τ(E)':<10} "
          f"{'M/M²':<8} {'M/E':<8} {'M²/M⁴':<8}")
    print("-"*78)
    
    for r in results:
        print(f"{r['L']:<6} "
              f"{r.get('m', -1):>9.0f} "
              f"{r.get('m2', -1):>9.0f} "
              f"{r.get('m4', -1):>9.0f} "
              f"{r.get('e', -1):>9.0f} "
              f"{r.get('ratio_m_m2', 0):>7.2f} "
              f"{r.get('ratio_m_e', 0):>7.2f} "
              f"{r.get('ratio_m2_m4', 0):>7.2f}")
    
    print("="*78)
    
    # Save to file
    out_file = ANALYSIS_DIR / "tau_int_comparison.dat"
    
    with open(out_file, 'w') as f:
        f.write("# tau_int comparison at beta_c for different observables\n")
        f.write("# Generated by autocorr.py (multi-observable analysis)\n")
        f.write(f"# beta_c = {BETA_CRIT:.17f}\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("#  L        : Lattice size\n")
        f.write("#  tau_m    : tau_int for magnetization M\n")
        f.write("#  tau_m2   : tau_int for second moment M²\n")
        f.write("#  tau_m4   : tau_int for fourth moment M⁴\n")
        f.write("#  tau_e    : tau_int for energy E\n")
        f.write("#  ratio_m_m2  : tau(M) / tau(M²)\n")
        f.write("#  ratio_m_e   : tau(M) / tau(E)\n")
        f.write("#  ratio_m2_m4 : tau(M²) / tau(M⁴)\n")
        f.write("#\n")
        f.write(f"{'# L':<6} {'tau_m':<10} {'tau_m2':<10} {'tau_m4':<10} "
                f"{'tau_e':<10} {'M/M2':<10} {'M/E':<10} {'M2/M4':<10}\n")
        
        for r in results:
            f.write(f"{r['L']:<6} "
                   f"{r.get('m', -1):>9.1f} "
                   f"{r.get('m2', -1):>9.1f} "
                   f"{r.get('m4', -1):>9.1f} "
                   f"{r.get('e', -1):>9.1f} "
                   f"{r.get('ratio_m_m2', -1):>9.3f} "
                   f"{r.get('ratio_m_e', -1):>9.3f} "
                   f"{r.get('ratio_m2_m4', -1):>9.3f}\n")
    
    print(f"\n✓ Numerical data saved: {out_file}")
    
    # Interpretation (IMPROVED FOR U4 CONTEXT)
    if results:
        avg_ratio_m_m2 = np.mean([r.get('ratio_m_m2', 0) for r in results if 'ratio_m_m2' in r])
        avg_ratio_m_e = np.mean([r.get('ratio_m_e', 0) for r in results if 'ratio_m_e' in r])
        avg_ratio_m2_m4 = np.mean([r.get('ratio_m2_m4', 0) for r in results if 'ratio_m2_m4' in r])
        
        print("\n" + "="*70)
        print("  INTERPRETATION FOR FSS ERROR CORRECTION")
        print("="*70)
        print(f"  Average τ(M)/τ(M²)  = {avg_ratio_m_m2:.2f}")
        print(f"  Average τ(M)/τ(E)   = {avg_ratio_m_e:.2f}")
        print(f"  Average τ(M²)/τ(M⁴) = {avg_ratio_m2_m4:.2f}")
        print()
        print("  Physical Interpretation:")
        print("  ─────────────────────────────────────────────────────────")
        print("  • M is an ODD operator → requires spin-flip tunneling")
        print("    τ(M) scales with dynamic exponent z ≈ 2.17")
        print()
        print("  • M², M⁴, E are EVEN operators → decorrelate ~2× faster")
        print("    No tunneling needed, fluctuate within single phase")
        print()
        print("  • Binder cumulant U₄ = 1 - ⟨m⁴⟩/(3⟨m²⟩²)")
        print("    Not a per-sweep observable, but ensemble average")
        print("    Effective τ(U₄) ≈ max(τ(m²), τ(m⁴)) via jackknife")
        print()
        print("  Impact on FSS Error Estimation:")
        print("  ─────────────────────────────────────────────────────────")
        print(f"  If block_size ∝ τ(M) was used for U₄:")
        print(f"    → Overestimate by factor √{avg_ratio_m_m2:.1f} ≈ {np.sqrt(avg_ratio_m_m2):.2f}")
        print()
        print(f"  If block_size ∝ L^z (from τ(M) fit) was used:")
        print(f"    → Additional overestimate from z ≈ 2.17")
        print()
        print("  Combined with jackknife instability (few β points):")
        print("  → Total error overestimation ~10× observed in preliminary FSS")
        print("="*70)

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
    
    plot_temp_dependence()          # C(dt) at different T for L=24, 64
    plot_critical_slowing_down()     # C(dt) + tau convergence at beta_c

    # Multi-observable tau_int comparison at beta_c
    analyze_tau_comparison()
    
    print("\n" + "="*70)
    print("  ✓ Extended autocorrelation analysis complete.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
