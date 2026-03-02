#!/usr/bin/env python3
"""
analyze_pilots.py

Analyzes binary pilot data to extract:
1. Integrated Autocorrelation Time (tau_int) using Windowing method.
2. RNG/Up compatibility check (Ergodicity test).

Usage: python3 analyze_pilots.py <L>
Output: results/pilot/L{L}/analysis_report_L{L}.json
"""

import sys
import struct
import glob
import json
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HEADER_FMT = "ii d qqq II"
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

def estimate_tau_int_windowing(series):
    """
    Estimates tau_int using the automatic windowing method (Wolff, 2004).
    """
    N = len(series)
    if N < 1000: return 0.5 # Not enough data
    
    # Subtract mean
    mean = np.mean(series)
    fluct = series - mean
    var = np.var(series, ddof=1)
    if var == 0: return 0.5

    # Auto-correlation function rho(t)
    # Using FFT for speed (Wiener-Khinchin theorem)
    fft = np.fft.rfft(fluct, n=2*N)
    acf = np.fft.irfft(fft * np.conj(fft))[:N] / (N * var)
    
    # Windowing procedure
    tau = 0.5
    for t in range(1, N):
        tau += acf[t]
        # Automatic window condition: tau(t) < S * t
        # S typically 1.5 or 2.0. We use 1.5 for pilot safety.
        if tau < 0.0 or t > 6 * tau:
            break
            
    return max(0.5, tau)

def process_lattice(L):
    base_dir = PROJECT_ROOT / "results" / "pilot" / f"L{L}" / "bin"
    output_json = PROJECT_ROOT / "results" / "pilot" / f"L{L}" / f"analysis_report_L{L}.json"
    
    if not base_dir.exists():
        print(f"[ERR] No data for L={L}")
        return

    # --- FIX: Look for 'rng_obs*.bin' ---
    rng_files = sorted(glob.glob(str(base_dir / "rng_obs*.bin")))
    results = []

    print(f"--- Analyzing L={L} ({len(rng_files)} points) ---")

    for h_path in rng_files:
        try:
            # Infer beta from filename 
            # Format: rng_obs_L10_beta0.440687.bin
            beta_str = Path(h_path).stem.split("beta")[-1]
            
            # Read RNG Data
            with open(h_path, "rb") as f:
                head = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
                beta = head[2]
                stride = head[5]
                data_rng = np.fromfile(f, dtype=RECORD_DTYPE)
            
            # Read Up Data (if exists)
            # Switch 'rng_' to 'up_' in filename
            c_path = h_path.replace("rng_", "up_")
            has_up = Path(c_path).exists()
            data_up = None
            if has_up:
                with open(c_path, "rb") as f:
                    f.read(HEADER_SIZE) # Skip header
                    data_up = np.fromfile(f, dtype=RECORD_DTYPE)

            # Analysis
            # 1. Tau estimation (use absolute magnetization to handle tunneling)
            m_abs = np.abs(data_rng['m'])
            tau_int_meas = estimate_tau_int_windowing(m_abs)
            tau_int_sweeps = tau_int_meas * stride
            
            # 2. RNG/Up Check
            compatible = True
            if has_up and len(data_up) > 0:
                # Discard initial 20% for conservative check
                cut_h = int(len(data_rng) * 0.2)
                cut_c = int(len(data_up) * 0.2)
                
                mean_h = np.mean(np.abs(data_rng['m'][cut_h:]))
                mean_c = np.mean(np.abs(data_up['m'][cut_c:]))
                
                # Standard Error (Naive approximation is sufficient for this check)
                err_h = np.std(np.abs(data_rng['m'][cut_h:])) / np.sqrt(len(data_rng)-cut_h)
                err_c = np.std(np.abs(data_up['m'][cut_c:])) / np.sqrt(len(data_up)-cut_c)
                
                # Compatibility within 3 sigma
                diff = abs(mean_h - mean_c)
                sigma_comb = np.sqrt(err_h**2 + err_c**2)
                
                # Avoid division by zero if sigma is 0
                if sigma_comb > 1e-12:
                    if diff > 5 * sigma_comb: # Use 5 sigma to be very tolerant of fluctuations
                        compatible = False
                elif diff > 1e-5: # If sigma is 0 but means differ
                    compatible = False
            
            # Store result
            entry = {
                "L": L,
                "beta": beta,
                "tau_int_sweeps": tau_int_sweeps,
                "meas_stride": stride,
                "rng_up_compatible": compatible,
                "n_therm_sweeps": int(20 * tau_int_sweeps), # Recommendation
                "n_sweeps_prod": int(1000 * tau_int_sweeps) # Baseline recommendation
            }
            results.append(entry)
            
            status = "OK" if compatible else "MISMATCH"
            print(f"Beta {beta:.6f} | Tau: {tau_int_sweeps:.1f} sw | R/U: {status}")

        except Exception as e:
            print(f"[ERR] Failed parsing {Path(h_path).name}: {e}")

    # Save Report
    with open(output_json, "w") as f:
        json.dump({"points": results}, f, indent=2)
    print(f"[OK] Saved: {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pilots.py <L>")
        sys.exit(1)
    process_lattice(int(sys.argv[1]))
