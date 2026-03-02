#!/usr/bin/env python3
"""
generate_plan.py

Generates the definitive Production Plan based on Finite-Size Scaling (FSS).

Strategy:
  1. Uses a universal scaling variable 'x = (beta - beta_c) * L^(1/nu)'.
  2. Maps 'x' to physical 'beta' for each Lattice size L.
  3. Interpolates 'tau_int' from pilot runs to determine required statistics.
  4. Filters beta values strictly within [0.35, 0.50].

Input: results/pilot/prod_params_from_pilots.json
Output: results/production/production_plan.dat
"""

import sys
import json
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# =============================================================================
# Configuration
# =============================================================================

# Exact critical inverse temperature for 2D Ising (Onsager)
BETA_C = 0.44068679350977151262 
L_LIST = [10, 16, 24, 32, 48, 64, 96, 128]
NU = 1.0  # Correlation length exponent

# --- HARD CUTS (Physics Filter) ---
BETA_MIN = 0.3500
BETA_MAX = 0.5000

# Statistical Targets (Effective independent measurements)
TARGET_N_EFF_CRIT = 20000  # Critical region (high precision)
TARGET_N_EFF_WING = 5000   # Off-critical region

# Safety Floors (Minimum sweeps regardless of tau)
MIN_PROD_SWEEPS = 100000
MIN_THERM_SWEEPS = 10000

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_JSON   = PROJECT_ROOT / "results" / "pilot" / "prod_params_from_pilots.json"
OUTPUT_PLAN  = PROJECT_ROOT / "results" / "production" / "production_plan.dat"

# =============================================================================
# Logic
# =============================================================================

def generate_x_grid():
    """Generates the universal grid in scaling variable x."""
    x_points = set()
    # 1. Ultra-Critical (|x| <= 1.5) -> Very dense step 0.05
    for x in np.arange(-1.5, 1.51, 0.05): x_points.add(round(x, 4))
    # 2. Scaling Region (1.5 < |x| <= 5.0) -> Dense step 0.25
    for x in np.arange(-5.0, 5.01, 0.25): x_points.add(round(x, 4))
    # 3. Wings (5.0 < |x| <= 10.0) -> Coarse step 1.0
    for x in np.arange(-10.0, 10.01, 1.0): x_points.add(round(x, 4))

    x_points.add(0.0) # Ensure exact criticality
    return sorted(list(x_points))

def get_interpolators(pilot_data, L):
    """Creates interpolation functions for tau_int and meas_stride from pilot data."""
    entries = [p for p in pilot_data if int(p["L"]) == L]
    if not entries: return None, None

    # Sort by beta to ensure monotonic interpolation domain
    entries.sort(key=lambda x: x["beta"])

    X = np.array([e["beta"] for e in entries])
    # Use a safety floor for tau (0.5)
    Y_tau = np.array([max(0.5, e["tau_int_sweeps"]) for e in entries])
    Y_stride = np.array([e["meas_stride"] for e in entries])

    # Linear interpolation for Tau, Nearest Neighbor for Stride
    f_tau = interp1d(X, Y_tau, kind='linear', fill_value="extrapolate", bounds_error=False)
    f_stride = interp1d(X, Y_stride, kind='nearest', fill_value="extrapolate", bounds_error=False)
    return f_tau, f_stride

def main():
    print("--- GENERATING PRODUCTION PLAN (FSS Single-Shot) ---")

    if not PILOT_JSON.exists():
        print(f"[ERR] Pilot data not found: {PILOT_JSON}")
        print("      Run 'scripts/pilot/run_pilot_analysis.sh' first.")
        sys.exit(1)

    try:
        with open(PILOT_JSON, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict): data = data["points"]
    except Exception as e:
        print(f"[ERR] JSON Parse Error: {e}"); sys.exit(1)

    base_xs = generate_x_grid()
    plan_rows = []

    for L in L_LIST:
        interp_tau, interp_stride = get_interpolators(data, L)
        if not interp_tau:
            print(f"[WARN] No pilot data for L={L}. Skipping.")
            continue

        # Smart Reduction for largest lattice (L=64) to save time in wings
        current_xs = []
        if L >= 64:
            for i, x in enumerate(base_xs):
                if abs(x) < 3.0: current_xs.append(x) # Full density in critical zone
                elif i % 2 == 0: current_xs.append(x) # Half density in wings
        else:
            current_xs = base_xs

        for x in current_xs:
            # Map x -> Beta
            offset = x / (L ** (1.0/NU))
            beta = BETA_C + offset

            # --- HARD FILTER ---
            if beta < BETA_MIN or beta > BETA_MAX:
                continue

            # Estimate simulation parameters
            tau_est = max(0.5, float(interp_tau(beta)))
            stride_est = max(1, int(interp_stride(beta)))

            # Set target statistics based on regime
            if abs(x) <= 2.0:
                neff = TARGET_N_EFF_CRIT
                regime = "CRIT"
            elif abs(x) <= 5.0:
                neff = TARGET_N_EFF_WING
                regime = "SCAL"
            else:
                neff = 3000
                regime = "WING"

            # Calculate total sweeps (N_eff * 2 * Tau)
            n_sweeps = max(MIN_PROD_SWEEPS, int(math.ceil(neff * 2.0 * tau_est)))
            # Thermalization (Sokal's criterion: >= 20 * Tau)
            n_therm = max(MIN_THERM_SWEEPS, int(math.ceil(20.0 * tau_est)))

            plan_rows.append({
                "L": L, "beta": beta, "x_val": x, "stride": stride_est,
                "n_therm": n_therm, "n_prod": n_sweeps, "regime": regime
            })

    # Write Output
    OUTPUT_PLAN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PLAN, "w") as f:
        f.write("# L beta meas_stride n_therm n_sweeps x_val type\n")
        for p in plan_rows:
            # Use %.6f for beta to match C code precision
            f.write(f"{p['L']} {p['beta']:.6f} {p['stride']} {p['n_therm']} {p['n_prod']} {p['x_val']:.4f} {p['regime']}\n")

    print(f"[OK] Plan saved to: {OUTPUT_PLAN}")
    print(f"     Total Runs: {len(plan_rows)}")

if __name__ == "__main__":
    main()
