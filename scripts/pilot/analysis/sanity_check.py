#!/usr/bin/env python3
"""
sanity_check.py

Performs validation checks on the pilot analysis reports (JSON).
Ensures that estimated parameters are physically reasonable before
proceeding to production planning.

Checks:
- Non-negative autocorrelation times (tau_int).
- RNG/Up compatibility rates.
- Statistical quality (N_samples / tau_int) distribution.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import numpy as np

# Default Lattice sizes to check if none provided
DEFAULT_L_LIST = [10, 16, 24, 32, 48, 64, 96, 128]

def check_single_L(L: int, results_root: Path) -> None:
    """
    Validates the analysis report for a specific lattice size L.
    """
    # Construct path: results/pilot/L{L}/analysis_report_L{L}.json
    report_path = results_root / f"L{L}" / f"analysis_report_L{L}.json"

    print(f"=== Sanity Check: L = {L} ===")

    if not report_path.exists():
        print(f"  [WARN] Report file not found: {report_path}")
        print("         Skipping this size.\n")
        return

    try:
        with open(report_path, "r") as f:
            rep = json.load(f)
    except Exception as exc:
        print(f"  [ERROR] Failed to read JSON: {exc}\n")
        return

    results = rep.get("results", [])
    if not results:
        print("  [WARN] Report contains no results data.\n")
        return

    # Metrics containers
    neg_meas = []
    neg_sweeps = []
    rng_up_ok = 0
    rng_up_fail = 0
    fail_list = []
    N_over_tau_vals = []

    for r in results:
        beta = float(r.get("beta", float("nan")))
        tau_meas = float(r.get("tau_int_meas", 0.0))
        tau_sw = float(r.get("tau_int_sweeps", 0.0))

        # 1. Check Physicality of Tau
        if tau_meas < 0:
            neg_meas.append((beta, tau_meas))
        if tau_sw < 0:
            neg_sweeps.append((beta, tau_sw))

        # 2. Check Hot/Cold Compatibility
        hc = bool(r.get("hot_cold_compatible", False))
        if hc:
            hotcold_ok += 1
        else:
            hotcold_fail += 1
            flips = int(r.get("n_flips", -1))
            fail_list.append((beta, flips))

        # 3. Check Statistical Quality (N / tau)
        val = r.get("N_over_tau_meas", None)
        if hc and val is not None:
            try:
                v = float(val)
                if math.isfinite(v) and v > 0:
                    N_over_tau_vals.append(v)
            except (TypeError, ValueError):
                pass

    # --- Reporting ---
    print(f"  Total Beta Points      : {len(results)}")
    print(f"  RNG/Up Compatible    : {rng_up_ok}")
    print(f"  RNG/Up Incompatible  : {rng_up_fail}")

    # Report Negative Tau Errors
    if neg_meas or neg_sweeps:
        print("  [ERROR] Negative autocorrelation times detected!")
        for beta, tau in neg_meas:
            print(f"    beta={beta:.6f} : tau_meas={tau:.3f}")
        for beta, tau in neg_sweeps:
            print(f"    beta={beta:.6f} : tau_sweeps={tau:.3f}")
    else:
        print("  [OK] All tau_int values are non-negative.")

    # Report Incompatible Points
    if fail_list:
        print("  [WARN] Incompatible Hot/Cold points (Possible hysteresis/tunneling issues):")
        for beta, flips in fail_list:
            print(f"    beta={beta:.6f}, flips={flips}")
    else:
        print("  [OK] All points are Hot/Cold compatible.")

    # Report Statistical Quality Stats
    if N_over_tau_vals:
        arr = np.array(N_over_tau_vals, dtype=float)
        print("  Statistical Quality (N_samples / tau_int):")
        print(f"    Min    : {arr.min():.1f}")
        print(f"    Median : {np.median(arr):.1f}")
        print(f"    Max    : {arr.max():.1f}")
        
        count_low = int((arr < 500.0).sum())
        if count_low > 0:
            print(f"    [WARN] {count_low} points have low statistics (N/tau < 500)")
        else:
            print("    [OK] All compatible points have sufficient statistics (N/tau >= 500)")
    else:
        print("  [WARN] No valid N/tau metrics found for compatible points.")

    print("") # Newline separator

def main() -> None:
    ap = argparse.ArgumentParser(description="Run sanity checks on pilot analysis reports.")
    ap.add_argument("L", nargs="*", type=int, help="Specific Lattice sizes to check (optional)")
    args = ap.parse_args()

    # Resolve Root Path
    project_root = Path(__file__).resolve().parents[3]
    results_root = project_root / "results" / "pilot"

    if not results_root.exists():
        print(f"[ERROR] Results directory not found: {results_root}")
        sys.exit(1)

    # Determine L list
    L_list = args.L if args.L else DEFAULT_L_LIST

    # Check existence of L directories first to avoid noise
    existing_L = []
    for L in L_list:
        if (results_root / f"L{L}").exists():
            existing_L.append(L)
    
    if not existing_L:
        print(f"[ERROR] No data directories found for requested sizes in {results_root}")
        print(f"        Available dirs: {[d.name for d in results_root.glob('L*')]}")
        sys.exit(1)

    # Run Checks
    for L in existing_L:
        check_single_L(L, results_root)

if __name__ == "__main__":
    main()
