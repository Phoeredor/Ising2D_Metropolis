#!/usr/bin/env python3
"""
Parse binary MC output into per-beta observables (JSON).

Reads each .bin file for a given L, applies data-blocking for primary
observables (E, |m|, m, m^2) and jackknife for derived ones (chi, Cv, U4).
Output: results/production/L{L}/parsed/observables_L{L}.json
"""

import sys
import json
import struct
import glob
import numpy as np
from pathlib import Path
import gc

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Binary file layout produced by main_prod.c
HEADER_FMT = "ii d qqq II"   # (L, L, beta, sweeps, therm, measure, stride, seed)
HEADER_SIZE = 48
RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"), ("e", "<f8"), ("m", "<f8"),
    ("e2", "<f8"), ("m2", "<f8"), ("m4", "<f8"),
])

def estimate_tau_int(series):
    """Quick estimate of integrated autocorrelation time via blocking."""
    N = len(series)
    if N < 100:
        return 0.5
    max_p = int(np.log2(N / 20))
    sigmas = []
    for p in range(max_p):
        k = 2**p
        nb = N // k
        if nb < 2:
            break
        m_blocks = series[:nb * k].reshape(nb, k).mean(axis=1)
        sigmas.append(np.var(m_blocks, ddof=1) * k)
    if not sigmas:
        return 0.5
    naive_var = np.var(series, ddof=1)
    # tau_int ~ 0.5 * sigma_blocked^2 / sigma_naive^2
    return max(0.5, 0.5 * (np.max(sigmas) / naive_var)) if naive_var > 0 else 0.5

def jackknife_err(blocks, func, V):
    """Delete-1 jackknife error for a nonlinear function of block means."""
    N = len(blocks[0])
    sums = [np.sum(b) for b in blocks]
    val_full = func([s / N for s in sums], V)
    j_vals = np.zeros(N)
    for i in range(N):
        means_i = [(s - b[i]) / (N - 1) for s, b in zip(sums, blocks)]
        j_vals[i] = func(means_i, V)
    err = np.sqrt((N - 1) * np.mean((j_vals - val_full)**2))
    return val_full, err

# Derived observables: chi'(|m|), Cv(e), Binder U4(m^2, m^4)
def calc_chi(means, V): return V * (means[1] - means[0]**2)
def calc_cv(means, V):  return V * (means[1] - means[0]**2)
def calc_u4(means, V):
    denom = 3 * means[1]**2
    return 1.0 - means[0] / denom if denom != 0 else 0.0

def process_L(L_target):
    """Parse all binary files for lattice size L and write JSON observables."""
    bin_dir = PROJECT_ROOT / "results" / "production" / f"L{L_target}" / "bin"
    out_dir = PROJECT_ROOT / "results" / "production" / f"L{L_target}" / "parsed"
    out_json = out_dir / f"observables_L{L_target}.json"

    if not bin_dir.exists():
        print(f"[ERR] Directory not found: {bin_dir}")
        return

    files = sorted(glob.glob(str(bin_dir / "*.bin")))
    print(f"--- Analysis L={L_target}: {len(files)} files found ---")

    results = []
    V = L_target**2

    for f_path in files:
        try:
            with open(f_path, "rb") as f:
                h = struct.unpack(HEADER_FMT, f.read(HEADER_SIZE))
                beta = h[2]
                stride = h[5]
                data = np.fromfile(f, dtype=RECORD_DTYPE)
            
            tau_est = estimate_tau_int(data['m2'])
            # Block size: max of (safe minimum, 10*tau, 5*expected_tau_scaling)
            block_size = int(max(2000, 10.0 * tau_est, 5.0 * (0.5 * L_target**2.17 / stride)))

            n_blocks = len(data) // block_size
            if n_blocks < 10:
                print(f"[SKIP] {Path(f_path).name}: Insufficient stats")
                continue

            t_len = n_blocks * block_size

            # Block averages for primary observables
            e_b  = data['e'][:t_len].reshape(n_blocks, block_size).mean(axis=1)
            e2_b = data['e2'][:t_len].reshape(n_blocks, block_size).mean(axis=1)
            m2_b = data['m2'][:t_len].reshape(n_blocks, block_size).mean(axis=1)
            m4_b = data['m4'][:t_len].reshape(n_blocks, block_size).mean(axis=1)
            m_signed_b = data['m'][:t_len].reshape(n_blocks, block_size).mean(axis=1)
            ma_b = np.abs(data['m'][:t_len]).reshape(n_blocks, block_size).mean(axis=1)

            # Primary observables: blocking error (sigma/sqrt(n_blocks))
            E_mean, E_err = np.mean(e_b), np.std(e_b, ddof=1) / np.sqrt(n_blocks)
            M_signed_mean, M_signed_err = np.mean(m_signed_b), np.std(m_signed_b, ddof=1) / np.sqrt(n_blocks)
            M_abs_mean, M_abs_err = np.mean(ma_b), np.std(ma_b, ddof=1) / np.sqrt(n_blocks)
            M2_mean, M2_err = np.mean(m2_b), np.std(m2_b, ddof=1) / np.sqrt(n_blocks)

            # Derived observables: jackknife error (nonlinear functions)
            chi, chi_e = jackknife_err([ma_b, m2_b], calc_chi, V)
            cv, cv_e   = jackknife_err([e_b, e2_b], calc_cv, V)
            u4, u4_e   = jackknife_err([m4_b, m2_b], calc_u4, V)
            
            results.append({
                "beta": beta, 
                "E_mean": E_mean, "E_err": E_err,
                "M_val": M_signed_mean, "M_err": M_signed_err,
                "M_abs": M_abs_mean, "M_abs_err": M_abs_err,
                "M2_mean": M2_mean, "M2_err": M2_err,
                "chi": chi, "chi_err": chi_e,
                "C": cv, "C_err": cv_e, 
                "U4": u4, "U4_err": u4_e
            })
            print(f"[OK] Beta {beta:.5f}")
            del data
            gc.collect()
        except Exception as e:
            print(f"[ERR] {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results.sort(key=lambda x: x["beta"])
    with open(out_json, "w") as j:
        json.dump(results, j, indent=2)
    print(f"Output: {out_json}")

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_L(int(sys.argv[1]))
