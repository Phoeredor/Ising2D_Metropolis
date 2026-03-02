#!/usr/bin/env python3
"""
check_integrity.py

Validates the integrity of pilot simulation data.

Features:
- Scans `results/pilot/L{L}/bin` for output files.
- Automatically detects ANY L size (including 96, 128, etc.).
- Checks binary header against expected file size.
- Verifies that measurements match the declared stride and sweep count.
- Detects truncated or corrupt files.

Usage:
  python3 scripts/pilot/check_integrity.py
"""

import os
import struct
import sys
from pathlib import Path
import math

# =============================================================================
# Configuration
# =============================================================================

# Project Root Resolution
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "results" / "pilot"

# Binary Header Structure (Must match C struct alignment)
# 4i (L) + 4x (pad) + 8d (beta) + 8q (n_therm) + 8q (n_sweeps) + 8q (stride) + 4I (seed1) + 4I (seed2)
HEADER_FMT = "ii d qqq II"
HEADER_SIZE = 4 + 4 + 8 + 8 + 8 + 8 + 4 + 4  # = 48 bytes

# Measurement Record Structure
# 8q (sweep) + 8d (e) + 8d (m) + 8d (e2) + 8d (m2) + 8d (m4)
RECORD_SIZE = 8 + 8 + 8 + 8 + 8 + 8          # = 48 bytes

# =============================================================================
# Validation Logic
# =============================================================================

def check_file(filepath: Path) -> str:
    """
    Validates a single binary file.
    Returns "OK" or an error description string.
    """
    if not filepath.exists():
        return "MISSING"
    
    try:
        file_size = filepath.stat().st_size
        
        # 1. Check Header Existence
        if file_size < HEADER_SIZE:
            return f"TRUNCATED_HEADER (Size: {file_size} bytes)"

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            # Unpack: L, _pad, beta, n_therm, n_sweeps, stride, seed1, seed2
            try:
                L, _, beta, n_therm, n_sweeps, stride, _, _ = struct.unpack(HEADER_FMT, header_bytes)
            except struct.error:
                 return "HEADER_UNPACK_ERROR"

            # 2. logical Consistency
            if L <= 0 or n_sweeps <= 0 or stride <= 0:
                return f"INVALID_METADATA (L={L}, N={n_sweeps}, Stride={stride})"

            # 3. Size Validation
            # Expected measurements = n_sweeps / stride
            expected_measurements = n_sweeps // stride
            expected_size = HEADER_SIZE + (expected_measurements * RECORD_SIZE)
            
            if file_size != expected_size:
                diff_bytes = expected_size - file_size
                missing_records = diff_bytes / RECORD_SIZE
                
                if diff_bytes > 0:
                     return f"INCOMPLETE (Missing ~{missing_records:.1f} records)"
                else:
                     return f"OVERSIZED (Extra bytes: {-diff_bytes})"

            return "OK"

    except Exception as e:
        return f"CORRUPT ({str(e)})"

def get_L_value(path: Path) -> int:
    """Helper to sort directories numerically (L10 < L96 < L100)."""
    try:
        return int(path.name.replace("L", ""))
    except ValueError:
        return 999999

def main():
    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root directory not found: {DATA_ROOT}")
        print("        Have you run the pilot simulations yet?")
        sys.exit(1)

    # Auto-discovery of L directories
    # We sort them numerically so L96 and L128 appear in correct order
    l_dirs = sorted(DATA_ROOT.glob("L*"), key=get_L_value)
    
    if not l_dirs:
        print(f"[WARNING] No L* directories found in {DATA_ROOT}")
        sys.exit(0)

    # Print confirmation of detected sizes
    detected_sizes = [d.name for d in l_dirs]
    print(f"--> Detected Lattice Sizes: {', '.join(detected_sizes)}")
    print("-" * 80)
    print(f"{'File':<45} {'Status':<20} {'Detail'}")
    print("-" * 80)

    total_files = 0
    bad_files = 0

    for l_dir in l_dirs:
        bin_dir = l_dir / "bin"
        if not bin_dir.exists():
            continue
            
        bin_files = sorted(bin_dir.glob("*.bin"))
        
        for filepath in bin_files:
            status = check_file(filepath)
            filename = filepath.name
            
            # Colorized Output
            if status == "OK":
                status_colored = "\033[92mOK\033[0m" # Green
            else:
                status_colored = f"\033[91m{status.split(' ')[0]}\033[0m" # Red
                bad_files += 1

            # Print concise report
            detail = status if status != "OK" else ""
            print(f"{filename:<45} {status_colored:<30} {detail}")
            total_files += 1

    print("-" * 80)
    print(f"Total Files Scanned: {total_files}")
    print(f"Corrupt/Incomplete:  {bad_files}")
    
    if bad_files == 0 and total_files > 0:
        print("\n\033[92mSUCCESS: All found pilot data appears valid.\033[0m")
    elif total_files == 0:
        print("\n\033[93mWARNING: No binary files found to check.\033[0m")
    else:
        print("\n\033[91mFAILURE: Some files are corrupt or incomplete. Check logs.\033[0m")
        sys.exit(1)

if __name__ == "__main__":
    main()
