#!/usr/bin/env python3
"""
Integrity audit for production binary files.

Scans *.bin files for a given lattice size L and checks:
  1. Truncation  – file smaller than the 48-byte header.
  2. Beta mismatch – filename beta vs header beta beyond tolerance.
  3. Append collision – file size >> expected from header params.

Usage:  python3 audit_production_files.py <L>
"""

import struct
import glob
import os
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Header layout (C struct, little-endian):
#   int32 L, int32 pad, double beta, int64 n_therm, int64 n_sweeps,
#   int64 stride, uint32 seed1, uint32 seed2
HEADER_FMT  = "ii d qqq II"
HEADER_SIZE = 48
RECORD_SIZE = 48   # bytes per measurement record

BETA_TOLERANCE     = 1e-5   # max |beta_filename - beta_header|
SIZE_TOLERANCE_RATIO = 1.10 # flag if file > 110 % of expected size

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_beta_from_filename(fname):
    """Parse the beta value encoded in the filename (*beta<VALUE>.bin)."""
    try:
        base = os.path.basename(fname)
        parts = base.split('beta')
        if len(parts) < 2:
            return None
        val_str = parts[1].split('.bin')[0]
        return float(val_str)
    except ValueError:
        return None


def audit_directory(L):
    """Validate every .bin file under results/production/L{L}/bin/.

    Returns a list of file paths flagged for deletion.
    """
    data_dir = PROJECT_ROOT / "results" / "production" / f"L{L}" / "bin"
    files = sorted(glob.glob(str(data_dir / "*.bin")))

    print(f"\n--- AUDIT REPORT: L={L} ({len(files)} files found) ---")
    print(f"{'FILENAME':<40} | {'HEADER BETA':<12} | {'STATUS':<10} | {'ACTION'}")
    print("-" * 90)

    files_to_delete = []

    for f_path in files:
        f_name = os.path.basename(f_path)
        filename_beta = get_beta_from_filename(f_path)

        try:
            file_size = os.path.getsize(f_path)

            # Check 1: truncated / empty file
            if file_size < HEADER_SIZE:
                print(f"\033[91m{f_name:<40} | {'EMPTY':<12} | BROKEN     | DELETE (Empty/Truncated)\033[0m")
                files_to_delete.append(f_path)
                continue

            # Unpack header
            with open(f_path, "rb") as f:
                header_bytes = f.read(HEADER_SIZE)
                data = struct.unpack(HEADER_FMT, header_bytes)
                header_beta = data[2]
                n_sweeps    = data[4]
                stride      = data[5]

            # Check 2: filename vs header beta
            if filename_beta is not None:
                is_name_rounded = abs(filename_beta - header_beta) > BETA_TOLERANCE
            else:
                is_name_rounded = False

            # Check 3: expected size from header vs actual
            num_records   = n_sweeps // stride
            expected_size = HEADER_SIZE + (num_records * RECORD_SIZE)
            ratio         = file_size / expected_size
            is_oversized  = ratio > SIZE_TOLERANCE_RATIO

            # Classify
            status     = "OK"
            action     = "KEEP"
            color_code = "\033[92m"  # green

            if is_oversized:
                status     = "COLLISION"
                action     = f"DELETE (Size mismatch: {ratio:.1f}x exp)"
                color_code = "\033[91m"  # red
                files_to_delete.append(f_path)

            elif is_name_rounded:
                status     = "BUG_NAME"
                action     = "DELETE (Naming Ambiguity)"
                color_code = "\033[93m"  # yellow
                files_to_delete.append(f_path)

            print(f"{color_code}{f_name:<40} | {header_beta:.6f}     | {status:<10} | {action}\033[0m")

        except Exception as e:
            print(f"\033[91m{f_name:<40} | {'ERROR':<12} | CORRUPT    | DELETE ({e})\033[0m")
            files_to_delete.append(f_path)

    return files_to_delete


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 audit_production_files.py <L>")
        sys.exit(1)

    L_target = int(sys.argv[1])
    bad_files = audit_directory(L_target)

    if bad_files:
        print(f"\n[WARNING] Found {len(bad_files)} problematic files.")
        confirm = input("Do you want to PERMANENTLY delete them? (yes/no): ")
        if confirm.lower() == "yes":
            for f in bad_files:
                try:
                    os.remove(f)
                    print(f"Deleted: {os.path.basename(f)}")
                except OSError as e:
                    print(f"Error deleting {f}: {e}")
            print("Cleanup complete.")
        else:
            print("Action cancelled. No files were deleted.")
    else:
        print(f"\n[SUCCESS] All files for L={L_target} pass the integrity check.")