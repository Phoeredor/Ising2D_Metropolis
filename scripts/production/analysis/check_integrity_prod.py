#!/usr/bin/env python3
"""
Binary integrity check for production MC data.

Scans every *.bin file under results/production/L*/bin/ and verifies:
  - File is at least HEADER_SIZE bytes (not truncated).
  - Header metadata is sane (L > 0, n_sweeps > 0, stride > 0).
  - Actual file size matches the expected size from header params.

Reports OK, INCOMPLETE, OVERSIZED, or CORRUPT per file.
"""

import os
import struct
import sys
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT    = PROJECT_ROOT / "results" / "production"

# Header layout (C struct, little-endian):
#   int32 L, int32 pad, double beta, int64 n_therm, int64 n_sweeps,
#   int64 stride, uint32 seed1, uint32 seed2
HEADER_FMT  = "ii d qqq II"
HEADER_SIZE = 48  # bytes

# Record: int64 sweep + 5 × double (e, m, e2, m2, m4)
RECORD_SIZE = 48  # bytes


# ============================================================================
# VALIDATION
# ============================================================================

def check_file(filepath: Path) -> str:
    """Return 'OK' or a diagnostic string for a single binary file."""
    if not filepath.exists():
        return "MISSING"

    try:
        file_size = filepath.stat().st_size

        if file_size < HEADER_SIZE:
            return f"TRUNCATED_HEADER (Size: {file_size} bytes)"

        with open(filepath, "rb") as f:
            header_bytes = f.read(HEADER_SIZE)
            L, _, beta, n_therm, n_sweeps, stride, _, _ = struct.unpack(
                HEADER_FMT, header_bytes)

            if L <= 0 or n_sweeps <= 0 or stride <= 0:
                return f"INVALID_METADATA (L={L}, N={n_sweeps}, Stride={stride})"

            expected_measurements = n_sweeps // stride
            expected_size = HEADER_SIZE + expected_measurements * RECORD_SIZE

            if file_size != expected_size:
                diff_bytes = expected_size - file_size
                missing_records = diff_bytes / RECORD_SIZE
                if diff_bytes > 0:
                    return f"INCOMPLETE (Missing ~{missing_records:.1f} records)"
                else:
                    return f"OVERSIZED (Extra bytes: {-diff_bytes})"

            return "OK"

    except Exception as e:
        return f"CORRUPT ({e})"


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root directory not found: {DATA_ROOT}")
        sys.exit(1)

    print(f"{'File':<45} {'Status':<15} {'Detail'}")
    print("-" * 80)

    total_files = 0
    bad_files = 0

    l_dirs = sorted(DATA_ROOT.glob("L*"))
    if not l_dirs:
        print(f"[WARNING] No L* directories found in {DATA_ROOT}")
        sys.exit(0)

    for l_dir in l_dirs:
        bin_dir = l_dir / "bin"
        if not bin_dir.exists():
            continue

        for filepath in sorted(bin_dir.glob("*.bin")):
            status = check_file(filepath)
            total_files += 1

            if status != "OK":
                status_colored = f"\033[91m{status.split(' ')[0]}\033[0m"
                bad_files += 1
                print(f"{filepath.name:<45} {status_colored:<15} {status}")

    print("-" * 80)
    print(f"Total Files Scanned: {total_files}")
    print(f"Corrupt/Incomplete:  {bad_files}")

    if bad_files == 0:
        print("\n\033[92mSUCCESS: All production data valid.\033[0m")
    else:
        print(f"\n\033[91mFAILURE: {bad_files} problematic file(s).\033[0m")


if __name__ == "__main__":
    main()
