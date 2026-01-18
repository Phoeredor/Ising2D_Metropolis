#!/usr/bin/env python3
"""
summarize_pilots.py

Aggregates individual analysis reports (L10, L16...) into a single master JSON.
This master file is the input for the Production Plan Generator.

Output: results/pilot/prod_params_from_pilots.json
"""

import json
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_RESULTS_DIR = PROJECT_ROOT / "results" / "pilot"
OUTPUT_FILE = PILOT_RESULTS_DIR / "prod_params_from_pilots.json"

def main():
    print("--- SUMMARIZING PILOT RESULTS ---")
    
    # Find all L directories
    report_files = sorted(glob.glob(str(PILOT_RESULTS_DIR / "L*" / "analysis_report_L*.json")))
    
    if not report_files:
        print("[WARN] No analysis reports found. Run 'analyze_pilots.py' first.")
        return

    aggregated_points = []
    
    for r_path in report_files:
        try:
            with open(r_path, "r") as f:
                data = json.load(f)
                points = data.get("points", [])
                aggregated_points.extend(points)
                print(f"Loaded {len(points)} points from {Path(r_path).name}")
        except Exception as e:
            print(f"[ERR] Failed reading {r_path}: {e}")

    # Save Aggregated Data
    final_db = {"points": aggregated_points}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_db, f, indent=2)
        
    print("-" * 40)
    print(f"[OK] Master DB saved: {OUTPUT_FILE}")
    print(f"     Total Pilot Points: {len(aggregated_points)}")

if __name__ == "__main__":
    main()
