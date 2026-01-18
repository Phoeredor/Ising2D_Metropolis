#!/usr/bin/env python3
"""
split_plan_local_remote.py
Ripartizione Aggiornata:
- SERVER: 10, 16, 24, 32
- LOCALE: 48, 64
Input: results/production/production_plan_ultra.dat
"""
from pathlib import Path

# CONFIGURAZIONE AGGIORNATA
SERVER_L = [10, 16, 24, 32]
LOCAL_L  = [48, 64]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_FILE = PROJECT_ROOT / "results" / "production" / "production_plan_ultra.dat"
OUT_LOCAL = PROJECT_ROOT / "results" / "production" / "plan_LOCAL.dat"
OUT_SERVER = PROJECT_ROOT / "results" / "production" / "plan_SERVER.dat"

def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] Missing {INPUT_FILE}")
        print("Run the generator first: python3 scripts/production/generate_plan_fss.py")
        return

    print(f"--- SPLITTING PLAN (Fixed) ---")
    print(f"Server L: {SERVER_L}")
    print(f"Local L : {LOCAL_L}")

    header = ""
    lines_local = []
    lines_server = []

    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.startswith("#"):
                header = line
                continue
            if not line.strip(): continue
            
            try:
                L = int(line.split()[0])
            except ValueError:
                continue
            
            if L in LOCAL_L:
                lines_local.append(line)
            elif L in SERVER_L:
                lines_server.append(line)

    with open(OUT_LOCAL, "w") as f:
        f.write(header)
        f.writelines(lines_local)
    print(f"[OK] plan_LOCAL.dat: {len(lines_local)} jobs (L={LOCAL_L})")

    with open(OUT_SERVER, "w") as f:
        f.write(header)
        f.writelines(lines_server)
    print(f"[OK] plan_SERVER.dat: {len(lines_server)} jobs (L={SERVER_L})")

if __name__ == "__main__":
    main()
