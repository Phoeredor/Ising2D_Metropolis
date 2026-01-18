#!/bin/bash
# =============================================================================
# RUN PILOT FSS (Range 0.35 - 0.50)
# Usage: ./run_pilot.sh [local|server]
# =============================================================================

set -u

# --- CONFIGURATION ---
MODE=${1:-local}
L_LIST="10 16 24 32 48 64 96 128"

# PILOT SETTINGS
# N_THERM=0 to observe approach to equilibrium in data file
N_THERM=0
N_SWEEPS=1000000
STRIDE=1

if [ "$MODE" == "server" ]; then
    MAX_JOBS=6
    NICE_CMD="nice -n 19"
else
    if command -v nproc > /dev/null; then
        MAX_JOBS=$(($(nproc) - 1))
        [ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
    else
        MAX_JOBS=4
    fi
    NICE_CMD=""
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
EXE_PILOT="${PROJECT_ROOT}/pilot_ising"
GEN_SCRIPT="${PROJECT_ROOT}/scripts/pilot/setup/gen_pilot_grid.py"

if [ ! -f "$EXE_PILOT" ]; then
    echo "[ERR] Executable missing. Run 'make clean && make'"
    exit 1
fi

echo "--- Starting FSS Pilot Runs (N_THERM=$N_THERM) ---"

for L in $L_LIST; do
    BETAS=$(python3 "$GEN_SCRIPT" "$L")
    
    OUT_DIR="${PROJECT_ROOT}/results/pilot/L${L}/bin"
    mkdir -p "$OUT_DIR"
    
    echo "=== Pilot L=$L ==="

    for beta in $BETAS; do
        # Format filename for check (6 decimals as in C code)
        beta_fmt=$(printf "%.6f" "$beta")
        FILE_CHECK="${OUT_DIR}/rng_obs_L${L}_beta${beta_fmt}.bin"

        if [ -s "$FILE_CHECK" ]; then
            continue
        fi

        while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 0.5
        done

        echo "  [RUN] L=$L beta=$beta (RNG+Up)"
        
        # RNG start
        (
            cd "$OUT_DIR" || exit 1
            $NICE_CMD "$EXE_PILOT" "$L" "$beta" "$N_THERM" "$N_SWEEPS" "$STRIDE" "rng" > "rng_obs_L${L}_beta${beta_fmt}.bin"
        ) &
        
        while [ "$(jobs -p | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 0.5
        done

        # UP start
        (
            cd "$OUT_DIR" || exit 1
            $NICE_CMD "$EXE_PILOT" "$L" "$beta" "$N_THERM" "$N_SWEEPS" "$STRIDE" "up" > "up_obs_L${L}_beta${beta_fmt}.bin"
        ) &
        
        sleep 0.05
    done
    wait
done

echo "All pilot runs completed."
