#!/bin/bash
# =============================================================================
# PRODUCTION RUNNER (Local/Server)
# Reads production_plan.dat and executes missing simulations.
# Features: Job Control, Nice level 19, Resume capability.
# =============================================================================

set -u

# --- CONFIGURATION ---
PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
EXE_PATH="${PROJECT_ROOT}/prod_ising"
PLAN_FILE="${PROJECT_ROOT}/results/production/production_plan.dat"
MAX_JOBS=6  # Max parallel jobs

# --- SANITY CHECKS ---
if [ ! -f "$EXE_PATH" ]; then
    echo "[ERR] Executable missing. Compile first: make clean && make"
    exit 1
fi

if [ ! -f "$PLAN_FILE" ]; then
    echo "[ERR] Plan missing: $PLAN_FILE"
    echo "      Run first: python3 scripts/production/generate_plan.py"
    exit 1
fi

echo "=== Starting Production ==="
echo "Plan: $PLAN_FILE"
echo "Jobs: $MAX_JOBS (Nice=19)"

# --- EXECUTION LOOP ---
while read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    # Parse line: L beta stride therm sweeps ...
    read -r L beta stride therm sweeps x_val regime <<< "$line"

    # Format beta to 6 decimals (matches C code logic)
    beta_fmt=$(printf "%.6f" "$beta")

    # Define paths
    OUT_DIR="${PROJECT_ROOT}/results/production/L${L}/bin"
    LOG_DIR="${PROJECT_ROOT}/results/production/L${L}/logs"
    LOG_FILE="${LOG_DIR}/prod_beta${beta_fmt}.log"
    mkdir -p "$OUT_DIR" "$LOG_DIR" "${OUT_DIR}/snapshots"

    # 1. Check Existence (Resume Capability)
    EXPECTED_BIN="${OUT_DIR}/rng_obs_L${L}_beta${beta_fmt}.bin"
    if [ -s "$EXPECTED_BIN" ]; then
        # echo "[SKIP] L=$L Beta=$beta_fmt (Completed)"
        continue
    fi

    # 2. Job Control (Wait if max jobs reached)
    while [ "$(pgrep -c -f "prod_ising")" -ge "$MAX_JOBS" ]; do
        sleep 2
    done

    echo "[RUN] L=$L Beta=$beta_fmt (X=$x_val)"

    # 3. Launch Simulation (Background)
    # Use subshell to cd safely. Use 'nice' for priority.
    (
        cd "$OUT_DIR" || exit 1
        nice -n 19 "$EXE_PATH" "$L" "$beta" "$therm" "$sweeps" "$stride" "rng" > "$LOG_FILE" 2>&1
    ) &

    # Small delay to prevent race conditions in process table
    sleep 0.2

done < "$PLAN_FILE"

echo "----------------------------------------"
echo "All jobs launched. Waiting for completion..."
wait
echo "Production Finished."
