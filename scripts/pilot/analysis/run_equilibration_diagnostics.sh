#!/bin/bash
# =============================================================================
# SCRIPT: run_equilibration_diagnostics.sh
# PURPOSE: Generate time-series data for equilibration analysis (L=128).
# METHOD:  Parallel execution with safe SANDBOXING to prevent file collisions.
# TARGETS: T=2.5, 2.4, 2.3 and T_critical (for Marinari comparison).
# OUTPUT:  Directly into results/pilot/MC_convergence/
# =============================================================================

set -u

# --- CONFIGURATION ---
PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
EXE_PILOT="${PROJECT_ROOT}/pilot_ising"
FINAL_OUT_DIR="${PROJECT_ROOT}/results/pilot/MC_convergence"

# Simulation Parameters
L=128                # Fixed to 128 as requested
N_THERM=0            
N_SWEEPS=5000000     
STRIDE=1             

BETA_C_EXACT="0.44068679350977151262"

if [ ! -f "$EXE_PILOT" ]; then
    echo "[ERR] Executable missing: $EXE_PILOT"
    exit 1
fi

# Prepare Output Directory (Wipe and Recreate)
if [ -d "$FINAL_OUT_DIR" ]; then
    echo "[INFO] Cleaning old convergence directory..."
    rm -rf "$FINAL_OUT_DIR"
fi
mkdir -p "$FINAL_OUT_DIR"

echo "=================================================================="
echo "   STARTING DIAGNOSTICS (DIRECT OUTPUT MODE)"
echo "   Lattice: $L"
echo "   Target:  $FINAL_OUT_DIR"
echo "=================================================================="

get_beta_from_x() {
    local x_val=$1
    python3 -c "print(f'{${BETA_C_EXACT} + ($x_val / $L):.18f}')"
}

# --- SANDBOXED RUN FUNCTION ---
run_job() {
    local x_val=$1
    local init_type=$2     
    local final_name=$3
    
    local beta_high_prec=$(get_beta_from_x "$x_val")
    local beta_short=$(printf "%.6f" "$beta_high_prec")

    # Map generic types to C-arguments
    local c_arg=""
    local msg=""
    case "$init_type" in
        "rng")  c_arg="rng";  msg="[RANDOM CONFIG]"; ;;
        "up")   c_arg="up";   msg="[UP CONFIG]";     ;;
        "down") c_arg="up";   msg="[DOWN CONFIG]";   ;;
    esac

    echo "LAUNCHING: x=$x_val | Beta~$beta_short | $msg"

    # --- EXECUTION IN BACKGROUND SUB-SHELL ---
    (
        # 1. Create a unique sandbox directory INSIDE the final output dir
        #    This ensures no filename collision between parallel jobs
        local sandbox="${FINAL_OUT_DIR}/temp_${init_type}_x${x_val}"
        mkdir -p "$sandbox"
        cd "$sandbox" || exit

        # 2. Run Pilot 
        "$EXE_PILOT" "$L" "$beta_high_prec" "$N_THERM" "$N_SWEEPS" "$STRIDE" "$c_arg" > /dev/null
        
        # 3. Find generated file
        local pattern="*beta${beta_short}*.bin"
        local gen_file=$(find . -name "$pattern" | head -n 1)

        if [ -n "$gen_file" ] && [ -f "$gen_file" ]; then
            # 4. Move to Parent Directory (The Final Destination)
            mv "$gen_file" "../${final_name}"
            echo " [DONE] $final_name"
            
            # 5. Cleanup Sandbox
            cd ..
            rm -rf "$sandbox"
        else
            echo " [FAIL] Could not generate $final_name"
        fi
    ) & 
    
    sleep 1 # Stagger start times for RNG seeding safety
}

# =============================================================================
# 1. ERGODICITY CHECK (High Temperature T ~ 2.5)
# x calculated for L=128 to match Beta ~ 0.400
# =============================================================================
X_HIGH="-5.21" 
BETA_HIGH_STR=$(printf "%.3f" $(get_beta_from_x "$X_HIGH"))

run_job "$X_HIGH" "rng"  "rng_obs_L${L}_beta${BETA_HIGH_STR}.bin"
run_job "$X_HIGH" "up"   "up_obs_L${L}_beta${BETA_HIGH_STR}.bin"
run_job "$X_HIGH" "down" "down_obs_L${L}_beta${BETA_HIGH_STR}.bin"

# =============================================================================
# 2. CRITICAL SLOWING DOWN (T ~ 2.4, 2.3, and Critical)
# =============================================================================

# T ~ 2.4 (Beta ~ 0.417)
X_MID1="-3.07"
BETA_MID1_STR=$(printf "%.3f" $(get_beta_from_x "$X_MID1"))
run_job "$X_MID1" "up" "up_obs_L${L}_beta${BETA_MID1_STR}.bin"

# T ~ 2.3 (Beta ~ 0.435)
X_MID2="-0.76"
BETA_MID2_STR=$(printf "%.3f" $(get_beta_from_x "$X_MID2"))
run_job "$X_MID2" "up" "up_obs_L${L}_beta${BETA_MID2_STR}.bin"

# Critical Point (T = Tc)
X_CRIT="0.0"
BETA_CRIT_STR=$(printf "%.3f" $(get_beta_from_x "$X_CRIT"))
run_job "$X_CRIT" "up" "up_obs_L${L}_beta${BETA_CRIT_STR}_CRIT.bin"

# Wait for completion
wait
echo "---------------------------------------------------"
echo "ALL JOBS COMPLETED."
ls -lh "$FINAL_OUT_DIR"
