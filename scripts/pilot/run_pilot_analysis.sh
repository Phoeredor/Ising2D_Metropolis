#!/bin/bash
# =============================================================================
# PILOT ANALYSIS PIPELINE
# 1. Runs analysis for each L (Tau estimation, H/C check).
# 2. Summarizes all results into a single JSON for production planning.
# =============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
ANALYSIS_DIR="${PROJECT_ROOT}/scripts/pilot/analysis"
L_LIST="10 16 24 32 48 64 96 128"

echo "--- STARTING PILOT ANALYSIS ---"

# 1. Per-Lattice Analysis
for L in $L_LIST; do
    echo ">> Analyzing L=$L..."
    python3 "${ANALYSIS_DIR}/analyze_pilots.py" "$L"
done

# 2. Aggregation
echo ">> Aggregating results..."
python3 "${ANALYSIS_DIR}/summarize_pilots.py"

echo "--- PILOT ANALYSIS COMPLETE ---"
echo "Next step: python3 scripts/production/generate_plan.py"
