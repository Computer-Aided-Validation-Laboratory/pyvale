#!/usr/bin/env bash
set -euo pipefail

# Two-hour decision campaign: complete BF7 native-projection states only.
# BF5/BF6 checkpoints are restored from the transferred workstation archive.

PYVALE_ROOT="${PYVALE_ROOT:-$(pwd)}"
DATASET="${EBW_DATASET:-/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm}"
CAMPAIGN_ROOT="${EBW_CAMPAIGN_ROOT:-$DATASET/identification/prepared/gate_objective_campaign_20260828}"
PARTIAL_ARCHIVE="${EBW_PARTIAL_ARCHIVE:-/tmp/notched-ebw-native-projection-noise-partial-20260829.tar.gz}"
WORK_ROOT="${EBW_WORK_ROOT:-/tmp/notched-ebw-two-hour-followup-20260829}"
OUTPUT="$WORK_ROOT/native_projection_noise_20260828"
NOISE_MODEL="$PYVALE_ROOT/dev/vfm/data/wdbn1_noise_model_20260828.yaml"
FINAL_ARCHIVE="${EBW_FINAL_ARCHIVE:-/tmp/notched-ebw-native-projection-bf5-7-20260829.tar.gz}"
JOBS="${EBW_JOBS:-4}"

mkdir -p "$WORK_ROOT"
if [[ ! -d "$OUTPUT" ]]; then
    tar -xzf "$PARTIAL_ARCHIVE" -C "$WORK_ROOT"
fi
mkdir -p "$OUTPUT/logs"

# State ordering is seed-major BF5, BF6, BF7, BF8.  These are BF7 only.
BF7_STATES=(2 6 10 14 18 22 26 30)

run_state() {
    local state_index="$1"
    local state_output state_log
    state_output=$(printf "%s/state_%02d" "$OUTPUT" "$state_index")
    state_log=$(printf "%s/logs/state_%02d.log" "$OUTPUT" "$state_index")
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UV_CACHE_DIR=/tmp/pyvale-uv-cache \
    MPLCONFIGDIR=/tmp/pyvale-matplotlib \
    uv run --no-sync python \
        "$PYVALE_ROOT/dev/vfm/run_notched_ebw_native_projection_noise.py" \
        --dataset "$DATASET" \
        --campaign-root "$CAMPAIGN_ROOT" \
        --noise-model "$NOISE_MODEL" \
        --output "$state_output" \
        --windows 7,15,29,57 \
        --noise-scales 0,0.5,1,1.5 \
        --noise-replicates 128 \
        --state-index "$state_index" \
        --skip-analysis \
        --resume >"$state_log" 2>&1
}
export -f run_state
export PYVALE_ROOT DATASET CAMPAIGN_ROOT OUTPUT NOISE_MODEL

printf '%s\n' "${BF7_STATES[@]}" | xargs -P "$JOBS" -n 1 bash -c 'run_state "$1"' _

MPLBACKEND=Agg \
QT_QPA_PLATFORM=offscreen \
UV_CACHE_DIR=/tmp/pyvale-uv-cache \
MPLCONFIGDIR=/tmp/pyvale-matplotlib \
uv run --no-sync python \
    "$PYVALE_ROOT/dev/vfm/analyse_notched_ebw_native_projection_noise.py" \
    --campaign-root "$OUTPUT" \
    --noise-model "$NOISE_MODEL" \
    --windows 7,15,29,57 \
    --noise-scales 0,0.5,1,1.5 \
    --noise-replicates 128 \
    --expected-states 24

tar -czf "$FINAL_ARCHIVE" -C "$WORK_ROOT" "$(basename "$OUTPUT")"
echo "BF5-7 CAMPAIGN COMPLETE"
echo "Report: $OUTPUT/NOTCHED_EBW_NATIVE_PROJECTION_NOISE.pdf"
echo "Archive: $FINAL_ARCHIVE"
