#!/usr/bin/env bash
set -euo pipefail

: "${EBW_DATASET:?Set EBW_DATASET to the synthetic pyvale-vfm dataset}"
: "${EBW_CAMPAIGN_ROOT:?Set EBW_CAMPAIGN_ROOT to the completed gate campaign}"
: "${EBW_OUTPUT:?Set EBW_OUTPUT to the new campaign output directory}"

EBW_JOBS="${EBW_JOBS:-32}"
EBW_STATE_COUNT="${EBW_STATE_COUNT:-32}"
EBW_REPLICATES="${EBW_REPLICATES:-128}"
EBW_WINDOWS="${EBW_WINDOWS:-7,15,29,57}"
EBW_NOISE_SCALES="${EBW_NOISE_SCALES:-0,0.5,1,1.5}"
EBW_NOISE_MODEL="${EBW_NOISE_MODEL:-dev/vfm/data/wdbn1_noise_model_20260828.yaml}"
EBW_ARCHIVE="${EBW_ARCHIVE:-/tmp/notched-ebw-native-projection-noise-results.tar.gz}"

mkdir -p "$EBW_OUTPUT/logs"

run_state() {
    local state_index="$1"
    local state_output state_log
    state_output=$(printf "%s/state_%02d" "$EBW_OUTPUT" "$state_index")
    state_log=$(printf "%s/logs/state_%02d.log" "$EBW_OUTPUT" "$state_index")
    MPLBACKEND=Agg \
    QT_QPA_PLATFORM=offscreen \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    UV_CACHE_DIR=/tmp/pyvale-uv-cache \
    MPLCONFIGDIR=/tmp/pyvale-matplotlib \
    uv run --no-sync python \
        dev/vfm/run_notched_ebw_native_projection_noise.py \
        --dataset "$EBW_DATASET" \
        --campaign-root "$EBW_CAMPAIGN_ROOT" \
        --noise-model "$EBW_NOISE_MODEL" \
        --output "$state_output" \
        --windows "$EBW_WINDOWS" \
        --noise-scales "$EBW_NOISE_SCALES" \
        --noise-replicates "$EBW_REPLICATES" \
        --state-index "$state_index" \
        --skip-analysis \
        --resume >"$state_log" 2>&1
}
export -f run_state
export EBW_DATASET EBW_CAMPAIGN_ROOT EBW_OUTPUT EBW_REPLICATES
export EBW_WINDOWS EBW_NOISE_SCALES EBW_NOISE_MODEL

echo "Starting $EBW_STATE_COUNT states with $EBW_JOBS concurrent processes."
seq 0 $((EBW_STATE_COUNT - 1)) | xargs -P "$EBW_JOBS" -n 1 bash -c 'run_state "$1"' _ &
campaign_pid=$!
while kill -0 "$campaign_pid" 2>/dev/null; do
    completed=$(find "$EBW_OUTPUT" -path '*/projection_noise_rows.jsonl' -type f | wc -l)
    echo "$(date '+%F %T') progress: $completed/$EBW_STATE_COUNT state checkpoints present"
    sleep 60
done
wait "$campaign_pid"

MPLBACKEND=Agg \
QT_QPA_PLATFORM=offscreen \
UV_CACHE_DIR=/tmp/pyvale-uv-cache \
MPLCONFIGDIR=/tmp/pyvale-matplotlib \
uv run --no-sync python \
    dev/vfm/analyse_notched_ebw_native_projection_noise.py \
    --campaign-root "$EBW_OUTPUT" \
    --noise-model "$EBW_NOISE_MODEL" \
    --windows "$EBW_WINDOWS" \
    --noise-scales "$EBW_NOISE_SCALES" \
    --noise-replicates "$EBW_REPLICATES" \
    --expected-states "$EBW_STATE_COUNT"

tar -czf "$EBW_ARCHIVE" -C "$(dirname "$EBW_OUTPUT")" "$(basename "$EBW_OUTPUT")"
echo "CAMPAIGN COMPLETE"
echo "Report: $EBW_OUTPUT/NOTCHED_EBW_NATIVE_PROJECTION_NOISE.pdf"
echo "Archive: $EBW_ARCHIVE"
