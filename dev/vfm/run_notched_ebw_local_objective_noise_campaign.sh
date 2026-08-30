#!/usr/bin/env bash
set -euo pipefail

EBW_DATASET="${EBW_DATASET:-/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm}"
EBW_CAMPAIGN="${EBW_CAMPAIGN:-local_objective_noise_20260829}"
EBW_ROOT="$EBW_DATASET/identification/prepared/$EBW_CAMPAIGN"
EBW_ARCHIVE="${EBW_ARCHIVE:-/tmp/notched-ebw-local-objective-noise-results.tar.gz}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export UV_CACHE_DIR=/tmp/pyvale-uv-cache
export MPLCONFIGDIR=/tmp/pyvale-matplotlib

uv run --no-sync python dev/vfm/run_notched_ebw_objective_noise_campaign.py \
    --dataset "$EBW_DATASET" \
    --campaign-name "$EBW_CAMPAIGN" \
    --jobs 4 \
    --parallel-workers 4 \
    --max-basis-functions 7 \
    --max-iterations 180 \
    --max-evaluations 16100 \
    --noise-scale 1.0 \
    --noise-seed 20260828 \
    --seed 0 \
    --stress-backend cython

uv run --no-sync python dev/vfm/analyse_notched_ebw_gate_campaign.py \
    --campaign-root "$EBW_ROOT" \
    --dataset "$EBW_DATASET"

uv run --no-sync python dev/vfm/summarise_notched_ebw_objective_noise_campaign.py \
    --campaign-root "$EBW_ROOT"

tar -czf "$EBW_ARCHIVE" -C "$(dirname "$EBW_ROOT")" "$(basename "$EBW_ROOT")"

echo "CAMPAIGN COMPLETE"
echo "Report: $EBW_ROOT/analysis/OBJECTIVE_NOISE_REPORT.md"
echo "Archive: $EBW_ARCHIVE"
