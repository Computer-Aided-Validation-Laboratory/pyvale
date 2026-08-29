#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: $0 DATASET RESCORE_OUTPUT [CAMPAIGN_ROOT] [DIRECT_FIT_SOURCE]}"
RESCORE_OUTPUT="${2:?usage: $0 DATASET RESCORE_OUTPUT [CAMPAIGN_ROOT] [DIRECT_FIT_SOURCE]}"
CAMPAIGN_ROOT="${3:-${DATASET}/identification/prepared/gate_objective_campaign_20260828}"
DIRECT_FIT_SOURCE="${4:-${DATASET}/identification/prepared/hybrid_objective_round1_20260829/direct_fit}"
PYVALE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NOISE_MODEL="${PYVALE_ROOT}/dev/vfm/data/wdbn1_noise_model_20260828.yaml"

mkdir -p "${RESCORE_OUTPUT}/screen" "${RESCORE_OUTPUT}/logs"
if [[ -d "${DIRECT_FIT_SOURCE}" && ! -d "${RESCORE_OUTPUT}/direct_fit" ]]; then
  cp -a "${DIRECT_FIT_SOURCE}" "${RESCORE_OUTPUT}/direct_fit"
  echo "reused direct-fit reference source=${DIRECT_FIT_SOURCE}"
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pyvale-matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/pyvale-uv-cache}"
cd "${PYVALE_ROOT}"

uv run --no-sync python dev/vfm/screen_notched_ebw_hybrid_objective.py \
  --dataset "${DATASET}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --noise-model "${NOISE_MODEL}" \
  --output "${RESCORE_OUTPUT}/screen" \
  --regime-mode relative \
  --onset-fraction 0.05 \
  --developed-fraction 0.50 \
  --late-fraction 0.80 \
  --minimum-regime-frames 2 \
  --noise-replicates 8 \
  --resume \
  2>&1 | tee "${RESCORE_OUTPUT}/logs/rescore.log"

uv run --no-sync python dev/vfm/report_notched_ebw_hybrid_objective_screen.py \
  --round1 "${RESCORE_OUTPUT}"

uv run --no-sync python dev/vfm/build_notched_ebw_raw_pilot_manifest.py \
  --screen "${RESCORE_OUTPUT}/screen" \
  --noise-model "${NOISE_MODEL}" \
  --output "${RESCORE_OUTPUT}/raw_pilot_manifest.json"

echo "relative-regime rescore complete output=${RESCORE_OUTPUT}"
