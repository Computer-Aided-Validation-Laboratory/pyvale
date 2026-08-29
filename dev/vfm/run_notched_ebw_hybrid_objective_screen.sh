#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: $0 DATASET ROUND1_OUTPUT [CAMPAIGN_ROOT]}"
ROUND1_OUTPUT="${2:?usage: $0 DATASET ROUND1_OUTPUT [CAMPAIGN_ROOT]}"
CAMPAIGN_ROOT="${3:-${DATASET}/identification/prepared/gate_objective_campaign_20260828}"
PYVALE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NOISE_MODEL="${PYVALE_ROOT}/dev/vfm/data/wdbn1_noise_model_20260828.yaml"

mkdir -p "${ROUND1_OUTPUT}/direct_fit" "${ROUND1_OUTPUT}/screen" "${ROUND1_OUTPUT}/logs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pyvale-matplotlib}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/pyvale-uv-cache}"

cd "${PYVALE_ROOT}"

DIRECT_PID=""; SCREEN_PID=""
cleanup() {
  [[ -n "${DIRECT_PID}" ]] && kill "${DIRECT_PID}" 2>/dev/null || true
  [[ -n "${SCREEN_PID}" ]] && kill "${SCREEN_PID}" 2>/dev/null || true
}
trap cleanup INT TERM

uv run --no-sync python dev/vfm/build_notched_ebw_direct_fit_reference.py \
  --input "${DATASET}/prepared" \
  --output "${ROUND1_OUTPUT}/direct_fit" \
  --max-bases 8 --starts 5 --resume \
  >"${ROUND1_OUTPUT}/logs/direct_fit.log" 2>&1 &
DIRECT_PID=$!

uv run --no-sync python dev/vfm/screen_notched_ebw_hybrid_objective.py \
  --dataset "${DATASET}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --noise-model "${NOISE_MODEL}" \
  --output "${ROUND1_OUTPUT}/screen" \
  --noise-replicates 8 --resume \
  >"${ROUND1_OUTPUT}/logs/screen.log" 2>&1 &
SCREEN_PID=$!

STARTED=$SECONDS
while kill -0 "${DIRECT_PID}" 2>/dev/null || kill -0 "${SCREEN_PID}" 2>/dev/null; do
  DIRECT_STATE="finished"; SCREEN_STATE="finished"
  kill -0 "${DIRECT_PID}" 2>/dev/null && DIRECT_STATE="running"
  kill -0 "${SCREEN_PID}" 2>/dev/null && SCREEN_STATE="running"
  echo "round1 heartbeat elapsed=$((SECONDS-STARTED))s direct_fit=${DIRECT_STATE} screen=${SCREEN_STATE}"
  tail -n 1 "${ROUND1_OUTPUT}/logs/direct_fit.log" 2>/dev/null || true
  tail -n 1 "${ROUND1_OUTPUT}/logs/screen.log" 2>/dev/null || true
  sleep 60
done

DIRECT_STATUS=0; SCREEN_STATUS=0
wait "${DIRECT_PID}" || DIRECT_STATUS=$?
wait "${SCREEN_PID}" || SCREEN_STATUS=$?
if [[ ${DIRECT_STATUS} -ne 0 || ${SCREEN_STATUS} -ne 0 ]]; then
  echo "round1 failed direct_fit_status=${DIRECT_STATUS} screen_status=${SCREEN_STATUS}" >&2
  echo "logs=${ROUND1_OUTPUT}/logs" >&2
  exit 1
fi

uv run --no-sync python dev/vfm/report_notched_ebw_hybrid_objective_screen.py \
  --round1 "${ROUND1_OUTPUT}"
echo "round1 complete output=${ROUND1_OUTPUT}"
