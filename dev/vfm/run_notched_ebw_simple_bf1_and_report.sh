#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/robh/1_Projects/pyvale}"
DATASET="${DATASET:-/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm}"
RUN_TAG="${RUN_TAG:-simple_sensitivity_gated_bf1_$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="prepared/${RUN_TAG}"
RUN_DIR="${DATASET}/identification/${RUN_NAME}"
REPORT_DIR="${REPO_ROOT}/dev/vfm/output/${RUN_TAG}"
CONFIG="${CONFIG:-${REPO_ROOT}/dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-5000}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS:-1}"
RANDOM_SEED="${RANDOM_SEED:-0}"
ARTIFICIAL_NOISE_MODEL="${ARTIFICIAL_NOISE_MODEL:-}"
ARTIFICIAL_NOISE_SCALE="${ARTIFICIAL_NOISE_SCALE:-0}"
ARTIFICIAL_NOISE_SEED="${ARTIFICIAL_NOISE_SEED:-20260828}"

if [[ -e "${RUN_DIR}" || -e "${REPORT_DIR}" ]]; then
    echo "Refusing to overwrite an existing run/report: ${RUN_TAG}" >&2
    exit 1
fi

mkdir -p "${REPORT_DIR}"
cd "${REPO_ROOT}"

NOISE_ARGS=()
if [[ "${ARTIFICIAL_NOISE_SCALE}" != "0" ]]; then
    [[ -n "${ARTIFICIAL_NOISE_MODEL}" && -f "${ARTIFICIAL_NOISE_MODEL}" ]] || {
        echo "A valid ARTIFICIAL_NOISE_MODEL is required for non-zero noise" >&2
        exit 2
    }
    NOISE_ARGS+=(
        --artificial-noise-model "${ARTIFICIAL_NOISE_MODEL}"
        --artificial-noise-scale "${ARTIFICIAL_NOISE_SCALE}"
        --artificial-noise-seed "${ARTIFICIAL_NOISE_SEED}"
    )
fi

uv run --no-sync python dev/vfm/call_notched_ebw_bivariate_identification.py \
    --input "${DATASET}/prepared" \
    --output-root "${DATASET}/identification" \
    --run-name "${RUN_NAME}" \
    --simple-data-driven-objective-config "${CONFIG}" \
    --kernel-type bivariate_spd \
    --basis-growth-policy sensitivity_correction \
    --fixed-basis-trajectory \
    --minimum-objective-improvement 0 \
    --max-basis-functions "${MAX_BASIS_FUNCTIONS}" \
    --phase-0-max-evaluations 50 \
    --max-iterations "${MAX_ITERATIONS}" \
    --max-evaluations "${MAX_EVALUATIONS}" \
    --parallel-workers "${PARALLEL_WORKERS}" \
    --random-seed "${RANDOM_SEED}" \
    --stress-backend cython \
    "${NOISE_ARGS[@]}" \
    2>&1 | tee "${REPORT_DIR}/identification.log"

uv run --no-sync python dev/vfm/report_notched_ebw_simple_identification.py \
    --input "${DATASET}/prepared" \
    --run "${RUN_DIR}" \
    --output "${REPORT_DIR}/SIMPLE_SENSITIVITY_GATED_IDENTIFICATION.pdf"

uv run --no-sync python dev/vfm/report_simple_gate_weight_calibration.py \
    --artifact-dir "${RUN_DIR}/diagnostic_artifacts" \
    --historical-output-root "${REPO_ROOT}/dev/vfm/output" \
    --output "${REPORT_DIR}/SIMPLE_GATE_WEIGHT_CALIBRATION.pdf"

echo "Run: ${RUN_DIR}"
echo "Identification report: ${REPORT_DIR}/SIMPLE_SENSITIVITY_GATED_IDENTIFICATION.pdf"
echo "Calibration report: ${REPORT_DIR}/SIMPLE_GATE_WEIGHT_CALIBRATION.pdf"
