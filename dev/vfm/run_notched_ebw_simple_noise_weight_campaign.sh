#!/usr/bin/env bash
set -euo pipefail

# Targeted robustness campaign: three representative objective balances across
# three realistic artificial-noise realisations, with bounded concurrency.

REPO_ROOT="${REPO_ROOT:-/home/robh/1_Projects/pyvale}"
DATASET="${DATASET:-/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm}"
CAMPAIGN_TAG="${CAMPAIGN_TAG:-simple_noise_weight_7bf_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_ROOT="${REPO_ROOT}/dev/vfm/output/${CAMPAIGN_TAG}"
BASE_CONFIG="${REPO_ROOT}/dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json"
ARTIFICIAL_NOISE_MODEL="${ARTIFICIAL_NOISE_MODEL:-${REPO_ROOT}/dev/vfm/data/wdbn1_noise_model_20260828.yaml}"
ARTIFICIAL_NOISE_SCALE="${ARTIFICIAL_NOISE_SCALE:-1}"
MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS:-7}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-15500}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
RANDOM_SEED="${RANDOM_SEED:-0}"
JOBS="${JOBS:-8}"

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
[[ -f "${BASE_CONFIG}" ]] || { echo "Missing base config: ${BASE_CONFIG}" >&2; exit 2; }
[[ -f "${ARTIFICIAL_NOISE_MODEL}" ]] || { echo "Missing noise model: ${ARTIFICIAL_NOISE_MODEL}" >&2; exit 2; }
[[ ! -e "${CAMPAIGN_ROOT}" ]] || { echo "Refusing to overwrite ${CAMPAIGN_ROOT}" >&2; exit 1; }

mkdir -p "${CAMPAIGN_ROOT}/configs"
STATUS_FILE="${CAMPAIGN_ROOT}/campaign_status.tsv"
printf 'case\tstatus\texit_code\n' > "${STATUS_FILE}"

declare -a WEIGHTS=(
  "baseline 0.75 0.15 0.10"
  "local_egi 0.80 0.10 0.10"
  "guards_dominant 0.40 0.30 0.30"
)
declare -a NOISE_SEEDS=(20260829 20260830 20260831)

cd "${REPO_ROOT}"
cases=()
for weight_entry in "${WEIGHTS[@]}"; do
    read -r weight_name informative fre broad <<< "${weight_entry}"
    config="${CAMPAIGN_ROOT}/configs/${weight_name}.json"
    CONFIG_PATH="${config}" BASE_CONFIG_PATH="${BASE_CONFIG}" \
      INFORMATIVE_WEIGHT="${informative}" FRE_WEIGHT="${fre}" BROAD_WEIGHT="${broad}" \
      .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["BASE_CONFIG_PATH"]).read_text())
payload["force_weight"] = float(os.environ["FRE_WEIGHT"])
payload["broad_guard_weight"] = float(os.environ["BROAD_WEIGHT"])
payload["informative_egi_weight"] = float(os.environ["INFORMATIVE_WEIGHT"])
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
    for noise_seed in "${NOISE_SEEDS[@]}"; do
        cases+=("${weight_name} ${noise_seed}")
    done
done

run_case() {
    local entry="$1" weight_name noise_seed name config run_tag status
    read -r weight_name noise_seed <<< "${entry}"
    name="${weight_name}_noise_seed${noise_seed}"
    config="${CAMPAIGN_ROOT}/configs/${weight_name}.json"
    run_tag="${CAMPAIGN_TAG}_${name}"
    echo "Starting ${name}"
    if REPO_ROOT="${REPO_ROOT}" DATASET="${DATASET}" CONFIG="${config}" RUN_TAG="${run_tag}" \
      MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS}" MAX_ITERATIONS="${MAX_ITERATIONS}" \
      MAX_EVALUATIONS="${MAX_EVALUATIONS}" PARALLEL_WORKERS="${PARALLEL_WORKERS}" \
      RANDOM_SEED="${RANDOM_SEED}" ARTIFICIAL_NOISE_MODEL="${ARTIFICIAL_NOISE_MODEL}" \
      ARTIFICIAL_NOISE_SCALE="${ARTIFICIAL_NOISE_SCALE}" ARTIFICIAL_NOISE_SEED="${noise_seed}" \
      bash dev/vfm/run_notched_ebw_simple_bf1_and_report.sh; then
        status=0
        printf '%s\tcomplete\t0\n' "${name}" >> "${STATUS_FILE}"
    else
        status=$?
        printf '%s\tfailed\t%s\n' "${name}" "${status}" >> "${STATUS_FILE}"
    fi
    return "${status}"
}

declare -A active_cases=()
failures=0
reap_one() {
    local finished_pid="" exit_code=0 case_name
    if wait -n -p finished_pid; then exit_code=0; else exit_code=$?; fi
    case_name="${active_cases[${finished_pid}]:-unknown}"
    unset "active_cases[${finished_pid}]"
    if (( exit_code == 0 )); then
        echo "Case complete: ${case_name}"
    else
        echo "Case failed: ${case_name} (exit ${exit_code})" >&2
        failures=$((failures + 1))
    fi
}

for entry in "${cases[@]}"; do
    while (( ${#active_cases[@]} >= JOBS )); do reap_one; done
    read -r weight_name noise_seed <<< "${entry}"
    run_case "${entry}" &
    active_cases[$!]="${weight_name}_noise_seed${noise_seed}"
done
while (( ${#active_cases[@]} > 0 )); do reap_one; done

echo "Campaign finished: ${CAMPAIGN_ROOT}; failures=${failures}"
(( failures == 0 ))
