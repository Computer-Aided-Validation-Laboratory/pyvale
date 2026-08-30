#!/usr/bin/env bash
set -euo pipefail

# Run the declared, simplex-valid weight cases with bounded process-level
# concurrency. Each identification retains its own internal worker pool,
# durable config snapshot, result directory, logs, and diagnostic PDFs.

REPO_ROOT="${REPO_ROOT:-/home/robh/1_Projects/pyvale}"
DATASET="${DATASET:-/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm}"
SWEEP_TAG="${SWEEP_TAG:-simple_weight_sweep_7bf_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${REPO_ROOT}/dev/vfm/output/${SWEEP_TAG}"
BASE_CONFIG="${REPO_ROOT}/dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json"
MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS:-7}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-15500}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
RANDOM_SEED="${RANDOM_SEED:-0}"
JOBS="${JOBS:-1}"
START_CASE_INDEX="${START_CASE_INDEX:-1}"

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be a positive integer" >&2; exit 2; }
[[ "${START_CASE_INDEX}" =~ ^([1-9]|10)$ ]] || { echo "START_CASE_INDEX must be 1..10" >&2; exit 2; }

if [[ -e "${SWEEP_ROOT}" ]]; then
    echo "Refusing to overwrite existing sweep output: ${SWEEP_ROOT}" >&2
    exit 1
fi
mkdir -p "${SWEEP_ROOT}/configs"
STATUS_FILE="${SWEEP_ROOT}/campaign_status.tsv"
printf 'case\tstatus\texit_code\n' > "${STATUS_FILE}"

declare -a CASES=(
  "01_baseline 0.75 0.15 0.10"
  "02_local_egi 0.80 0.10 0.10"
  "03_balanced_guards 0.60 0.20 0.20"
  "04_half_guards 0.50 0.25 0.25"
  "05_near_equal 0.34 0.33 0.33"
  "06_fre_emphasis 0.60 0.30 0.10"
  "07_broad_egi_emphasis 0.60 0.10 0.30"
  "08_strong_fre 0.40 0.40 0.20"
  "09_strong_broad_egi 0.40 0.20 0.40"
  "10_guards_dominant 0.40 0.30 0.30"
)

cd "${REPO_ROOT}"
selected_cases=()
for case_offset in "${!CASES[@]}"; do
    case_index=$((case_offset + 1))
    (( case_index >= START_CASE_INDEX )) || continue
    entry="${CASES[case_offset]}"
    read -r name informative fre broad <<< "${entry}"
    selected_cases+=("${entry}")
    config="${SWEEP_ROOT}/configs/${name}.json"
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
payload["notes"] = list(payload.get("notes", [])) + [
    "Sweep weights: informative EGI={}; FRE={}; broad EGI={}.".format(
        os.environ["INFORMATIVE_WEIGHT"], os.environ["FRE_WEIGHT"], os.environ["BROAD_WEIGHT"]
    )
]
Path(os.environ["CONFIG_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY
done

run_case() {
    local entry="$1" name informative fre broad config run_tag status
    read -r name informative fre broad <<< "${entry}"
    config="${SWEEP_ROOT}/configs/${name}.json"
    run_tag="${SWEEP_TAG}_${name}"
    echo "Starting ${name}: informative=${informative}, FRE=${fre}, broad=${broad}"
    if REPO_ROOT="${REPO_ROOT}" DATASET="${DATASET}" CONFIG="${config}" RUN_TAG="${run_tag}" \
      MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS}" \
      MAX_ITERATIONS="${MAX_ITERATIONS}" \
      MAX_EVALUATIONS="${MAX_EVALUATIONS}" \
      PARALLEL_WORKERS="${PARALLEL_WORKERS}" \
      RANDOM_SEED="${RANDOM_SEED}" \
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
    if wait -n -p finished_pid; then
        exit_code=0
    else
        exit_code=$?
    fi
    case_name="${active_cases[${finished_pid}]:-unknown}"
    unset 'active_cases['"${finished_pid}"']'
    if (( exit_code == 0 )); then
        echo "Case complete: ${case_name}"
    else
        echo "Case failed: ${case_name} (exit ${exit_code})" >&2
        failures=$((failures + 1))
    fi
}

for entry in "${selected_cases[@]}"; do
    while (( ${#active_cases[@]} >= JOBS )); do
        reap_one
    done
    read -r name _ <<< "${entry}"
    run_case "${entry}" &
    active_cases[$!]="${name}"
done

while (( ${#active_cases[@]} > 0 )); do
    reap_one
done

echo "Sweep finished: ${SWEEP_ROOT}; failures=${failures}"
(( failures == 0 ))
