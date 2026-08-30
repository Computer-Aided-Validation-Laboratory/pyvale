#!/usr/bin/env bash
set -euo pipefail

# Run the ten declared, simplex-valid weight cases serially. Each case gets a
# durable config snapshot, an independent identification directory, and both
# diagnostic PDFs. Override the environment variables below for workstation
# execution or a smaller pilot.

REPO_ROOT="${REPO_ROOT:-/home/robh/1_Projects/pyvale}"
SWEEP_TAG="${SWEEP_TAG:-simple_weight_sweep_7bf_$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${REPO_ROOT}/dev/vfm/output/${SWEEP_TAG}"
BASE_CONFIG="${REPO_ROOT}/dev/vfm/data/wdbn1_simple_sensitivity_gated_objective_v1_20260830.json"
MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS:-7}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
MAX_EVALUATIONS="${MAX_EVALUATIONS:-15500}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-8}"
RANDOM_SEED="${RANDOM_SEED:-0}"

if [[ -e "${SWEEP_ROOT}" ]]; then
    echo "Refusing to overwrite existing sweep output: ${SWEEP_ROOT}" >&2
    exit 1
fi
mkdir -p "${SWEEP_ROOT}/configs"

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
for entry in "${CASES[@]}"; do
    read -r name informative fre broad <<< "${entry}"
    config="${SWEEP_ROOT}/configs/${name}.json"
    run_tag="${SWEEP_TAG}_${name}"
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
    echo "Starting ${name}: informative=${informative}, FRE=${fre}, broad=${broad}"
    CONFIG="${config}" RUN_TAG="${run_tag}" \
      MAX_BASIS_FUNCTIONS="${MAX_BASIS_FUNCTIONS}" \
      MAX_ITERATIONS="${MAX_ITERATIONS}" \
      MAX_EVALUATIONS="${MAX_EVALUATIONS}" \
      PARALLEL_WORKERS="${PARALLEL_WORKERS}" \
      RANDOM_SEED="${RANDOM_SEED}" \
      bash dev/vfm/run_notched_ebw_simple_bf1_and_report.sh
done

echo "Completed sweep: ${SWEEP_ROOT}"
