#!/usr/bin/env bash
# Frozen five-run WDBN1 guarded-EGI campaign. Run only on the R0379 WSL host.
set -euo pipefail

REPO=/home/bx0923/projects/pyvale
DATA_ROOT=/home/bx0923/projects/pyvale-vfm-test-data/notched-ebw
OBJECTIVE_CONFIG=${REPO}/dev/vfm/data/wdbn1_guarded_egi_primary_v1_20260901.json
FRE_CONFIG=${REPO}/dev/vfm/data/wdbn1_fre_finest_stable_v1_20260901.json
FRE_ROI=${REPO}/dev/vfm/data/wdbn1_nominal_fre_roi_v1_20260901.yaml
CALLER=${REPO}/dev/vfm/call_notched_ebw_bivariate_identification.py
REPORTER=${REPO}/dev/vfm/report_guarded_egi_identification.py

common_args=(
  --guarded-egi-objective-config "${OBJECTIVE_CONFIG}"
  --egi-support-set fine-broad
  --force-slices auto
  --fre-resolution-config "${FRE_CONFIG}"
  --force-axis y
  --kernel-type bivariate_spd
  --basis-growth-policy sensitivity_correction
  --correction-sensitivity-perturbation-factor 0.01
  --correction-feature-fraction 0.2
  --fixed-basis-trajectory
  --minimum-objective-improvement 0
  --max-basis-functions 7
  --phase-0-max-evaluations 50
  --initial-mesh-size 0.1
  --minimum-mesh-size 0.001
  --objective-relative-tolerance 0.0001
  --centre-bounds-span-factor 1
  --max-iterations 200
  --max-evaluations 15500
  --parallel-workers 8
  --random-seed 0
  --stress-backend cython
  --egi-fft-dtype float32
  --egi-fft-groups split-broad
  --egi-skip-derived-diagnostics
  --artificial-noise-scale 0
)

launch_case() {
  local session=$1 input=$2 output_root=$3 fine=$4 fre_mode=$5
  local run_dir=${output_root}/prepared/${session}
  local log_dir=${output_root}/logs
  local log_file=${log_dir}/${session}.log
  local report_dir=${run_dir}/reports
  local -a correction_args=() report_args=()
  if [[ ${fre_mode} == enabled ]]; then
    correction_args=(--fre-region-of-interest "${FRE_ROI}")
    report_args=(--fre-region-of-interest "${FRE_ROI}" --experimental)
  elif [[ ${fre_mode} != disabled ]]; then
    echo "STOP: invalid FRE correction mode ${fre_mode}" >&2
    return 2
  fi
  [[ ! -e ${run_dir} ]] || { echo "STOP: output already exists: ${run_dir}" >&2; return 2; }
  ! tmux has-session -t "${session}" 2>/dev/null || { echo "STOP: tmux session exists: ${session}" >&2; return 2; }
  mkdir -p "${log_dir}"
  local -a identify=(uv run --no-sync python "${CALLER}" --input "${input}" --output-root "${output_root}" --run-name "prepared/${session}" --fine-egi-window "${fine}" "${correction_args[@]}" "${common_args[@]}")
  local -a report=(uv run --no-sync python "${REPORTER}" --input "${input}" --run "${run_dir}" --output "${report_dir}/${session}_REPORT.pdf" --title "${session} guarded EGI-primary identification" "${report_args[@]}")
  local identify_q report_q
  printf -v identify_q '%q ' "${identify[@]}"
  printf -v report_q '%q ' "${report[@]}"
  local body
  printf -v body 'cd %q && set -o pipefail && export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib-%q && %s 2>&1 | tee %q; run_status=${PIPESTATUS[0]}; if [[ ${run_status} -eq 0 ]]; then %s 2>&1 | tee -a %q; report_status=${PIPESTATUS[0]}; else report_status=99; fi; printf "RUN_EXIT=%%s REPORT_EXIT=%%s\\n" "${run_status}" "${report_status}" | tee -a %q; exit ${run_status}' "${REPO}" "${session}" "${identify_q}" "${log_file}" "${report_q}" "${log_file}" "${log_file}"
  tmux new-session -d -s "${session}" "bash -lc $(printf '%q' "${body}")"
  printf 'Started %-24s FRE correction=%-8s input=%s\n' "${session}" "${fre_mode}" "${input}"
}

case ${1:-help} in
  preflight)
    cd "${REPO}"
    uv run --no-sync python dev/vfm/validate_guarded_overnight_campaign.py
    ;;
  launch)
    cd "${REPO}"
    launch_case guarded_wdbn1_exp \
      "${DATA_ROOT}/experimental/wdbn1/vfm-input-data_2026-08-17_04-10" \
      "${DATA_ROOT}/experimental/wdbn1/identification" 43 enabled
    launch_case guarded_x2_clean \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/prepared" \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/identification" 21 disabled
    launch_case guarded_x2_noisy1x \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/prepared" \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/identification" 21 disabled
    launch_case guarded_full_clean \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-final-h0125-r4-fe-roi/pyvale-vfm/prepared" \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-final-h0125-r4-fe-roi/pyvale-vfm/identification" 43 disabled
    launch_case guarded_full_noisy1x \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-r2-fe-roi/pyvale-vfm/prepared" \
      "${DATA_ROOT}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-r2-fe-roi/pyvale-vfm/identification" 43 disabled
    ;;
  status)
    tmux list-sessions || true
    pgrep -af 'call_notched_ebw_bivariate_identification.py' || true
    find "${DATA_ROOT}" -path '*/prepared/guarded_*/identification_result.yaml' -print
    ;;
  *)
    echo "Usage: $0 {preflight|launch|status}" >&2
    exit 2
    ;;
esac
