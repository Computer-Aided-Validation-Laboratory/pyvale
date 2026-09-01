#!/usr/bin/env bash
set -euo pipefail

REPO=/home/bx0923/projects/pyvale
TEST_DATA=/home/bx0923/projects/pyvale-vfm-test-data/notched-ebw
CALLER=${REPO}/dev/vfm/call_notched_ebw_five_phase_identification.py
OBJECTIVE_CONFIG=${REPO}/dev/vfm/data/wdbn1_guarded_egi_primary_v1_20260901.json
FRE_CONFIG=${REPO}/dev/vfm/data/wdbn1_fre_finest_stable_v1_20260901.json
FRE_ROI=${REPO}/dev/vfm/data/wdbn1_nominal_fre_roi_v1_20260901.yaml
PS50_INPUT=${TEST_DATA}/experimental/wdbn1-ps50/prepared
SESSION_FILE=${REPO}/dev/vfm/output/workstation_five_phase/latest_session.txt

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
  --sbvf-max-evaluations 500
)

preflight() {
  command -v tmux >/dev/null || { echo "STOP: tmux is not installed" >&2; return 2; }
  grep -q -- "phase-2-fix-hardening" "${CALLER}" || {
    echo "STOP: source lacks --phase-2-fix-hardening" >&2
    return 2
  }
  grep -q -- "default=_json_default" "${REPO}/src/pyvale/vfm/campaignprogress.py" || {
    echo "STOP: source lacks the NumPy manifest serialization fix" >&2
    return 2
  }
  local required
  for required in \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/prepared/experiment_data.yaml" \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/prepared/experiment_data.yaml" \
    "${PS50_INPUT}/experiment_data.yaml" \
    "${OBJECTIVE_CONFIG}" "${FRE_CONFIG}" "${FRE_ROI}"
  do
    [[ -f ${required} ]] || { echo "STOP: missing ${required}" >&2; return 2; }
  done
}

launch_window() {
  local session=$1 tag=$2 log_dir=$3 window=$4 input=$5 output=$6 fine=$7
  local hardening_mode=$8 physical_fre_roi=$9
  local run_name="five_phase_${window}_${tag}"
  local -a extra_args=()
  [[ ${hardening_mode} == fixed ]] && extra_args+=(--phase-2-fix-hardening)
  [[ ${physical_fre_roi} == yes ]] && extra_args+=(--fre-region-of-interest "${FRE_ROI}")
  local -a command=(
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
    NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR="/tmp/matplotlib-${run_name}"
    uv run --no-sync python "${CALLER}"
    --input "${input}" --output-root "${output}" --run-name "${run_name}"
    --fine-egi-window "${fine}" "${extra_args[@]}" "${common_args[@]}"
  )
  local command_q body
  printf -v command_q '%q ' "${command[@]}"
  printf -v body \
    'cd %q && set -o pipefail && %s 2>&1 | tee %q; status=${PIPESTATUS[0]}; printf "\nRUN_EXIT=%%s\n" "${status}"; exec bash' \
    "${REPO}" "${command_q}" "${log_dir}/${run_name}.log"
  tmux new-window -d -t "${session}:" -n "${window}" \
    "bash -lc $(printf '%q' "${body}")"
}

launch() {
  preflight
  local tag session log_dir
  tag=$(date +%Y%m%d_%H%M%S)
  session="fivephase4_${tag}"
  log_dir="${REPO}/dev/vfm/output/workstation_five_phase/${tag}"
  mkdir -p "${log_dir}" "$(dirname "${SESSION_FILE}")"
  tmux new-session -d -s "${session}" -n control "bash"
  launch_window "${session}" "${tag}" "${log_dir}" clean_hfixed \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/prepared" \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-clean-fe-roi-spatial-x2/pyvale-vfm/identification" \
    21 fixed no
  launch_window "${session}" "${tag}" "${log_dir}" noisy_hfixed \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/prepared" \
    "${TEST_DATA}/synthetic-fe/wdbn1-representative-fe-v1-noisy-1x-seed20260830-fe-roi-spatial-x2/pyvale-vfm/identification" \
    21 fixed no
  launch_window "${session}" "${tag}" "${log_dir}" ps50_hfixed \
    "${PS50_INPUT}" "${TEST_DATA}/experimental/wdbn1-ps50/identification" \
    43 fixed yes
  launch_window "${session}" "${tag}" "${log_dir}" ps50_hfree \
    "${PS50_INPUT}" "${TEST_DATA}/experimental/wdbn1-ps50/identification" \
    43 free yes
  printf '%s\n' "${session}" > "${SESSION_FILE}"
  tmux select-window -t "${session}:clean_hfixed"
  echo "Started ${session}"
  echo "Logs: ${log_dir}"
  echo "Attach: bash $0 attach"
}

latest_session() {
  [[ -f ${SESSION_FILE} ]] || { echo "STOP: no recorded session" >&2; return 2; }
  head -n 1 "${SESSION_FILE}"
}

status() {
  local session window
  session=$(latest_session)
  date -Is
  uptime
  free -h | sed -n '1,2p'
  tmux list-windows -t "${session}" -F '#{window_name}: dead=#{pane_dead} pid=#{pane_pid}'
  for window in clean_hfixed noisy_hfixed ps50_hfixed ps50_hfree; do
    echo
    echo "===== ${window} ====="
    tmux capture-pane -p -t "${session}:${window}" -S -6 | tail -6
  done
}

case ${1:-help} in
  preflight) preflight ;;
  launch) launch ;;
  status) status ;;
  attach) tmux attach -t "$(latest_session)" ;;
  *) echo "Usage: $0 {preflight|launch|status|attach}" >&2; exit 2 ;;
esac
