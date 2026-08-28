#!/usr/bin/env bash
set -uo pipefail

# Overnight diagnostic sweep following the 2026-08-27 identifiability study.
# This changes no production configuration. Each case is resumable: an existing
# identification_result.yaml or analysis summary causes that stage to be skipped.

dataset="/home/robh/1_Projects/pyvale-vfm-test-data/notched-ebw/synthetic-fe/wdbn1-idealised-yield/pyvale-vfm"
prepared="$dataset/identification/prepared"
sweep_name="overnight_geometry_sweep_20260827"
report_root="dev/vfm/output/$sweep_name"
log_root="$report_root/logs"

mkdir -p "$log_root"

run_fixed_case() {
    local case_name="$1"
    local case_letter="$2"
    local output="$report_root/fixed_geometry"
    local summary="$output/$case_name/summary.json"
    local log="$log_root/fixed_${case_letter}.log"

    if [[ -f "$summary" ]]; then
        echo "SKIP fixed $case_letter: $summary exists"
        return
    fi
    echo "START fixed $case_letter $(date --iso-8601=seconds)"
    MPLCONFIGDIR=/tmp/pyvale-matplotlib \
    UV_CACHE_DIR=/tmp/pyvale-uv-cache \
    uv run python dev/vfm/run_notched_ebw_fixed_geometry_tests.py \
        --case "$case_letter" \
        --output "$output" \
        --max-iterations 120 \
        --max-evaluations 3000 \
        --parallel-workers 12 \
        --initial-hardening 4000 \
        >"$log" 2>&1
    local status=$?
    echo "END fixed $case_letter status=$status $(date --iso-8601=seconds)"
}

run_identification() {
    local label="$1"
    local bases="$2"
    local seed="$3"
    local smoothing="$4"
    local mesh="$5"
    local run_name="prepared/${sweep_name}_${label}"
    local result="$dataset/identification/$run_name/identification_result.yaml"
    local analysis="$dataset/identification/${run_name}_analysis"
    local run_log="$log_root/${label}_identification.log"
    local analysis_log="$log_root/${label}_analysis.log"

    if [[ -f "$result" ]]; then
        echo "SKIP identification $label: result exists"
    else
        echo "START identification $label $(date --iso-8601=seconds)"
        MPLCONFIGDIR=/tmp/pyvale-matplotlib \
        UV_CACHE_DIR=/tmp/pyvale-uv-cache \
        uv run python dev/vfm/call_notched_ebw_bivariate_identification.py \
            --run-name "$run_name" \
            --egi-windows 29,57 \
            --force-weight 0.1 \
            --force-slices 63 \
            --max-basis-functions "$bases" \
            --minimum-objective-improvement 0.0 \
            --refinement-smoothing-points "$smoothing" \
            --initial-mesh-size "$mesh" \
            --minimum-mesh-size 0.0005 \
            --max-iterations 250 \
            --max-evaluations 17000 \
            --parallel-workers 12 \
            --random-seed "$seed" \
            --stress-backend cython \
            --no-progress \
            >"$run_log" 2>&1
        local status=$?
        echo "END identification $label status=$status $(date --iso-8601=seconds)"
    fi

    if [[ ! -f "$result" ]]; then
        echo "SKIP analysis $label: identification result missing"
        return
    fi
    if [[ -f "$analysis/summary.json" ]]; then
        echo "SKIP analysis $label: summary exists"
        return
    fi
    echo "START analysis $label $(date --iso-8601=seconds)"
    MPLCONFIGDIR=/tmp/pyvale-matplotlib \
    UV_CACHE_DIR=/tmp/pyvale-uv-cache \
    uv run python dev/vfm/analyse_vfm_results_notched_ebw.py \
        --result "$result" \
        --output "$analysis" \
        --egi-window-size 29 \
        >"$analysis_log" 2>&1
    local status=$?
    echo "END analysis $label status=$status $(date --iso-8601=seconds)"
}

echo "Overnight sweep started $(date --iso-8601=seconds)"

# Closure controls: confirm that the fixed oracle result remains good when the
# deliberately short 30-iteration diagnostic is allowed to converge further.
run_fixed_case "A_oracle_geometry_hardening_fixed" A
run_fixed_case "B_oracle_geometry_hardening_free" B

# Controlled model-order check. The zero gate is diagnostic: it allows growth
# while still rejecting a basis if it fails to improve the mechanical objective.
run_identification "bases4_seed0_smooth3_mesh010" 4 0 3 0.10

# Five-basis repeatability: tests whether useful geometry is found reliably.
run_identification "bases5_seed0_smooth3_mesh010" 5 0 3 0.10
run_identification "bases5_seed1_smooth3_mesh010" 5 1 3 0.10
run_identification "bases5_seed2_smooth3_mesh010" 5 2 3 0.10
run_identification "bases5_seed3_smooth3_mesh010" 5 3 3 0.10

# Seed-map resolution: changes only EGI peak smoothing.
run_identification "bases5_seed0_smooth1_mesh010" 5 0 1 0.10
run_identification "bases5_seed0_smooth5_mesh010" 5 0 5 0.10

# Geometry search scale: changes only the initial normalised pattern-search mesh.
run_identification "bases5_seed0_smooth3_mesh005" 5 0 3 0.05
run_identification "bases5_seed0_smooth3_mesh020" 5 0 3 0.20

echo "Overnight sweep finished $(date --iso-8601=seconds)"
echo "Logs: $log_root"
echo "Results: $prepared/${sweep_name}_*"
