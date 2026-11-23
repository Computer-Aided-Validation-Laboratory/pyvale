# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for all validation examples and their isolated output directories."""

from collections.abc import Callable
from pathlib import Path

import pytest

CUBE_OUTPUTS = tuple(
    f"{component}_{diagnostic}.png"
    for component in (
        "strain_xx",
        "strain_yy",
        "strain_zz",
        "strain_xy",
        "strain_xz",
        "strain_yz",
    )
    for diagnostic in ("cdfs", "mavm_time")
)


PULSE25X_OUTPUTS = (
    tuple(
        f"0_sim_allcdfs_{channel}.png"
        for channel in (
            "TC1",
            "TC2",
            "TC3",
            "TC4",
            "TC5",
            "TC6",
            "TC7",
            "TC8",
            "TC9",
            "TC10",
            "CV",
        )
    )
    + tuple(
        f"{diagnostic}_{channel}.png"
        for channel in ("TC2", "TC3", "TC5", "TC6", "TC8", "TC9", "TC10", "CV")
        for diagnostic in (
            "1_exp-sim-cdfs",
            "2_dextremes_wcdfs",
            "3_dextremes_wexpcdfs",
            "4_dextremes_simcdfsonly",
        )
    )
    + tuple(
        f"mavm_support_envelope_{channel}.png"
        for channel in ("TC2", "TC3", "TC5", "TC6", "TC8", "TC9", "TC10", "CV")
    )
)


EXAMPLES = (
    (
        "ex00_valid_quickstart",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_mavm_time.png",
            "temperature_mavm_d_minus_on_sim.png",
            "temperature_mavm_d_plus_on_sim.png",
        ),
    ),
    (
        "ex01_valid_scalar_metrics",
        (
            "observable_cdfs.png",
            "avm_overlay.png",
            "avm_area.png",
            "ks_overlay.png",
            "cvm_overlay.png",
            "rmse_overlay.png",
            "relative_error_overlay.png",
            "u_pooling.png",
            "mavm_d_plus_area.png",
            "mavm_d_minus_area.png",
            "mavm_support_envelope.png",
        ),
    ),
    (
        "ex02_valid_steady_state",
        (
            "temperature_cdfs.png",
            "temperature_avm_on_sim.png",
            "mavm_support_envelope.png",
        ),
    ),
    (
        "ex03_valid_transient",
        (
            "temperature_cdfs.png",
            "temperature_avm_time.png",
            "temperature_mavm_time.png",
        ),
    ),
    (
        "ex04_valid_epistemic_bound_metrics",
        (
            "avm_bounds.png",
            "avm_original.png",
            "mavm_support_envelope_bounds.png",
            "mavm_support_envelope_original.png",
        ),
    ),
    (
        "ex05_valid_nuisance_ecdf_profiles",
        (
            "ecdf_sim_nuis_envelope_vs_prof_median.png",
            "ecdf_exp_nuis_envelope_vs_prof_median.png",
            "ecdf_sim_exp_nuis_envelope_vs_prof_median.png",
            "ecdf_sim_exp_nuis_envelope_vs_prof_mean.png",
            "ecdf_sim_exp_nuis_envelope_vs_prof_q25.png",
            "ks_nuis_envelope_gap.png",
            "ks_nuis_prof_median_gap.png",
            "mavm_support_envelope_original_pairs.png",
        ),
    ),
    (
        "ex06_valid_nuisance_metric_profiles",
        (
            "validation_case.npz",
            "validation_result.npz",
            "avm_original_support.png",
            "mavm_nuisance_support_envelope.png",
            "avm_distance_profile.png",
            "avm_distance_profile.npz",
            "ks_statistic_profile.png",
            "ks_statistic_profile.npz",
            "cvm_statistic_profile.png",
            "cvm_statistic_profile.npz",
            "rmse_error_profile.png",
            "rmse_error_profile.npz",
            "relative_error_error_profile.png",
            "relative_error_error_profile.npz",
            "mavm_profile.png",
            "mavm_d_plus_extrema_ecdfs.png",
            "mavm_profile.npz",
            "total_width_minimum_support_envelope.png",
            "total_width_median_support_envelope.png",
            "total_width_maximum_support_envelope.png",
            "added_width_minimum_support_envelope.png",
            "added_width_median_support_envelope.png",
            "added_width_maximum_support_envelope.png",
        ),
    ),
    (
        "ex07_valid_epistemic_offsets",
        (
            "mavm_offset_confidence_ecdfs.png",
            "mavm_time.png",
        ),
    ),
    (
        "ex08_valid_sim_sensor_sampling",
        (
            "temperature_average_cdfs.png",
            "displacement_local_cdfs.png",
            "strain_local_cdfs.png",
            "temperature_average_mavm_time.png",
            "displacement_local_mavm_time.png",
            "strain_local_mavm_time.png",
        ),
    ),
    (
        "ex09_valid_sim_sweep_sampling",
        CUBE_OUTPUTS
        + (
            "nominal_avm_time.png",
            "heat_flux_plus_10pct_avm_time.png",
            "thermal_expansion_plus_10pct_avm_time.png",
        ),
    ),
    ("ex10_valid_pulse25x_benchmark", PULSE25X_OUTPUTS),
    (
        "ex11_valid_sensorsim_import",
        ("temperature_ecdfs.png", "temperature_mavm_time.png"),
    ),
    (
        "ex12_parallel_valid_ir_camera",
        ("ir_camera_avm.png",),
    ),
)


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "outputs"),
    [(f"validation/{name}.py", outputs) for name, outputs in EXAMPLES],
)
def test_valid_example(
    run_example: Callable[..., Path],
    example: str,
    outputs: tuple[str, ...],
) -> None:
    """Every example runs unattended and keeps its artifacts in its own directory."""
    output_directory = "valid_" + Path(example).stem.replace("_valid_", "_", 1)
    expected_outputs = tuple(
        f"pyvale-output/{output_directory}/{output}" for output in outputs
    )
    work_dir = run_example(example, expected_outputs)
    assert not tuple(work_dir.glob("*.png"))
    assert tuple(path.name for path in (work_dir / "pyvale-output").iterdir()) == (
        output_directory,
    )
    for output in expected_outputs:
        assert (work_dir / output).stat().st_size > 0



def test_valid_gallery_matches_registered_examples() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "src/pyvale/examples/validation"
    assert sorted(path.stem for path in examples_root.glob("ex*.py")) == sorted(
        name for name, _ in EXAMPLES
    )
