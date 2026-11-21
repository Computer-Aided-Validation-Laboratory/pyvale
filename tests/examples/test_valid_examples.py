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
    + ("pointsensors_mavm.csv", "channel_status.csv")
)


EXAMPLES = (
    (
        "ex00_valid_quickstart",
        "valid_ex0_quickstart",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_mavm_time.png",
            "temperature_mavm_d_minus_on_sim.png",
            "temperature_mavm_d_plus_on_sim.png",
        ),
    ),
    (
        "ex01_valid_metric",
        "valid_ex1_metric",
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
        "ex02_valid_steady",
        "valid_ex2_steady",
        (
            "temperature_cdfs.png",
            "temperature_avm_on_sim.png",
            "mavm_support_envelope.png",
        ),
    ),
    (
        "ex03_valid_transient",
        "valid_ex3_transient",
        (
            "temperature_cdfs.png",
            "temperature_avm_time.png",
            "temperature_mavm_time.png",
        ),
    ),
    (
        "ex04_valid_transferable_bounds",
        "valid_ex4_transferable_bounds",
        (
            "avm_bounds.png",
            "avm_original.png",
            "mavm_support_envelope_bounds.png",
            "mavm_support_envelope_original.png",
        ),
    ),
    (
        "ex05_valid_epistemic_axes",
        "valid_ex5_epistemic_axes",
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
        "ex06_valid_nuisance_profiles",
        "valid_ex6_nuisance_profiles",
        (
            "avm_original_support.png",
            "mavm_nuisance_support_envelope.png",
            "avm_distance_profile.png",
            "avm_distance_profile.csv",
            "ks_statistic_profile.png",
            "ks_statistic_profile.csv",
            "cvm_statistic_profile.png",
            "cvm_statistic_profile.csv",
            "rmse_error_profile.png",
            "rmse_error_profile.csv",
            "relative_error_error_profile.png",
            "relative_error_error_profile.csv",
            "mavm_profile.png",
            "mavm_d_plus_profile.csv",
            "mavm_d_minus_profile.csv",
            "total_width_minimum_support_envelope.png",
            "total_width_median_support_envelope.png",
            "total_width_maximum_support_envelope.png",
            "added_width_minimum_support_envelope.png",
            "added_width_median_support_envelope.png",
            "added_width_maximum_support_envelope.png",
        ),
    ),
    (
        "ex07_valid_offsets",
        "valid_ex7_offsets",
        (
            "exp_offset_envelopes.png",
            "mavm_time.png",
            "mavm_confidence_support_envelope.png",
            "mavm_nominal_d_plus_area.png",
            "mavm_nominal_d_minus_area.png",
            "mavm_lower_d_plus_area.png",
            "mavm_lower_d_minus_area.png",
            "mavm_upper_d_plus_area.png",
            "mavm_upper_d_minus_area.png",
        ),
    ),
    (
        "ex08_valid_sampling",
        "valid_ex8_sampling",
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
        "ex09_valid_cube_sweep",
        "valid_ex9_cube_sweep",
        CUBE_OUTPUTS
        + (
            "parameter_sets.json",
            "nominal_avm_time.png",
            "heat_flux_plus_10pct_avm_time.png",
            "thermal_expansion_plus_10pct_avm_time.png",
        ),
    ),
    ("ex10_valid_pulse25x", "valid_ex10_pulse25x", PULSE25X_OUTPUTS),
)


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "output_directory", "outputs"),
    [(f"validation/{name}.py", directory, outputs) for name, directory, outputs in EXAMPLES],
)
def test_valid_example(
    run_example: Callable[..., Path],
    example: str,
    output_directory: str,
    outputs: tuple[str, ...],
) -> None:
    """Every example runs unattended and keeps its artifacts in its own directory."""
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



def test_valid_gallery_contains_exactly_the_documented_eleven_examples() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "src/pyvale/examples/validation"
    assert sorted(path.stem for path in examples_root.glob("ex*.py")) == sorted(
        name for name, _, _ in EXAMPLES
    )
    assert len(EXAMPLES) == 11
