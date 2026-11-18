# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for all validation examples and their isolated output directories."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
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
    for diagnostic in ("cdfs", "d_plus_time", "d_minus_time")
) + tuple(
    f"strain_xx_{direction}_on_{background}.png"
    for direction in ("d_plus", "d_minus")
    for background in (
        "temperature",
        "strain_xx",
        "equivalent_strain",
        "first_principal_strain",
    )
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
        f"mavm_{direction}_limiting_ecdf_{channel}.png"
        for direction in ("d_plus", "d_minus")
        for channel in ("TC2", "TC3", "TC5", "TC6", "TC8", "TC9", "TC10", "CV")
    )
    + ("pointsensors_mavm.csv", "channel_status.csv", "parity.json", "axis_limits.json")
)


EXAMPLES = (
    (
        "ex00_valid_quickstart",
        "valid_ex0_quickstart",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_d_plus_time.png",
            "temperature_d_minus_time.png",
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
            "mavm_d_plus_limiting_ecdf.png",
            "mavm_d_minus_limiting_ecdf.png",
        ),
    ),
    (
        "ex02_valid_steady",
        "valid_ex2_steady",
        (
            "temperature_cdfs.png",
            "temperature_avm_on_sim.png",
            "bias_reference.csv",
            "mavm_d_plus_limiting_ecdf.png",
            "mavm_d_minus_limiting_ecdf.png",
        ),
    ),
    (
        "ex03_valid_transient",
        "valid_ex3_transient",
        (
            "temperature_cdfs.png",
            "temperature_avm_time.png",
            "temperature_d_plus_time.png",
            "temperature_d_minus_time.png",
            "temperature_mavm_d_plus_on_sim.png",
            "temperature_mavm_d_minus_on_sim.png",
            "bias_reference.csv",
        ),
    ),
    (
        "ex04_valid_transferable_bounds",
        "valid_ex4_transferable_bounds",
        (
            "comparison.json",
            "avm_bounds.png",
            "avm_original.png",
            "mavm_d_plus_bounds.png",
            "mavm_d_minus_bounds.png",
            "mavm_d_plus_original.png",
            "mavm_d_minus_original.png",
        ),
    ),
    (
        "ex05_valid_epistemic_axes",
        "valid_ex5_epistemic_axes",
        (
            "outer_outer.png",
            "fixed_nuisance.png",
            "subset.png",
            "median_nuisance.png",
            "lower_quartile.png",
            "upper_quartile.png",
            "mean_curves.png",
            "median_curves.png",
            "selected_scenarios.png",
            "independent_views.png",
            "sim_median_nuisance.png",
            "exp_envelopes.png",
            "avm_normalised_p0.5.png",
            "avm_normalised_p0.9.png",
            "ks_bound_distances.png",
            "mavm_d_plus_support.png",
            "mavm_d_minus_support.png",
            "mavm_d_plus_area.png",
            "mavm_d_minus_area.png",
            "all_bounds.csv",
            "mean_bounds.csv",
            "median_bounds.csv",
            "minimum_bounds.csv",
            "lower_quartile_bounds.csv",
            "upper_quartile_bounds.csv",
            "maximum_bounds.csv",
        ),
    ),
    (
        "ex06_valid_nuisance_profiles",
        "valid_ex6_nuisance_profiles",
        (
            "profiling.json",
            "avm_original_support.png",
            "mavm_d_plus_nuisance_support.png",
            "mavm_d_minus_nuisance_support.png",
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
            "mavm_d_plus_profile.png",
            "mavm_d_plus_profile.csv",
            "mavm_d_minus_profile.png",
            "mavm_d_minus_profile.csv",
            "total_width_minimum_d_plus.png",
            "total_width_minimum_d_minus.png",
            "total_width_median_d_plus.png",
            "total_width_median_d_minus.png",
            "total_width_maximum_d_plus.png",
            "total_width_maximum_d_minus.png",
            "added_width_minimum_d_plus.png",
            "added_width_minimum_d_minus.png",
            "added_width_median_d_plus.png",
            "added_width_median_d_minus.png",
            "added_width_maximum_d_plus.png",
            "added_width_maximum_d_minus.png",
        ),
    ),
    (
        "ex07_valid_offsets",
        "valid_ex7_offsets",
        (
            "exp_offset_envelopes.png",
            "confidence_supports.json",
            "mavm_d_plus_time.png",
            "mavm_d_minus_time.png",
            "mavm_d_plus_confidence_support.png",
            "mavm_d_minus_confidence_support.png",
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
            "sampling_bias.csv",
            "temperature_average_cdfs.png",
            "displacement_local_cdfs.png",
            "strain_local_cdfs.png",
            "temperature_average_d_plus_time.png",
            "temperature_average_d_minus_time.png",
            "displacement_local_d_plus_time.png",
            "displacement_local_d_minus_time.png",
            "strain_local_d_plus_time.png",
            "strain_local_d_minus_time.png",
            "strain_local_d_plus_on_temperature.png",
            "strain_local_d_plus_on_equivalent_strain.png",
            "strain_local_d_minus_on_temperature.png",
            "strain_local_d_minus_on_equivalent_strain.png",
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

    output_root = work_dir / "pyvale-output" / output_directory
    if example in (
        "validation/ex02_valid_steady.py",
        "validation/ex03_valid_transient.py",
        "validation/ex08_valid_sampling.py",
    ):
        reference = np.loadtxt(
            output_root
            / ("sampling_bias.csv" if "sampling" in example else "bias_reference.csv"),
            delimiter=",",
            skiprows=1,
        )
        np.testing.assert_allclose(
            reference[:, -1], reference[:, -2], rtol=0, atol=1.0e-10
        )
        if "steady" in example:
            expected = (
                3
                * (reference[:, 0] - np.min(reference[:, 0]))
                / np.ptp(reference[:, 0])
            )
            np.testing.assert_allclose(reference[:, 1], expected, rtol=0, atol=1.0e-10)
        elif "transient" in example:
            final = reference[reference[:, 1] == np.max(reference[:, 1]), 2]
            assert np.min(final) == pytest.approx(2.0)
            assert np.max(final) == pytest.approx(5.0)
            assert np.all(reference[reference[:, 1] == np.min(reference[:, 1]), 2] == 0)
        else:
            np.testing.assert_allclose(np.unique(reference[:, 1]), [1.0, 2.0])
    if example == "validation/ex04_valid_transferable_bounds.py":
        comparison = json.loads((output_root / "comparison.json").read_text())
        np.testing.assert_allclose(comparison["bounds_upper"], [75.0, 135.0])
        np.testing.assert_allclose(comparison["original_upper"], [95.0, 155.0])
    if example == "validation/ex06_valid_nuisance_profiles.py":
        profiling = json.loads((output_root / "profiling.json").read_text())
        assert "sim_trans" in profiling["MAVM"]["retained_axes"]
        assert "sim_nuis" not in profiling["MAVM"]["retained_axes"]
        for score in ("total_width", "added_width"):
            ranking = profiling[score]
            scores = np.asarray(ranking["scores"])
            order = np.asarray(ranking["order"])
            assert np.all(np.diff(scores[order]) >= -1.0e-12)
            assert len(ranking["cases"]) == 9
    if example == "validation/ex10_valid_pulse25x.py":
        with (
            work_dir / "pyvale-output" / output_directory / "parity.json"
        ).open() as stream:
            parity = json.load(stream)
        assert parity["compared_values"] == 64
        assert parity["maximum_absolute_difference"] < 1e-12


def test_valid_gallery_contains_exactly_the_documented_eleven_examples() -> None:
    examples_root = Path(__file__).resolve().parents[2] / "src/pyvale/examples/validation"
    assert sorted(path.stem for path in examples_root.glob("ex*.py")) == sorted(
        name for name, _, _ in EXAMPLES
    )
    assert len(EXAMPLES) == 11
