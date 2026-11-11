# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Smoke tests for all validation examples and their isolated output directories."""

import json
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
    for diagnostic in ("cdfs", "d_minus_time")
) + tuple(
    f"strain_xx_balance_on_{background}.png"
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
    + ("pointsensors_mavm.csv", "channel_status.csv", "parity.json", "axis_limits.json")
)


EXAMPLES = (
    (
        "ex0_val_quickstart",
        "valid_ex0_quickstart",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_d_plus_time.png",
            "temperature_mavm_balance_on_sim.png",
        ),
    ),
    ("ex1_val_metric", "valid_ex1_metric", ("observable_cdfs.png",)),
    (
        "ex2_val_steady",
        "valid_ex2_steady",
        ("temperature_cdfs.png", "temperature_avm_on_sim.png"),
    ),
    (
        "ex3_val_epistemic",
        "valid_ex3_epistemic",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_mavm_balance_on_sim.png",
        ),
    ),
    (
        "ex4_val_bounds",
        "valid_ex4_bounds",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_mavm_balance_on_sim.png",
        ),
    ),
    (
        "ex5_val_transient",
        "valid_ex5_transient",
        (
            "temperature_cdfs.png",
            "temperature_d_plus_time.png",
            "temperature_mavm_balance_on_sim.png",
        ),
    ),
    (
        "ex6_val_diagnostics",
        "valid_ex6_diagnostics",
        (
            "temperature_cdfs.png",
            "temperature_envelopes.png",
            "temperature_d_plus_time.png",
            "temperature_mavm_balance_on_sim.png",
            "conditional_cdf_families.png",
            "mavm_distributions.png",
            "mavm_comparisons.png",
            "mavm_correction.png",
            "mavm_model_correction.png",
            "directional_metric_distributions.png",
        ),
    ),
    (
        "ex7_val_field",
        "valid_ex7_field",
        (
            "strain_xx_avm_on_sim.png",
            "temperature_avm_time.png",
            "temperature_cdfs.png",
        ),
    ),
    ("ex8_val_cube_tensor", "valid_ex8_cube_tensor", CUBE_OUTPUTS),
    (
        "ex9_val_cube_sweep",
        "valid_ex9_cube_sweep",
        CUBE_OUTPUTS
        + (
            "nominal_avm_time.png",
            "heat_flux_plus_10pct_avm_time.png",
            "thermal_expansion_plus_10pct_avm_time.png",
        ),
    ),
    ("ex10_val_pulse25x", "valid_ex10_pulse25x", PULSE25X_OUTPUTS),
)


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "output_directory", "outputs"),
    [(f"valid/{name}.py", directory, outputs) for name, directory, outputs in EXAMPLES],
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

    if example == "valid/ex10_val_pulse25x.py":
        with (
            work_dir / "pyvale-output" / output_directory / "parity.json"
        ).open() as stream:
            parity = json.load(stream)
        assert parity["compared_values"] == 64
        assert parity["maximum_absolute_difference"] < 1e-12
