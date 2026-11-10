"""Smoke tests for the validation examples and their saved figures."""

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "outputs"),
    (
        ("valid/ex1_val_metric.py", ("pyvale-output/ex1_val_metric_cdfs.png",)),
        (
            "valid/ex2_val_steady.py",
            (
                "pyvale-output/ex2_val_steady_cdfs.png",
                "pyvale-output/ex2_val_steady_map.png",
            ),
        ),
        (
            "valid/ex3_val_epistemic.py",
            (
                "pyvale-output/ex3_val_epistemic_envelope.png",
                "pyvale-output/ex3_val_epistemic_distribution.png",
            ),
        ),
        (
            "valid/ex4_val_bounds.py",
            (
                "pyvale-output/ex4_val_bounds_envelope.png",
                "pyvale-output/ex4_val_bounds_directional_map.png",
                "pyvale-output/ex4_val_bounds_time.png",
            ),
        ),
        (
            "valid/ex5_val_transient.py",
            (
                "pyvale-output/ex5_val_transient_time.png",
                "pyvale-output/ex5_val_transient_cdfs.png",
            ),
        ),
        (
            "valid/ex6_val_field.py",
            (
                "pyvale-output/ex6_val_field_strain_map.png",
                "pyvale-output/ex6_val_field_temperature_time.png",
            ),
        ),
    ),
)
def test_valid_example(
    run_example: Callable[..., Path],
    example: str,
    outputs: tuple[str, ...],
) -> None:
    """Every validation example runs unattended and saves expected outputs."""
    work_dir = run_example(example, outputs)
    assert not tuple(work_dir.glob("*.png"))
    for output in outputs:
        assert (work_dir / output).stat().st_size > 0
