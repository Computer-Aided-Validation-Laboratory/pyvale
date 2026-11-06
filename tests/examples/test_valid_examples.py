"""Smoke tests for the validation examples and their saved figures."""

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "outputs"),
    (
        ("valid/ex1_load_exp_data.py", ()),
        ("valid/ex2_load_prob_sim_data.py", ()),
        ("valid/ex3_cdfs.py", ("pyvale-output/ex3_cdfs_output.png",)),
        (
            "valid/ex4_mavm_tc_analysis.py",
            (
                "pyvale-output/ex4_mavm_tc3_cdf.png",
                "pyvale-output/ex4_mavm_summary_bars.png",
            ),
        ),
        ("valid/ex5_validation_metrics_comparison.py", ()),
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
