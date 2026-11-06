"""Run every mooseherder tutorial using packaged dependencies."""

from collections.abc import Callable
from pathlib import Path

import pytest

EXAMPLES = (
    ("ex01_rabbit_quickstart", "ex01_solve/results.e"),
    ("ex02_moose_cli_parameters", "ex02_solve/results.e"),
    ("ex03_moose_included_inputs", "ex03_solve/results.e"),
    ("ex04_moose_multiple_inputs", "ex04_solve/results.e"),
    ("ex05_gmsh_geo", "ex05_mesh/mesh.msh"),
    ("ex06_gmsh_python", "ex06_mesh/mesh.msh"),
    ("ex07_mesh_and_solve", "ex07_specimen/sweep-0001/sweep.json"),
    ("ex08_sequential_sweep", "ex08_sweep/sweep-0001/sweep.json"),
    ("ex09_parallel_sweep", "ex09_sweep/sweep-0001/sweep.json"),
    ("ex10_mpi_solve", "ex10_solve/results.e"),
    ("ex11_custom_installations", "ex11_solve/results.e"),
    ("ex12_read_sweep_results", "ex12_sweep/sweep-0001/sweep.json"),
    ("ex13_modify_input_text", "ex13_solve/results.e"),
)


@pytest.mark.example
@pytest.mark.parametrize(
    ("example", "output"),
    [(f"mooseherder/{name}.py", output) for name, output in EXAMPLES],
)
def test_mooseherder_example(
    run_example: Callable[..., Path],
    example: str,
    output: str,
) -> None:
    """Every example runs unattended, including MPI/custom demos' defaults."""
    directory = run_example(example, (f"pyvale-output/{output}",))
    assert (directory / "pyvale-output" / output).stat().st_size > 0
