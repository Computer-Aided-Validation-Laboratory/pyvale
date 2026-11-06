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
    ("ex07_mesh_and_solve", "ex07_case17/sweep-0001/sweep.json"),
    ("ex08_sequential_sweep", "ex08_sweep/sweep-0001/sweep.json"),
    ("ex09_parallel_sweep", "ex09_sweep/sweep-0001/sweep.json"),
    ("ex10_mpi_solve", "ex10_solve/results.e"),
    ("ex11_custom_installations", "ex11_solve/results.e"),
    ("ex12_read_sweep_results", "ex12_sweep/sweep-0001/sweep.json"),
    ("ex13_modify_input_text", "ex13_solve/results.e"),
    ("ex14_reuse_particle_directories", "ex14_particles/particle-0007/evaluation.json"),
    ("ex15_pymoo_femu", "ex15_femu/identification.json"),
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

    if example.endswith("ex15_pymoo_femu.py"):
        import json

        import numpy as np

        root = directory / "pyvale-output/ex15_femu"
        summary = json.loads((root / "identification.json").read_text())
        assert summary["case"] == "case17"
        assert summary["reaction_global"] == "react_y_top"
        assert summary["modulus_relative_error"] < 0.02
        assert summary["poisson_absolute_error"] < 0.005
        assert summary["strain_components"] == ["strain_xx", "strain_yy"]
        assert summary["n_evaluations"] == 288
        assert summary["cost"] < summary["initial_best_cost"]
        history = np.loadtxt(root / "evaluations.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(history[:, 7], 0.5 * (history[:, 5] + history[:, 6]))
        with (
            np.load(root / "measurements.npz") as observed,
            np.load(root / "best_fit.npz") as fitted,
        ):
            assert observed["strain"].ndim == 2
            assert observed["strain"].shape[1] == 2
            assert fitted["strain"].shape == observed["strain"].shape
            np.testing.assert_array_equal(
                fitted["positions_mm"], observed["positions_mm"]
            )
            force_cost = (
                (float(fitted["force_n"]) - float(observed["force_n"]))
                / summary["sigma_force_n"]
            ) ** 2
            strain_cost = np.mean(
                ((fitted["strain"] - observed["strain"]) / summary["sigma_strain"]) ** 2
            )
            np.testing.assert_allclose(
                summary["cost"], 0.5 * (force_cost + strain_cost), rtol=1e-8
            )
        assert (root / "femu_fit.png").stat().st_size > 0
        particles = list((root / "particles").glob("particle-*"))
        assert len(particles) == summary["population_size"]
        assert len(list((root / "particles").glob("*/solve/results.e"))) == 13
        assert not list((root / "particles").glob("sweep-*"))
