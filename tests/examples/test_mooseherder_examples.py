# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Run every mooseherder tutorial using packaged dependencies."""

from collections.abc import Callable
from pathlib import Path

import pytest

EXAMPLES = (
    ("ex01a_rabbit_quickstart", "mooseherder_ex01a_solve/results.e"),
    ("ex01b_inspect_simdata", "ex01b_simdata/simdata_inspection.npz"),
    ("ex02a_moose_cli_parameters", "mooseherder_ex02a_solve/cube_parameters.e"),
    ("ex02b_moose_included_inputs", "ex02b_solve/results.e"),
    ("ex02c_moose_multiple_inputs", "ex02c_solve/results.e"),
    ("ex02d_modify_input_text", "ex02d_solve/results.e"),
    ("ex02e_mpi_solve", "ex02e_solve/results.e"),
    ("ex02f_custom_moose_installation", "ex02f_solve/results.e"),
    ("ex03a_gmsh_geo", "ex03a_mesh/mesh.msh"),
    ("ex03b_gmsh_python", "ex03b_mesh/mesh.msh"),
    ("ex03c_custom_gmsh_installation", "ex03c_mesh/plate.inp"),
    ("ex04a_mesh_and_solve", "ex04a_case17/sweep_0001/sweep.json"),
    ("ex04b_sequential_sweep", "ex04b_sweep/sweep_0001/sweep.json"),
    ("ex04c_parallel_sweep", "ex04c_sweep/sweep_0001/sweep.json"),
    ("ex04d_read_sweep_results", "ex04d_sweep/sweep_0001/sweep.json"),
    (
        "ex05a_reuse_particle_directories",
        "ex05a_particles/particle_0007/evaluation.json",
    ),
    ("ex05b_pymoo_femu", "ex05b_femu/identification.json"),
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

    if example.endswith("ex01b_inspect_simdata.py"):
        import numpy as np

        with np.load(directory / "pyvale-output" / output) as arrays:
            np.testing.assert_allclose(
                arrays["final_disp_y_mm"], 1000 * arrays["final_disp_y_m"]
            )
            np.testing.assert_allclose(
                arrays["selected_disp_y_m"][:, 0], arrays["final_disp_y_m"]
            )
            np.testing.assert_allclose(
                arrays["top_disp_y_m"],
                arrays["final_disp_y_m"][arrays["top_node_indices"]],
            )
            assert arrays["reaction_y_n"].shape == arrays["time_s"].shape

    if example.endswith("ex05b_pymoo_femu.py"):
        import json

        import numpy as np

        root = directory / "pyvale-output/ex05b_femu"
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
        particles = list((root / "particles").glob("particle_*"))
        assert len(particles) == summary["population_size"]
        assert len(list((root / "particles").glob("*/solve/results.e"))) == 13
        assert not list((root / "particles").glob("sweep_*"))
