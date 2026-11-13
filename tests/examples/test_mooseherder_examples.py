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
    ("ex01a_quickstart_rabbitfem", "mooseherder_ex01a_solve/results.e"),
    ("ex01b_quickstart_simdata", "mooseherder_ex01b_simdata/simdata_inspection.npz"),
    ("ex02a_moose_cli_parameters", "mooseherder_ex02a_solve/cube_parameters.e"),
    ("ex02b_moose_included_inputs", "mooseherder_ex02b_solve/results.e"),
    ("ex02c_moose_multiple_inputs", "mooseherder_ex02c_solve/results.e"),
    ("ex02d_moose_modify_input_text", "mooseherder_ex02d_solve/results.e"),
    ("ex02e_moose_mpi_solve", "mooseherder_ex02e_solve/results.e"),
    ("ex02f_moose_custom_installation", "mooseherder_ex02f_solve/results.e"),
    ("ex03a_gmsh_geo", "mooseherder_ex03a_mesh/mesh.msh"),
    ("ex03b_gmsh_python", "mooseherder_ex03b_mesh/mesh.msh"),
    ("ex03c_gmsh_custom_installation", "mooseherder_ex03c_mesh/plate.inp"),
    ("ex03d_gmsh_plate3d", "mooseherder_ex03d_mesh/mesh.msh"),
    ("ex03e_gmsh_monoblock", "mooseherder_ex03e_mesh/mesh.msh"),
    ("ex04a_simsweep_mesh_and_solve", "mooseherder_ex04a_case17/sweep_0001/sweep.json"),
    (
        "ex04b_simsweep_sequential_sweep",
        "mooseherder_ex04b_sweep/sweep_0001/sweep.json",
    ),
    ("ex04c_simsweep_parallel_sweep", "mooseherder_ex04c_sweep/sweep_0001/sweep.json"),
    (
        "ex04d_simsweep_read_sweep_results",
        "mooseherder_ex04d_sweep/sweep_0001/sweep.json",
    ),
    ("ex05a_femu", "mooseherder_ex05a_femu/identification.npz"),
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
    assert all(
        path.is_dir() and path.name.startswith("mooseherder_")
        for path in (directory / "pyvale-output").iterdir()
    )

    if example.endswith(("ex03d_gmsh_plate3d.py", "ex03e_gmsh_monoblock.py")):
        import gmsh
        import numpy as np

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(directory / "pyvale-output" / output))
            _, coordinates, _ = gmsh.model.mesh.getNodes()
            coordinates = np.asarray(coordinates).reshape(-1, 3)
            material_names = {
                gmsh.model.getPhysicalName(dimension, tag)
                for dimension, tag in gmsh.model.getPhysicalGroups(3)
            }
            element_types, element_tags, _ = gmsh.model.mesh.getElements(3)
            assert sum(len(tags) for tags in element_tags) > 0
            if example.endswith("ex03d_gmsh_plate3d.py"):
                np.testing.assert_allclose(coordinates.min(axis=0), 0, atol=1e-12)
                np.testing.assert_allclose(
                    coordinates.max(axis=0), [0.1, 0.15, 0.005], atol=1e-12
                )
                assert material_names == {"plate"}
                boundary_names = {
                    gmsh.model.getPhysicalName(dimension, tag)
                    for dimension, tag in gmsh.model.getPhysicalGroups(2)
                }
                assert boundary_names == {"bc-base", "bc-top"}
                assert all(
                    gmsh.model.mesh.getElementProperties(int(kind))[0]
                    == "Hexahedron 27"
                    for kind in element_types
                )
            else:
                np.testing.assert_allclose(
                    coordinates.min(axis=0), [-0.0125, 0, 0], atol=1e-12
                )
                np.testing.assert_allclose(
                    coordinates.max(axis=0), [0.0125, 0.033, 0.012], atol=1e-12
                )
                assert material_names == {"armour-w", "interlayer-cu", "pipe-cucrzr"}
                assert all(
                    gmsh.model.mesh.getElementProperties(int(kind))[0]
                    == "Hexahedron 20"
                    for kind in element_types
                )
            for _, physical_tag in gmsh.model.getPhysicalGroups(3):
                entities = gmsh.model.getEntitiesForPhysicalGroup(3, physical_tag)
                assert sum(
                    len(tags)
                    for entity in entities
                    for tags in gmsh.model.mesh.getElements(3, int(entity))[1]
                ) > 0
        finally:
            gmsh.finalize()

    if example.endswith("ex01b_quickstart_simdata.py"):
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

    if example.endswith("ex05a_femu.py"):
        import numpy as np

        import pyvale.mooseherder as mh

        root = directory / "pyvale-output/mooseherder_ex05a_femu"
        with mh.ExodusLoader(root / "evaluations/fit_0/solve/results.e") as loader:
            fitted = loader.load_all_sim_data()
        with np.load(root / "identification.npz") as saved:
            material_params = saved["material_params"]
            assert material_params.shape == (2,)
            assert abs(material_params[0] / 108.0 - 1) < 0.02
            assert abs(material_params[1] - 0.33) < 0.005
            assert saved["convergence"].shape == (24,)
            assert float(saved["cost"]) < saved["convergence"][0]
            assert np.all(np.diff(saved["convergence"]) <= 0)
            predicted_force_n = float(fitted.glob_vars["react_y_top"][-1]) * float(
                saved["thickness_mm"]
            )
            predicted_strain = np.column_stack(
                (
                    fitted.node_vars["strain_xx"][:, -1],
                    fitted.node_vars["strain_yy"][:, -1],
                )
            )
            assert predicted_strain.shape == saved["observed_strain"].shape
            assert predicted_strain.shape[1] == 2
            force_cost = (
                (predicted_force_n - float(saved["observed_force_n"]))
                / float(saved["force_noise_n"])
            ) ** 2
            strain_cost = np.mean(
                (predicted_strain - saved["observed_strain"]) ** 2
                / float(saved["strain_noise"]) ** 2
            )
            np.testing.assert_allclose(
                float(saved["cost"]), 0.5 * (force_cost + strain_cost), rtol=1e-8
            )
        assert (root / "convergence.png").stat().st_size > 0
        assert {
            path.name
            for path in (root / "evaluations").iterdir()
            if path.is_dir() and not path.name.startswith(".")
        } == {f"fit_{index}" for index in range(12)}
        assert len(list((root / "evaluations").glob("*/solve/results.e"))) == 12
