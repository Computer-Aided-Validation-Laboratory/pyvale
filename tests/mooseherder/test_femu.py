"""Case17 reaction units and modulus sensitivity used by the FEMU tutorial."""

from pathlib import Path

import numpy as np

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh


def test_plate_hole_modulus_sensitivity(tmp_path: Path) -> None:
    mesh = mh.GmshRunner().run(
        dataset.sim_case_gmsh_file_path(17),
        output_dir=tmp_path / "mesh",
    )
    fields: list[io.SimData] = []
    for modulus_mpa in (80e3, 120e3):
        result = mh.MooseRunner().run(
            dataset.sim_case_input_file_path(17),
            parameters={
                "Mesh/file": str(mesh.output),
                "EMod": modulus_mpa,
                "endTime": 1,
                "timeStep": 1,
            },
            output_dir=tmp_path / f"E-{modulus_mpa}",
        )
        reader = mh.ExodusLoader(result.output)
        try:
            fields.append(reader.load_all_sim_data())
        finally:
            reader.close()
    for field in fields:
        assert field.glob_vars["react_y_top"][-1] > 0
        assert field.glob_vars["react_y_bot"][-1] < 0
        np.testing.assert_allclose(
            field.glob_vars["react_y_top"][-1],
            -field.glob_vars["react_y_bot"][-1],
            rtol=1e-6,
        )
        # Original benchmark coordinates/displacements are mm, not metres.
        np.testing.assert_allclose(field.coords[:, :2].max(axis=0), [100, 150])
        np.testing.assert_allclose(field.glob_vars["disp_y_max"][-1], 1)
    np.testing.assert_array_equal(fields[0].coords, fields[1].coords)
    np.testing.assert_allclose(
        fields[1].glob_vars["react_y_top"][-1] / fields[0].glob_vars["react_y_top"][-1],
        120 / 80,
        rtol=1e-8,
    )
    # At fixed Poisson ratio and prescribed displacement, E changes force,
    # but not the strain field: the field residual cannot identify E alone.
    np.testing.assert_allclose(
        fields[0].node_vars["strain_yy"][:, -1],
        fields[1].node_vars["strain_yy"][:, -1],
        atol=1e-10,
        rtol=1e-8,
    )


def test_plate_hole_strains_identify_poisson_ratio(tmp_path: Path) -> None:
    """Changing nu at fixed E changes lateral contraction and the axial field."""
    mesh = mh.GmshRunner().run(
        dataset.sim_case_gmsh_file_path(17),
        output_dir=tmp_path / "mesh",
    )
    strains = []
    for poisson_ratio in (0.25, 0.4):
        result = mh.MooseRunner().run(
            dataset.sim_case_input_file_path(17),
            parameters={
                "Mesh/file": str(mesh.output),
                "EMod": 108e3,
                "PRatio": poisson_ratio,
                "endTime": 1,
                "timeStep": 1,
            },
            output_dir=tmp_path / f"nu-{poisson_ratio}",
        )
        reader = mh.ExodusLoader(result.output)
        try:
            data = reader.load_all_sim_data()
        finally:
            reader.close()
        strains.append(
            np.column_stack(
                (
                    data.node_vars["strain_xx"][:, -1],
                    data.node_vars["strain_yy"][:, -1],
                )
            )
        )
    difference = strains[1] - strains[0]
    # Compare with the tutorial's 50-microstrain noise floor; these are real
    # finite-element fields, not a cost-function or optimiser approximation.
    assert np.sqrt(np.mean(difference[:, 0] ** 2)) > 5 * 50e-6
    assert np.sqrt(np.mean(difference[:, 1] ** 2)) > 10e-6
    assert np.mean(strains[1][:, 0]) < np.mean(strains[0][:, 0])


def test_existing_plate_with_python_mesh(tmp_path: Path) -> None:
    """The alternative Python mesh uses the original case17 geometry and BCs."""
    input_path = dataset.sim_case_input_file_path(17)
    mesh = mh.GmshRunner().run(
        input_path.with_name("case17_mesh.py"),
        parameters={"mesh_size_mm": 20.0},
        output_dir=tmp_path / "mesh",
    )
    result = mh.MooseRunner().run(
        input_path,
        output_dir=tmp_path / "solve",
        parameters={"Mesh/file": str(mesh.output), "endTime": 1, "timeStep": 1},
    )
    reader = mh.ExodusLoader(result.output)
    try:
        simulation = reader.load_all_sim_data()
    finally:
        reader.close()
    np.testing.assert_allclose(simulation.coords[:, :2].max(axis=0), [100, 150])
    np.testing.assert_allclose(simulation.coords[:, :2].min(axis=0), [0, 0])
    radii = np.linalg.norm(simulation.coords[:, :2] - np.array([50, 75]), axis=1)
    assert radii.min() >= 12.5 - 1e-8
    assert simulation.glob_vars["react_y_top"][-1] > 0
    assert simulation.glob_vars["react_y_bot"][-1] < 0
    np.testing.assert_allclose(simulation.glob_vars["disp_y_max"][-1], 1)
    assert "strain_xx" in simulation.node_vars
    assert "strain_yy" in simulation.node_vars
