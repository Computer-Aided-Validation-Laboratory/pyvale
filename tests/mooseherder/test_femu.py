# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Case17 reaction units and modulus sensitivity used by the FEMU tutorial."""

from pathlib import Path

import numpy as np
import pytest

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
            input_params={
                "Mesh/file": str(mesh.output_file),
                "EMod": modulus_mpa,
                "endTime": 1,
                "timeStep": 1,
            },
            output_dir=tmp_path / f"E-{modulus_mpa}",
        )
        with mh.ExodusLoader(result.output_file) as reader:
            fields.append(reader.load_all_sim_data())
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
            input_params={
                "Mesh/file": str(mesh.output_file),
                "EMod": 108e3,
                "PRatio": poisson_ratio,
                "endTime": 1,
                "timeStep": 1,
            },
            output_dir=tmp_path / f"nu-{poisson_ratio}",
        )
        with mh.ExodusLoader(result.output_file) as reader:
            data = reader.load_all_sim_data()
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
        input_params={"mesh_size_mm": 20.0},
        output_dir=tmp_path / "mesh",
    )
    result = mh.MooseRunner().run(
        input_path,
        output_dir=tmp_path / "solve",
        input_params={"Mesh/file": str(mesh.output_file), "endTime": 1, "timeStep": 1},
    )
    with mh.ExodusLoader(result.output_file) as reader:
        simulation = reader.load_all_sim_data()
    np.testing.assert_allclose(simulation.coords[:, :2].max(axis=0), [100, 150])
    np.testing.assert_allclose(simulation.coords[:, :2].min(axis=0), [0, 0])
    radii = np.linalg.norm(simulation.coords[:, :2] - np.array([50, 75]), axis=1)
    assert radii.min() >= 12.5 - 1e-8
    assert simulation.glob_vars["react_y_top"][-1] > 0
    assert simulation.glob_vars["react_y_bot"][-1] < 0
    np.testing.assert_allclose(simulation.glob_vars["disp_y_max"][-1], 1)
    assert "strain_xx" in simulation.node_vars
    assert "strain_yy" in simulation.node_vars


def test_femu_objective_submits_candidates_as_one_parallel_batch(
    input_tree: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The example adapter runs two cases concurrently and preserves cost order."""
    import ast
    import json
    import time
    from collections.abc import Callable, Iterator
    from contextlib import contextmanager
    from functools import partial
    from threading import Barrier
    from types import SimpleNamespace

    from pymoo.core.algorithm import Algorithm
    from pymoo.core.problem import Problem

    barrier = Barrier(2)

    class ReferenceRunner(mh.SimRunner):
        def run(
            self,
            input_files: mh.InputFiles,
            *,
            output_dir: Path,
            input_params: dict[str, mh.InputValues] | None = None,
            input_root: Path | None = None,
        ) -> mh.RunResult:
            # A sequential call cannot reach the second barrier participant.
            barrier.wait(timeout=5)
            output_dir.mkdir(parents=True)
            output_file = output_dir / "prediction.json"
            output_file.write_text(json.dumps(input_params))
            return mh.RunResult(output_file, output_dir, (), output_dir / "run.log")

    @contextmanager
    def load_prediction(output_file: Path) -> Iterator[SimpleNamespace]:
        input_params = json.loads(output_file.read_text())
        poisson_ratio = input_params["PRatio"]
        simulation = io.SimData(
            coords=np.zeros((1, 3)),
            glob_vars={"react_y_top": np.array([input_params["EMod"] * 1e-3])},
            node_vars={
                "strain_xx": np.array([[poisson_ratio]]),
                "strain_yy": np.array([[2 * poisson_ratio]]),
            },
        )
        yield SimpleNamespace(load_all_sim_data=lambda: simulation)

    monkeypatch.setattr(mh, "ExodusLoader", load_prediction)
    example = (
        Path(__file__).resolve().parents[2]
        / "src/pyvale/examples"
        / "mooseherder/ex05a_femu.py"
    )
    definitions = [
        node
        for node in ast.parse(example.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "np": np,
        "time": time,
        "mh": mh,
        "io": io,
        "Callable": Callable,
        "Problem": Problem,
        "Algorithm": Algorithm,
        "N_GENERATIONS": 24,
        "FORCE_NOISE_N": 100.0,
        "STRAIN_NOISE": 1.0,
        "THICKNESS_MM": 1.0,
    }
    exec(  # noqa: S102 - compile trusted example definitions without running FEM solves.
        compile(ast.Module(body=definitions, type_ignores=[]), str(example), "exec"),
        namespace,
    )
    herd = mh.MooseHerd(
        [mh.SimStep("solve", ReferenceRunner(), input_tree / "model.i")],
        dir_manager=mh.DirectoryManager(tmp_path / "fits", mode=mh.EDirMode.REUSE),
        parallel_runners=2,
    )
    objective = partial(
        namespace["calc_cost"],
        herd=herd,
        base_input_params={},
        observed_force_n=0.0,
        observed_strain=np.zeros((1, 2)),
        observed_coords=np.zeros((1, 3)),
    )
    problem = namespace["FemuProblem"](objective)
    costs = problem.evaluate(
        np.array([[100.0, 0.2], [120.0, 0.4]]), algorithm=SimpleNamespace(n_gen=3)
    )
    assert "Generation 3/24 with 2 particles took " in capsys.readouterr().out
    np.testing.assert_allclose(costs, [[0.55], [0.92]])
    assert (tmp_path / "fits/fit_0/solve/prediction.json").is_file()
    assert (tmp_path / "fits/fit_1/solve/prediction.json").is_file()
