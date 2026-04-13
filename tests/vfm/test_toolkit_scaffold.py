from __future__ import annotations

import numpy as np
from scipy.io import savemat

from pyvale.vfm.mechanical_properties import ConstituitiveLaw, ParameterName
from pyvale.vfm.mat_to_py_data_parser import (
    convert_mat_to_py_data,
    load_parsed_test_data,
    parse_test_data_from_mat,
)
from pyvale.vfm.metric_sensitivity_based_vf import build_sensitivity_based_vf_metric
from pyvale.vfm.metrics import MetricContext
from pyvale.vfm.project_definition import (
    MetricSpec,
    ParameterDefinition,
    ParameterisationSpec,
    PhaseResult,
    TestData,
    create_default_project,
)
from pyvale.vfm.project_io import load_project, save_project
from pyvale.vfm.spatial_parameterisation import (
    ParameterState,
    build_parameter_state,
    collect_active_dofs,
    resolve_parameter_maps,
)


def _make_small_test_data() -> TestData:
    x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    specimen_mask = np.array([[True, True], [True, False]], dtype=bool)
    area = np.ones((2, 2), dtype=np.float64)
    strain = np.zeros((1, 3, 2, 2), dtype=np.float64)
    force = np.zeros((1, 2), dtype=np.float64)
    time = np.array([0.0], dtype=np.float64)
    return TestData(x=x, y=y, specimen_mask=specimen_mask, area=area, strain=strain, force=force, time=time)


def test_project_yaml_round_trip(tmp_path) -> None:
    project = create_default_project(ConstituitiveLaw.LinearHardening)
    project.name = "demo"
    project.test_data_path = tmp_path / "testData.mat"
    project.parameters[ParameterName.ElasticModulus.name].initial_value_type = "2d np array"
    project.parameters[ParameterName.ElasticModulus.name].initial_value = "maps/elastic_modulus.npy"
    project.phases[0].parameterisations[ParameterName.YieldStrength.name].append(
        ParameterisationSpec(
            kind="basis_function",
            options={"kernel_shape": "univariate", "initial_count": 2},
        )
    )

    project_path = tmp_path / "project.yaml"
    save_project(project, project_path)
    reloaded = load_project(project_path)

    assert reloaded.name == "demo"
    assert reloaded.constituitive_law is ConstituitiveLaw.LinearHardening
    assert reloaded.parameters[ParameterName.ElasticModulus.name].initial_value_type == "2d np array"
    assert reloaded.parameters[ParameterName.ElasticModulus.name].initial_value == "maps/elastic_modulus.npy"
    assert len(reloaded.phases[0].parameterisations[ParameterName.YieldStrength.name]) == 2
    assert (
        reloaded.phases[0].parameterisations[ParameterName.YieldStrength.name][1].kind
        == "basis_function"
    )


def test_default_linear_hardening_project_uses_expected_defaults() -> None:
    project = create_default_project(ConstituitiveLaw.LinearHardening)

    assert project.parameters[ParameterName.ElasticModulus.name].initial_value == 190.0e3
    assert project.parameters[ParameterName.ElasticModulus.name].initial_value_type == "float"
    assert project.parameters[ParameterName.ElasticModulus.name].lower_bound == 150.0e3
    assert project.parameters[ParameterName.ElasticModulus.name].upper_bound == 250.0e3
    assert project.parameters[ParameterName.PoissonsRatio.name].initial_value == 0.28
    assert len(project.phases) == 1
    assert project.phases[0].metrics[0].kind == "sbvf"
    assert project.phases[0].optimiser.kind == "least_squares"


def test_homogeneous_parameter_state_resolves_map() -> None:
    test_data = _make_small_test_data()
    parameter_definition = ParameterDefinition(
        name=ParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[ParameterisationSpec(kind="homogeneous")],
    )

    parameter_maps = resolve_parameter_maps(
        {ParameterName.YieldStrength.name: parameter_state},
        test_data,
    )

    yield_map = parameter_maps[ParameterName.YieldStrength.name]
    assert yield_map.shape == (2, 2)
    assert np.isclose(yield_map[0, 0], 320.0)
    assert np.isnan(yield_map[1, 1])


def test_known_parameter_state_loads_npy_initial_map(tmp_path) -> None:
    test_data = _make_small_test_data()
    map_path = tmp_path / "yield_strength.npy"
    np.save(map_path, np.array([[300.0, 310.0], [320.0, 330.0]], dtype=np.float64))

    parameter_definition = ParameterDefinition(
        name=ParameterName.YieldStrength,
        initial_value_type="2d np array",
        initial_value=str(map_path),
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[ParameterisationSpec(kind="known")],
    )

    parameter_map = parameter_state.to_map(test_data)

    assert np.isclose(parameter_map[0, 0], 300.0)
    assert np.isclose(parameter_map[1, 0], 320.0)
    assert np.isnan(parameter_map[1, 1])


def test_linked_parameterisation_only_unfreezes_requested_groups() -> None:
    parameter_definition = ParameterDefinition(
        name=ParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    previous_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[ParameterisationSpec(kind="homogeneous")],
    )
    previous_result = PhaseResult(
        phase_name="phase_1",
        cost=0.0,
        metric_values={},
        parameter_maps={},
        parameter_states={ParameterName.YieldStrength.name: previous_state},
    )

    linked_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[
            ParameterisationSpec(
                kind="linked",
                free_dof_groups=["value"],
            )
        ],
        previous_result=previous_result,
    )

    dofs = linked_state.collect_dofs()
    assert len(dofs) == 1
    assert dofs[0].active is True


def test_basis_function_parameterisation_resolves_gaussian_map() -> None:
    test_data = _make_small_test_data()
    parameter_definition = ParameterDefinition(
        name=ParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[
            ParameterisationSpec(
                kind="basis_function",
                options={
                    "kernel_shape": "univariate",
                    "active_groups": ["rbf_heights"],
                    "kernels": [
                        {
                            "x": 0.0,
                            "y": 0.0,
                            "height": 10.0,
                            "height_lower_bound": 0.0,
                            "height_upper_bound": 20.0,
                            "variance_x": 0.25,
                            "variance_lower_bound": 0.01,
                            "variance_upper_bound": 2.0,
                        }
                    ],
                },
            )
        ],
    )

    dofs = parameter_state.collect_dofs()
    active_dofs = collect_active_dofs({ParameterName.YieldStrength.name: parameter_state})
    parameter_map = parameter_state.to_map(test_data)

    assert len(dofs) == 4
    assert len(active_dofs) == 1
    assert parameter_map[0, 0] > parameter_map[0, 1]
    assert parameter_map[0, 0] > parameter_map[1, 0]
    assert np.isnan(parameter_map[1, 1])


def test_basis_function_parameterisation_can_build_default_kernels_from_count() -> None:
    test_data = _make_small_test_data()
    parameter_definition = ParameterDefinition(
        name=ParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=ParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[
            ParameterisationSpec(
                kind="basis_function",
                options={
                    "kernel_shape": "univariate",
                    "initial_count": 2,
                },
            )
        ],
    )

    parameter_state.prepare(test_data)
    dofs = parameter_state.collect_dofs()
    parameter_map = parameter_state.to_map(test_data)

    assert len(dofs) == 8
    assert np.isfinite(parameter_map[0, 0])
    assert np.isnan(parameter_map[1, 1])


def test_sbvf_metric_zero_active_dofs_returns_zero_cost() -> None:
    test_data = _make_small_test_data()
    stress = np.zeros((1, 3, 2, 2), dtype=np.float64)
    metric = build_sensitivity_based_vf_metric(
        MetricSpec(
            kind="sbvf",
            options={
                "virtual_mesh_size": [1, 1],
                "boundary_settings": [[0, 1, 0, 2], [0, 1, 0, 1]],
            },
        )
    )

    metric.prepare(test_data, MetricContext())
    result = metric.evaluate(
        stress,
        test_data,
        MetricContext(
            base_mechanical_properties=None,
            parameter_states={},
            active_dofs=[],
        ),
    )

    assert result.value == 0.0


def test_saved_project_yaml_has_extra_spacing_for_readability(tmp_path) -> None:
    project = create_default_project(ConstituitiveLaw.LinearHardening)
    project_path = tmp_path / "project.yaml"

    save_project(project, project_path)
    yaml_text = project_path.read_text(encoding="utf-8")

    assert "\n\nconstituitive_law:" in yaml_text
    assert "\n\nphases:\n" in yaml_text


def test_mat_to_py_data_parser_rearranges_and_round_trips(tmp_path) -> None:
    x = np.array([[0.0, 1.0], [0.0, np.nan]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [1.0, np.nan]], dtype=np.float64)
    area = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    c11 = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float64,
    )
    c22 = c11 + 100.0
    c12 = c11 + 200.0
    force = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    time = np.array([0.1, 0.2], dtype=np.float64)

    mat_path = tmp_path / "testData.mat"
    savemat(
        mat_path,
        {
            "testData": {
                "X": x,
                "Y": y,
                "area": area,
                "FGlob": force,
                "time": {"time": time},
                "strain": {
                    "c11": c11,
                    "c22": c22,
                    "c12": c12,
                },
            }
        },
    )

    parsed = parse_test_data_from_mat(mat_path)
    npz_path = convert_mat_to_py_data(mat_path)
    reloaded = load_parsed_test_data(npz_path)

    assert parsed.x.shape == (2, 2)
    assert parsed.y.shape == (2, 2)
    assert parsed.area.shape == (2, 2)
    assert parsed.strain.shape == (2, 3, 2, 2)
    assert parsed.specimen_mask.tolist() == [[True, False], [True, True]]
    assert np.isclose(parsed.strain[0, 0, 0, 0], 2.0)
    assert np.isclose(parsed.strain[1, 2, 1, 0], 210.0)
    assert np.array_equal(reloaded.force, force)
    assert np.array_equal(reloaded.time, time)
    assert np.allclose(reloaded.x, parsed.x, equal_nan=True)
