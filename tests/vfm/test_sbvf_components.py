from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.testing as nptest

from pyvale.vfm.identification_nonlinear import _build_least_squares_residual_vector
from pyvale.vfm.metric_sensitivity_based_vf import SensitivityBasedVFMetric
from pyvale.vfm.mechanical_properties import (
    ConstituitiveLaw,
    KnownParameter,
    MechanicalProperties,
    ParameterName,
)
from pyvale.vfm.parameterisation_homogeneous import build_homogeneous_parameterisation
from pyvale.vfm.project_definition import (
    ParameterDefinition,
    ParameterisationSpec,
    TestData,
)
from pyvale.vfm.metrics import MetricResult
from pyvale.vfm.mat_to_py_data_parser import load_parsed_test_data, save_parsed_test_data
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.sensitivity_based_virtual_fields import (
    generate_sensitivity_based_virtual_fields,
)
from pyvale.vfm.spatial_parameterisation import ParameterState, resolve_parameter_maps
from pyvale.vfm.stress_sensitivity import (
    StressSensitivity,
    calculate_stress_sensitivity,
)
from pyvale.vfm.virtual_fields_mesh import (
    _compute_glyph_half_size_from_spacing,
    generate_virtual_fields_mesh,
)


def _build_small_test_data() -> TestData:
    x = np.tile(np.array([[0.5, 1.5]], dtype=np.float64), (2, 1))
    y = np.tile(np.array([[0.5], [1.5]], dtype=np.float64), (1, 2))
    specimen_mask = np.ones((2, 2), dtype=bool)
    area = np.ones((2, 2), dtype=np.float64)

    strain = np.zeros((3, 3, 2, 2), dtype=np.float64)
    strain[1, 0, :, :] = 1.0e-3
    strain[2, 0, :, :] = 2.0e-3
    strain[:, 2, :, :] = 2.0e-4

    force = np.zeros((3, 2), dtype=np.float64)
    time = np.array([0.1, 0.2, 0.5], dtype=np.float64)

    return TestData(
        x=x,
        y=y,
        specimen_mask=specimen_mask,
        area=area,
        strain=strain,
        force=force,
        time=time,
        thickness=1.8,
    )


def _build_small_virtual_fields_mesh(
    settings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    x = np.array([0.5, 1.5], dtype=np.float64)
    y = np.array([0.5, 1.5], dtype=np.float64)
    indices = np.arange(4, dtype=np.uint32)
    mesh = generate_virtual_fields_mesh(
        x=x,
        y=y,
        indices=indices,
        settings=settings,
        mesh_size=np.array([1, 1], dtype=np.uint32),
    )
    return x, y, indices, mesh


def test_compute_glyph_half_size_from_spacing_uses_smallest_adjacent_spacing() -> None:
    grid_x = np.array([[0.0, 2.0, 5.0], [0.0, 2.0, 5.0]], dtype=np.float64)
    grid_y = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]], dtype=np.float64)

    nptest.assert_allclose(
        _compute_glyph_half_size_from_spacing(grid_x, axis=1),
        0.3,
    )
    nptest.assert_allclose(
        _compute_glyph_half_size_from_spacing(grid_y, axis=0),
        0.6,
    )


def _build_parameter_state(
    parameter_name: EParameterName,
    initial_value: float,
    lower_bound: float,
    upper_bound: float,
    kind: str,
) -> ParameterState:
    parameter_definition = ParameterDefinition(
        name=parameter_name,
        initial_value_type="float",
        initial_value=initial_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    parameterisation = build_homogeneous_parameterisation(
        parameter_name=parameter_name.name,
        parameter_definition=parameter_definition,
        parameterisation_spec=ParameterisationSpec(kind=kind),
        previous_result=None,
    )
    return ParameterState(
        parameter_name=parameter_name,
        parameterisations=[parameterisation],
    )


def _build_resolved_properties(
    base_mechanical_properties: MechanicalProperties,
    parameter_states: dict[str, ParameterState],
    test_data: TestData,
) -> MechanicalProperties:
    resolved_parameters = dict(base_mechanical_properties.parameters)

    for parameter_name, parameter_map in resolve_parameter_maps(
        parameter_states,
        test_data,
    ).items():
        resolved_parameters[ParameterName[parameter_name]] = KnownParameter(parameter_map)

    return MechanicalProperties(
        constituitive_law=base_mechanical_properties.constituitive_law,
        parameters=resolved_parameters,
    )


def _reshape_measured_values(
    values: np.ndarray,
    indices: np.ndarray,
    size_y: int,
    size_x: int,
) -> np.ndarray:
    flat = np.full(size_y * size_x, np.nan, dtype=np.float64)
    flat[indices] = values
    return flat.reshape((size_y, size_x), order="F")


def test_generate_virtual_fields_mesh_reconstructs_affine_fields() -> None:
    x, y, _, mesh = _build_small_virtual_fields_mesh(
        settings=np.zeros((2, 4), dtype=np.uint32),
    )

    node_x = mesh.x
    node_y = mesh.y
    u_nodes = 2.0 * node_x + 3.0 * node_y + 1.0
    v_nodes = -1.0 * node_x + 4.0 * node_y - 2.0

    virtual_displacement = np.empty(mesh.b_glob.shape[1], dtype=np.float64)
    virtual_displacement[0::2] = u_nodes
    virtual_displacement[1::2] = v_nodes

    reconstructed_strain = mesh.b_glob @ virtual_displacement
    num_measured_points = mesh.indices.size

    nptest.assert_allclose(reconstructed_strain[:num_measured_points], 2.0)
    nptest.assert_allclose(reconstructed_strain[num_measured_points : 2 * num_measured_points], 4.0)
    nptest.assert_allclose(reconstructed_strain[2 * num_measured_points :], 2.0)
    nptest.assert_allclose(mesh.n_glob.sum(axis=1), 1.0)

    grid_x, grid_y = np.meshgrid(x, y)
    x_points = grid_x.flatten(order="F")
    y_points = grid_y.flatten(order="F")

    nptest.assert_allclose(mesh.n_glob @ u_nodes, 2.0 * x_points + 3.0 * y_points + 1.0)
    nptest.assert_allclose(mesh.n_glob @ v_nodes, -1.0 * x_points + 4.0 * y_points - 2.0)


def test_calculate_stress_sensitivity_matches_explicit_perturbation() -> None:
    test_data = _build_small_test_data()

    parameter_states = {
        ParameterName.ElasticModulus.name: _build_parameter_state(
            ParameterName.ElasticModulus,
            initial_value=210.0e3,
            lower_bound=100.0e3,
            upper_bound=300.0e3,
            kind="known",
        ),
        ParameterName.PoissonsRatio.name: _build_parameter_state(
            ParameterName.PoissonsRatio,
            initial_value=0.3,
            lower_bound=0.2,
            upper_bound=0.4,
            kind="known",
        ),
        ParameterName.YieldStrength.name: _build_parameter_state(
            ParameterName.YieldStrength,
            initial_value=320.0,
            lower_bound=100.0,
            upper_bound=1000.0,
            kind="homogeneous",
        ),
        ParameterName.HardeningModulus.name: _build_parameter_state(
            ParameterName.HardeningModulus,
            initial_value=3000.0,
            lower_bound=1000.0,
            upper_bound=10000.0,
            kind="known",
        ),
    }

    for parameter_state in parameter_states.values():
        parameter_state.initialise_from_map(test_data, None)
        parameter_state.prepare(test_data)

    active_dofs = parameter_states[ParameterName.YieldStrength.name].collect_dofs()
    assert len(active_dofs) == 1

    base_mechanical_properties = MechanicalProperties(
        constituitive_law=ConstituitiveLaw.LinearHardening,
        parameters={
            ParameterName.ElasticModulus: KnownParameter(210.0e3),
            ParameterName.PoissonsRatio: KnownParameter(0.3),
            ParameterName.YieldStrength: KnownParameter(320.0),
            ParameterName.HardeningModulus: KnownParameter(3000.0),
        },
    )

    reference_properties = _build_resolved_properties(
        base_mechanical_properties,
        parameter_states,
        test_data,
    )
    stress_reference, _, _, _ = radial_return(test_data.strain, reference_properties)

    sensitivities = calculate_stress_sensitivity(
        stress_reference=stress_reference,
        test_data=test_data,
        base_mechanical_properties=base_mechanical_properties,
        parameter_states=parameter_states,
        active_dofs=active_dofs,
        perturbation_factor=0.15,
    )

    perturbed_states = deepcopy(parameter_states)
    perturbed_dof = perturbed_states[EParameterName.YieldStrength.name].collect_dofs()[0]
    perturbed_dof.value = float(
        np.clip(
            perturbed_dof.value * 0.85,
            perturbed_dof.lower_bound,
            perturbed_dof.upper_bound,
        )
    )

    perturbed_properties = _build_resolved_properties(
        base_mechanical_properties,
        perturbed_states,
        test_data,
    )
    perturbed_stress, _, _, _ = radial_return(test_data.strain, perturbed_properties)

    expected_total = stress_reference - perturbed_stress
    expected_incremental = np.zeros_like(expected_total)
    expected_incremental[1:, :, :, :] = np.diff(expected_total, axis=0)

    timestep_deltas = np.array([0.1, 0.1, 0.3], dtype=np.float64)
    expected_incremental = (
        expected_incremental / timestep_deltas[:, np.newaxis, np.newaxis, np.newaxis]
    )

    sensitivity = sensitivities[active_dofs[0].uid]
    nptest.assert_allclose(sensitivity.total, expected_total)
    nptest.assert_allclose(sensitivity.incremental, expected_incremental)


def test_generate_sensitivity_based_virtual_fields_reconstructs_round_trip() -> None:
    _, _, indices, mesh = _build_small_virtual_fields_mesh(
        settings=np.array([[0, 1, 0, 2], [0, 1, 0, 1]], dtype=np.uint32),
    )

    assert mesh.act_dofs.size == 1

    virtual_displacement = np.zeros(mesh.b_glob.shape[1], dtype=np.float64)
    virtual_displacement[mesh.act_dofs[0]] = 0.25

    right_edge_nodes = mesh.virtual_elements[:, -1]
    right_master_node = mesh.virtual_elements[-1, -1]
    right_slave_nodes = right_edge_nodes[:-1]
    virtual_displacement[2 * right_slave_nodes] = virtual_displacement[2 * right_master_node]

    target_strain_vector = mesh.b_glob @ virtual_displacement
    sensitivity_map = np.empty((1, 3, 2, 2), dtype=np.float64)

    for component in range(3):
        start = component * indices.size
        stop = (component + 1) * indices.size
        sensitivity_map[0, component, :, :] = _reshape_measured_values(
            target_strain_vector[start:stop],
            indices,
            size_y=2,
            size_x=2,
        )

    virtual_fields = generate_sensitivity_based_virtual_fields(
        stress_sensitivities={
            "dof": StressSensitivity(
                total=sensitivity_map,
                incremental=np.zeros_like(sensitivity_map),
            )
        },
        virtual_fields_mesh=mesh,
        use_incremental=False,
    )
    sbvf = virtual_fields["dof"]

    expected_x = _reshape_measured_values(
        mesh.n_glob @ virtual_displacement[0::2],
        indices,
        size_y=2,
        size_x=2,
    )
    expected_y = _reshape_measured_values(
        mesh.n_glob @ virtual_displacement[1::2],
        indices,
        size_y=2,
        size_x=2,
    )

    nptest.assert_allclose(sbvf.virtual_strain, sensitivity_map)
    nptest.assert_allclose(sbvf.full_displacement[0, 0, :, :], expected_x)
    nptest.assert_allclose(sbvf.full_displacement[0, 1, :, :], expected_y)
    nptest.assert_allclose(
        sbvf.edge_displacement,
        np.array([[[0.125, 0.0, 0.125, 0.25], [0.0, 0.0, 0.0, 0.0]]]),
    )


def test_sbvf_metric_prepare_can_write_virtual_mesh_plot(tmp_path) -> None:
    plot_path = tmp_path / "virtual_mesh.png"
    metric = SensitivityBasedVFMetric(
        options={
            "virtual_mesh_size": [1, 1],
            "boundary_settings": [[0, 1, 0, 2], [0, 1, 0, 1]],
            "plot_virtual_mesh": True,
            "plot_virtual_mesh_path": str(plot_path),
        }
    )

    metric.prepare(_build_small_test_data())

    assert metric.virtual_fields_mesh is not None
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_least_squares_residual_vector_uses_metric_residuals_and_weights() -> None:
    metric_results = [
        MetricResult(
            name="sbvf",
            value=5.0,
            details={"residual_vector": np.array([1.0, -2.0])},
        ),
        MetricResult(
            name="fallback",
            value=9.0,
            details={},
        ),
    ]

    residual_vector = _build_least_squares_residual_vector(
        metrics_with_weights=[(object(), 4.0), (object(), 0.25)],
        metric_results=metric_results,
    )

    nptest.assert_allclose(residual_vector, np.array([2.0, -4.0, 1.5]))


def test_parsed_test_data_round_trips_thickness(tmp_path) -> None:
    source = _build_small_test_data()
    output_path = tmp_path / "test_data.npz"

    save_parsed_test_data(source, output_path)
    loaded = load_parsed_test_data(output_path)

    assert loaded.thickness == source.thickness
