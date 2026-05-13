from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyvale.vfm.identification_nonlinear import run_nonlinear_identification
from pyvale.vfm.mechanical_properties import (
    EConstituitiveLaw,
    EParameterName,
    KnownParameter,
    MechanicalProperties,
)
from pyvale.vfm.metrics import MetricContext
from pyvale.vfm.metric_udvf_slicewise import build_udvf_slicewise_metric
from pyvale.vfm.project_definition import (
    BoundaryConditions,
    EdgeBoundaryCondition,
    EEdgeBoundaryCondition,
    MetricSpec,
    OptimiserSpec,
    ParameterDefinition,
    ParameterisationSpec,
    PhaseDefinition,
    TestData,
)
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.spatial_parameterisation import build_parameter_state


def _make_boundary_conditions() -> BoundaryConditions:
    return BoundaryConditions(
        min_x_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FIXED,
            y=EEdgeBoundaryCondition.FREE,
        ),
        max_y_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FREE,
            y=EEdgeBoundaryCondition.FREE,
        ),
        max_x_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.TRACTION,
            y=EEdgeBoundaryCondition.FREE,
        ),
        min_y_edge=EdgeBoundaryCondition(
            x=EEdgeBoundaryCondition.FREE,
            y=EEdgeBoundaryCondition.FREE,
        ),
    )


def _make_structured_test_data(
    num_rows: int,
    num_cols: int,
    num_timesteps: int,
) -> TestData:
    x_line = np.arange(num_cols, dtype=np.float64)
    y_line = np.arange(num_rows, dtype=np.float64)
    x = np.tile(x_line, (num_rows, 1))
    y = np.tile(y_line[:, np.newaxis], (1, num_cols))

    return TestData(
        x=x,
        y=y,
        specimen_mask=np.ones((num_rows, num_cols), dtype=bool),
        area=np.ones((num_rows, num_cols), dtype=np.float64),
        strain=np.zeros((num_timesteps, 3, num_rows, num_cols), dtype=np.float64),
        force=np.zeros((num_timesteps, 2), dtype=np.float64),
        time=np.arange(num_timesteps, dtype=np.float64),
        thickness=1.0,
        boundary_conditions=_make_boundary_conditions(),
    )


def test_slicewise_parameterisation_builds_piecewise_constant_map() -> None:
    test_data = _make_structured_test_data(num_rows=2, num_cols=4, num_timesteps=1)
    parameter_definition = ParameterDefinition(
        name=EParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=EParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[
            ParameterisationSpec(
                kind="slicewise",
                options={"num_slices": 2, "constant_coordinate": "y"},
            )
        ],
    )

    parameter_state.prepare(test_data)
    dofs = parameter_state.collect_dofs()
    assert len(dofs) == 2

    dofs[0].value = 300.0
    dofs[1].value = 500.0
    parameter_map = parameter_state.to_map(test_data)

    expected = np.array(
        [
            [300.0, 300.0, 500.0, 500.0],
            [300.0, 300.0, 500.0, 500.0],
        ],
        dtype=np.float64,
    )
    npt.assert_allclose(parameter_map, expected, rtol=0.0, atol=0.0)


def test_udvf_slicewise_metric_returns_zero_for_exact_slice_balance() -> None:
    test_data = _make_structured_test_data(num_rows=1, num_cols=4, num_timesteps=2)
    test_data.force[:, 0] = np.array([10.0, 20.0], dtype=np.float64)

    parameter_definition = ParameterDefinition(
        name=EParameterName.YieldStrength,
        initial_value=320.0,
        lower_bound=100.0,
        upper_bound=1000.0,
    )
    parameter_state = build_parameter_state(
        parameter_name=EParameterName.YieldStrength.name,
        parameter_definition=parameter_definition,
        parameterisation_specs=[
            ParameterisationSpec(
                kind="slicewise",
                options={"num_slices": 2, "constant_coordinate": "y"},
            )
        ],
    )
    parameter_state.prepare(test_data)
    metric = build_udvf_slicewise_metric(
        MetricSpec(
            kind="udvf_slicewise",
            options={
                "virtual_component": "xx",
                "traction_edge": 3,
                "scaling": False,
            },
        )
    )
    metric.prepare(
        test_data,
        MetricContext(
            parameter_states={EParameterName.YieldStrength.name: parameter_state},
            active_dofs=parameter_state.collect_dofs(),
        ),
    )

    stress = np.zeros((2, 3, 1, 4), dtype=np.float64)
    stress[:, 0, :, :] = test_data.force[:, 0][:, np.newaxis, np.newaxis]

    result = metric.evaluate(stress, test_data, MetricContext())
    npt.assert_allclose(result.value, 0.0, rtol=0.0, atol=1.0e-12)
    npt.assert_allclose(
        result.details["residual_vector"],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_independent_slices_solver_recovers_single_slice_linear_hardening_parameters() -> None:
    true_yield_strength = 260.0
    true_hardening_modulus = 1800.0
    elastic_modulus = 190000.0
    poissons_ratio = 0.28

    test_data = _make_structured_test_data(num_rows=1, num_cols=2, num_timesteps=6)
    axial_strain = np.array(
        [0.0, 1.4e-3, 2.2e-3, 3.0e-3, 3.8e-3, 4.6e-3],
        dtype=np.float64,
    )
    test_data.strain[:, 0, 0, :] = axial_strain[:, np.newaxis]

    true_properties = MechanicalProperties(
        constituitive_law=EConstituitiveLaw.LinearHardening,
        parameters={
            EParameterName.ElasticModulus: KnownParameter(elastic_modulus),
            EParameterName.PoissonsRatio: KnownParameter(poissons_ratio),
            EParameterName.YieldStrength: KnownParameter(true_yield_strength),
            EParameterName.HardeningModulus: KnownParameter(true_hardening_modulus),
        },
    )
    true_stress, _, _, _ = radial_return(test_data.strain, true_properties)
    test_data.force[:, 0] = true_stress[:, 0, 0, 0]

    phase_definition = PhaseDefinition(
        name="slicewise_phase",
        parameterisations={
            EParameterName.ElasticModulus.name: [ParameterisationSpec(kind="known")],
            EParameterName.PoissonsRatio.name: [ParameterisationSpec(kind="known")],
            EParameterName.YieldStrength.name: [
                ParameterisationSpec(
                    kind="slicewise",
                    options={"num_slices": 1, "constant_coordinate": "y"},
                )
            ],
            EParameterName.HardeningModulus.name: [
                ParameterisationSpec(
                    kind="slicewise",
                    options={"num_slices": 1, "constant_coordinate": "y"},
                )
            ],
        },
        metrics=[
            MetricSpec(
                kind="udvf_slicewise",
                options={
                    "virtual_component": "xx",
                    "traction_edge": 3,
                    "scaling": False,
                },
            )
        ],
        optimiser=OptimiserSpec(
            kind="independent_slices",
            options={
                "local_solver": "least_squares",
                "method": "trf",
                "max_nfev": 80,
                "verbose": 0,
            },
        ),
    )

    parameter_definitions = {
        EParameterName.ElasticModulus.name: ParameterDefinition(
            name=EParameterName.ElasticModulus,
            initial_value=elastic_modulus,
            lower_bound=150000.0,
            upper_bound=250000.0,
        ),
        EParameterName.PoissonsRatio.name: ParameterDefinition(
            name=EParameterName.PoissonsRatio,
            initial_value=poissons_ratio,
            lower_bound=0.2,
            upper_bound=0.4,
        ),
        EParameterName.YieldStrength.name: ParameterDefinition(
            name=EParameterName.YieldStrength,
            initial_value=320.0,
            lower_bound=150.0,
            upper_bound=400.0,
        ),
        EParameterName.HardeningModulus.name: ParameterDefinition(
            name=EParameterName.HardeningModulus,
            initial_value=3000.0,
            lower_bound=500.0,
            upper_bound=4000.0,
        ),
    }

    parameter_states = {
        parameter_name: build_parameter_state(
            parameter_name=parameter_name,
            parameter_definition=parameter_definition,
            parameterisation_specs=phase_definition.parameterisations[parameter_name],
        )
        for parameter_name, parameter_definition in parameter_definitions.items()
    }

    base_mechanical_properties = MechanicalProperties(
        constituitive_law=EConstituitiveLaw.LinearHardening,
        parameters={
            EParameterName.ElasticModulus: KnownParameter(elastic_modulus),
            EParameterName.PoissonsRatio: KnownParameter(poissons_ratio),
            EParameterName.YieldStrength: KnownParameter(320.0),
            EParameterName.HardeningModulus: KnownParameter(3000.0),
        },
    )

    result = run_nonlinear_identification(
        test_data=test_data,
        phase_definition=phase_definition,
        base_mechanical_properties=base_mechanical_properties,
        parameter_states=parameter_states,
    )

    recovered_yield_strength = float(
        np.nanmean(result.parameter_maps[EParameterName.YieldStrength.name])
    )
    recovered_hardening_modulus = float(
        np.nanmean(result.parameter_maps[EParameterName.HardeningModulus.name])
    )

    assert np.isfinite(result.cost)
    assert result.cost < 1.0e-8
    assert abs(recovered_yield_strength - true_yield_strength) < 2.0
    assert abs(recovered_hardening_modulus - true_hardening_modulus) < 20.0
