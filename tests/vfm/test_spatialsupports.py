import copy

import pytest
import numpy as np

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identification import prepare_phase_runtime
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncvector import VectorFirstResultPassthrough
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.optimiserslicewiseindependent import (
    SliceWiseIndependentLeastSquares,
)
from pyvale.vfm.refinement import BasisAddRemoveRefinement
from pyvale.vfm.refinement import RefinementContext
from pyvale.vfm.refinement import SliceMergeSplitRefinement
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparambasisfuncs import (
    BasisFunctionKernelUnivariate,
    SpatialParameterisationBasisFunction,
    SupportBasis,
)
from pyvale.vfm.spatialparamslicewise import (
    SliceConfig,
    SliceWiseSpatialParameterisation,
    SupportSlice,
)
from pyvale.vfm.vfmregionofinterest import (
    RoiDefinition,
    RoiShape,
    VfmRegionOfInterest,
)
from pyvale.vfm.validation import validate_slicewise_independent_phase
from pyvale.vfm.validation import run_validation
from pyvale.vfm.constlaw import IConstitutiveLaw, EIdentificationType
from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.metric import MetricResult
from pyvale.vfm.metric import IMetric
from pyvale.vfm.objectivefunc import IVectorObjectiveFunction
from pyvale.vfm.optimiser import IOptimiser


class _DummyConstitutiveLaw(IConstitutiveLaw):
    def calculate_stress(
        self,
        strain: np.ndarray,
        parameter_maps: dict[str, np.ndarray],
    ) -> np.ndarray:
        timesteps = strain.shape[0]
        shape = strain.shape[2:]
        stress = np.zeros((timesteps, 3, shape[0], shape[1]), dtype=np.float64)
        for parameter_map in parameter_maps.values():
            stress[:, 0, :, :] += parameter_map[np.newaxis, :, :]
        return stress

    def get_required_parameters(self) -> tuple[str, ...]:
        return ("yield_strength", "hardening_modulus")

    def get_identification_type(self) -> EIdentificationType:
        return EIdentificationType.Nonlinear


class _DummyMetric(IMetric):
    def initialise(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        return

    def evaluate(
        self,
        stress: np.ndarray,
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: np.ndarray,
        spatial_parameterisations,
        experiment_data: ExperimentData,
    ) -> MetricResult:
        return MetricResult(
            residual=stress[:, 0, :, :].reshape(-1),
        )


class _DummyObjective(IVectorObjectiveFunction):
    def evaluate(
        self,
        metric_results: list[MetricResult],
    ) -> np.ndarray:
        assert metric_results[0].residual is not None
        return metric_results[0].residual


class _DummySliceForceResult:
    def __init__(self, weighted_temporal_rms: np.ndarray) -> None:
        self.weighted_temporal_rms = weighted_temporal_rms


def _build_experiment_data() -> ExperimentData:
    x_grid_1d = np.linspace(0.0, 1.0, 5)
    y_grid_1d = np.linspace(0.0, 1.0, 4)
    x_grid, y_grid = np.meshgrid(x_grid_1d, y_grid_1d)
    pixel_area = np.full_like(x_grid, 0.1, dtype=np.float64)
    region_of_interest = VfmRegionOfInterest.from_definition(
        RoiDefinition(
            shapes=(
                RoiShape(
                    shape_type="rectangle",
                    index=0,
                    is_cutting=False,
                    rectangle=(0.0, 0.0, 1.0, 1.0),
                ),
            ),
        )
    )

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        pixel_area,
        1.0,
        region_of_interest,
    )

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_x_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            min_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Free),
            max_y_edge=Edge(x=EEdgeCondition.Free, y=EEdgeCondition.Traction),
        ),
        np.ones((3, 2), dtype=np.float64),
    )

    strain = np.zeros((3, 3, y_grid.shape[0], x_grid.shape[1]), dtype=np.float64)
    timesteps = np.array([0.0, 1.0, 2.0], dtype=np.float64)

    return ExperimentData(
        strain,
        specimen_geometry,
        boundary_conditions,
        timesteps,
    )


def _build_refinement_context(
    experiment_data: ExperimentData,
    parameter_maps: dict[str, np.ndarray],
) -> RefinementContext:
    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    return RefinementContext(
        experiment_data=experiment_data,
        constitutive_law=_DummyConstitutiveLaw(),
        constitutive_parameters={
            parameter_name: ConstitutiveParameter(
                parameter_map,
                -1.0e6,
                1.0e6,
            )
            for parameter_name, parameter_map in parameter_maps.items()
        },
        parameter_map_size=parameter_map_size,
        parameter_maps=parameter_maps,
    )


def test_phase_spatial_state_collects_shared_basis_support_dofs_once() -> None:
    x_grid_1d = np.linspace(0.0, 1.0, 5)
    y_grid_1d = np.linspace(0.0, 1.0, 4)
    x_grid, y_grid = np.meshgrid(x_grid_1d, y_grid_1d)

    shared_support = SupportBasis(
        x=x_grid,
        y=y_grid,
        kernels=[
            BasisFunctionKernelUnivariate(
                x=DegreeOfFreedom(0.25, 0.0, 1.0),
                y=DegreeOfFreedom(0.5, 0.0, 1.0),
                variance=DegreeOfFreedom(0.1, 0.01, 1.0),
            )
        ],
    )

    spatial_parameterisations = {
        "yield_strength": [
            SpatialParameterisationBasisFunction(
                support=shared_support,
                heights=[DegreeOfFreedom(2.0, -5.0, 5.0)],
            )
        ],
        "hardening_modulus": [
            SpatialParameterisationBasisFunction(
                support=shared_support,
                heights=[DegreeOfFreedom(3.0, -5.0, 5.0)],
            )
        ],
    }

    phase_spatial_state = PhaseSpatialState(spatial_parameterisations)
    degrees_of_freedom = phase_spatial_state.collect_degrees_of_freedom()
    assert len(degrees_of_freedom) == 5

    original_yield_map = spatial_parameterisations["yield_strength"][0].to_map(
        np.array(x_grid.shape, dtype=np.uint32)
    )
    original_hardening_map = spatial_parameterisations["hardening_modulus"][0].to_map(
        np.array(x_grid.shape, dtype=np.uint32)
    )

    perturbed_dofs = phase_spatial_state.collect_normalised_degrees_of_freedom()
    perturbed_dofs[0] = 0.75
    perturbed_phase_spatial_state = phase_spatial_state.copy()
    perturbed_phase_spatial_state.update_from_normalised_degrees_of_freedom(
        perturbed_dofs
    )

    yield_parameterisation = (
        perturbed_phase_spatial_state.spatial_parameterisations["yield_strength"][0]
    )
    hardening_parameterisation = (
        perturbed_phase_spatial_state.spatial_parameterisations["hardening_modulus"][0]
    )
    assert yield_parameterisation.support is hardening_parameterisation.support

    perturbed_yield_map = yield_parameterisation.to_map(
        np.array(x_grid.shape, dtype=np.uint32)
    )
    perturbed_hardening_map = hardening_parameterisation.to_map(
        np.array(x_grid.shape, dtype=np.uint32)
    )

    assert not np.allclose(perturbed_yield_map, original_yield_map)
    assert not np.allclose(perturbed_hardening_map, original_hardening_map)


def test_basis_refinement_adds_kernel_to_shared_support_and_height_slots() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportBasis(
        x=experiment_data.specimen_geometry.x,
        y=experiment_data.specimen_geometry.y,
    )
    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SpatialParameterisationBasisFunction(support=shared_support)
            ],
            "hardening_modulus": [
                SpatialParameterisationBasisFunction(support=shared_support)
            ],
        },
        metrics=[],
        objective_function=_DummyObjective(),
        optimiser=OptimiserLeastSquares(),
        refinement_policy=BasisAddRemoveRefinement(
            target=shared_support,
            max_refinements=1,
            seed_parameter_name="yield_strength",
        ),
    )
    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    assert phase_runtime.refinement_policy is not None

    parameter_maps = {
        "yield_strength": np.asarray(
            experiment_data.specimen_geometry.x
            + experiment_data.specimen_geometry.y,
            dtype=np.float64,
        ),
        "hardening_modulus": np.asarray(
            2.0 * experiment_data.specimen_geometry.x,
            dtype=np.float64,
        ),
    }
    context = _build_refinement_context(experiment_data, parameter_maps)
    context.parameter_map_size = parameter_map_size

    action = phase_runtime.refinement_policy.propose(phase_runtime, context)
    assert action is not None
    action.apply(phase_runtime, context)

    runtime_support = phase_runtime.resolve_support_target(
        phase_runtime.refinement_policy.target,
    )
    assert isinstance(runtime_support, SupportBasis)
    assert runtime_support.kernels is not None
    assert len(runtime_support.kernels) == 1

    _, yield_parameterisation = phase_runtime.get_parameterisation("yield_strength", 0)
    _, hardening_parameterisation = phase_runtime.get_parameterisation(
        "hardening_modulus",
        0,
    )
    assert isinstance(yield_parameterisation, SpatialParameterisationBasisFunction)
    assert isinstance(hardening_parameterisation, SpatialParameterisationBasisFunction)
    assert yield_parameterisation.support is hardening_parameterisation.support
    assert len(yield_parameterisation.heights) == 1
    assert len(hardening_parameterisation.heights) == 1

    phase_runtime.initialise_from_constitutive_parameters(
        context.constitutive_parameters,
        parameter_map_size,
    )
    assert yield_parameterisation.heights[0] is not None
    assert hardening_parameterisation.heights[0] is not None


def test_basis_refinement_removes_small_shared_kernel() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportBasis(
        x=experiment_data.specimen_geometry.x,
        y=experiment_data.specimen_geometry.y,
        kernels=[
            BasisFunctionKernelUnivariate(
                x=DegreeOfFreedom(0.25, 0.0, 1.0),
                y=DegreeOfFreedom(0.5, 0.0, 1.0),
                variance=DegreeOfFreedom(0.1, 0.01, 1.0),
            )
        ],
    )
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SpatialParameterisationBasisFunction(
                    support=shared_support,
                    heights=[DegreeOfFreedom(1.0e-4, -5.0, 5.0)],
                )
            ],
            "hardening_modulus": [
                SpatialParameterisationBasisFunction(
                    support=shared_support,
                    heights=[DegreeOfFreedom(2.0e-4, -5.0, 5.0)],
                )
            ],
        },
        metrics=[],
        objective_function=_DummyObjective(),
        optimiser=OptimiserLeastSquares(),
        refinement_policy=BasisAddRemoveRefinement(
            target=shared_support,
            max_refinements=1,
            mode="remove",
            remove_height_threshold=1.0e-3,
        ),
    )
    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    assert phase_runtime.refinement_policy is not None
    context = _build_refinement_context(
        experiment_data,
        {
            "yield_strength": np.zeros(experiment_data.specimen_geometry.x.shape),
            "hardening_modulus": np.zeros(experiment_data.specimen_geometry.x.shape),
        },
    )

    action = phase_runtime.refinement_policy.propose(phase_runtime, context)
    assert action is not None
    action.apply(phase_runtime, context)

    runtime_support = phase_runtime.resolve_support_target(
        phase_runtime.refinement_policy.target,
    )
    assert isinstance(runtime_support, SupportBasis)
    assert runtime_support.kernels == []
    _, yield_parameterisation = phase_runtime.get_parameterisation("yield_strength", 0)
    _, hardening_parameterisation = phase_runtime.get_parameterisation(
        "hardening_modulus",
        0,
    )
    assert isinstance(yield_parameterisation, SpatialParameterisationBasisFunction)
    assert isinstance(hardening_parameterisation, SpatialParameterisationBasisFunction)
    assert yield_parameterisation.heights == []
    assert hardening_parameterisation.heights == []


def test_shared_slice_support_is_reused_by_metric_and_parameterisations() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3)
    )

    yield_parameterisation = SliceWiseSpatialParameterisation(support=shared_support)
    hardening_parameterisation = SliceWiseSpatialParameterisation(
        support=shared_support
    )
    metric = SliceWiseForceReconstructionMetric(support=shared_support)

    phase_spatial_state = PhaseSpatialState(
        {
            "yield_strength": [yield_parameterisation],
            "hardening_modulus": [hardening_parameterisation],
        }
    )
    phase_spatial_state.prepare(experiment_data)
    metric.initialise(experiment_data)

    assert yield_parameterisation.support is hardening_parameterisation.support
    assert yield_parameterisation.support is metric.support
    assert yield_parameterisation.slice_partition is metric.slice_partition


def test_run_identification_handles_shared_slice_support_with_general_optimiser() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3)
    )

    parameters = {
        "yield_strength": ConstitutiveParameter(
            2.0,
            0.5,
            5.0,
            np.array(experiment_data.specimen_geometry.x.shape, dtype=np.uint32),
        ),
        "hardening_modulus": ConstitutiveParameter(
            3.0,
            0.5,
            5.0,
            np.array(experiment_data.specimen_geometry.x.shape, dtype=np.uint32),
        ),
    }

    identification = IdentificationConfig(
        constitutive_law=_DummyConstitutiveLaw(),
        parameters=parameters,
        phases=[
            IdentificationPhase(
                spatial_parameterisations={
                    "yield_strength": [
                        SliceWiseSpatialParameterisation(support=shared_support)
                    ],
                    "hardening_modulus": [
                        SliceWiseSpatialParameterisation(support=shared_support)
                    ],
                },
                metrics=[_DummyMetric()],
                objective_function=_DummyObjective(),
                optimiser=OptimiserLeastSquares(),
            )
        ],
    )

    result = run_identification(
        experiment_data,
        identification,
    )

    assert result.parameter_maps["yield_strength"].shape == (
        experiment_data.specimen_geometry.x.shape
    )
    assert result.parameter_maps["hardening_modulus"].shape == (
        experiment_data.specimen_geometry.x.shape
    )


def test_validate_slicewise_independent_phase_is_pure_validation() -> None:
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3)
    )
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
        },
        metrics=[SliceWiseForceReconstructionMetric(support=shared_support)],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=SliceWiseIndependentLeastSquares(),
    )

    validate_slicewise_independent_phase(phase, 0)

    metric = phase.metrics[0]
    assert isinstance(metric, SliceWiseForceReconstructionMetric)
    assert shared_support.slice_partition is None
    assert metric.slice_partition is None
    assert phase.spatial_parameterisations["yield_strength"][0].slice_partition is None
    assert phase.spatial_parameterisations["yield_strength"][0].support is shared_support


def test_run_validation_checks_slicewise_independent_phases() -> None:
    parameters = {
        "yield_strength": ConstitutiveParameter(
            2.0,
            0.5,
            5.0,
            np.array((4, 5), dtype=np.uint32),
        ),
        "hardening_modulus": ConstitutiveParameter(
            3.0,
            0.5,
            5.0,
            np.array((4, 5), dtype=np.uint32),
        ),
    }

    identification = IdentificationConfig(
        constitutive_law=_DummyConstitutiveLaw(),
        parameters=parameters,
        phases=[
            IdentificationPhase(
                spatial_parameterisations={
                    "yield_strength": [
                        SliceWiseSpatialParameterisation(
                            support=SupportSlice(
                                slice_config=SliceConfig(axis="x", num_slices=3)
                            )
                        )
                    ],
                    "hardening_modulus": [
                        SliceWiseSpatialParameterisation(
                            support=SupportSlice(
                                slice_config=SliceConfig(axis="x", num_slices=3)
                            )
                        )
                    ],
                },
                metrics=[_DummyMetric()],
                objective_function=VectorFirstResultPassthrough(),
                optimiser=SliceWiseIndependentLeastSquares(),
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match="SliceWiseIndependentLeastSquares requires exactly one "
        "SliceWiseForceReconstructionMetric",
    ):
        run_validation(_build_experiment_data(), identification)


def test_validate_slicewise_independent_phase_requires_shared_support_instance() -> None:
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SliceWiseSpatialParameterisation(
                    support=SupportSlice(
                        slice_config=SliceConfig(axis="x", num_slices=3)
                    ),
                    values=[1.0, 1.0, 1.0],
                )
            ],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(
                    support=SupportSlice(
                        slice_config=SliceConfig(axis="x", num_slices=3)
                    ),
                    values=[2.0, 2.0, 2.0],
                )
            ],
        },
        metrics=[
            SliceWiseForceReconstructionMetric(
                support=SupportSlice(
                    slice_config=SliceConfig(axis="x", num_slices=3)
                )
            )
        ],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=SliceWiseIndependentLeastSquares(),
    )

    with pytest.raises(
        ValueError,
        match="requires this parameterisation to reference the same SupportSlice object",
    ):
        validate_slicewise_independent_phase(phase, 0)


def test_prepare_phase_runtime_preserves_shared_independent_slice_support() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3)
    )
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SliceWiseSpatialParameterisation(
                    support=shared_support,
                    values=[1.0, 1.0, 1.0],
                )
            ],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(
                    support=shared_support,
                    values=[2.0, 2.0, 2.0],
                )
            ],
        },
        metrics=[
            SliceWiseForceReconstructionMetric(
                support=shared_support
            )
        ],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=SliceWiseIndependentLeastSquares(),
    )

    validate_slicewise_independent_phase(phase, 0)
    phase_runtime = prepare_phase_runtime(phase, experiment_data)

    metric = phase_runtime.metrics[0]
    assert isinstance(metric, SliceWiseForceReconstructionMetric)
    assert metric.slice_partition is not None

    phase_spatial_state = phase_runtime.spatial_state
    yield_parameterisation = phase_spatial_state.spatial_parameterisations["yield_strength"][0]
    hardening_parameterisation = phase_spatial_state.spatial_parameterisations["hardening_modulus"][0]
    assert isinstance(yield_parameterisation, SliceWiseSpatialParameterisation)
    assert isinstance(hardening_parameterisation, SliceWiseSpatialParameterisation)
    assert yield_parameterisation.support is metric.support
    assert hardening_parameterisation.support is metric.support
    assert metric.support is not shared_support
    assert yield_parameterisation.slice_partition is not None
    assert hardening_parameterisation.slice_partition is not None
    assert yield_parameterisation.slice_partition is metric.slice_partition
    assert hardening_parameterisation.slice_partition is metric.slice_partition

    # Runtime preparation preserves the caller's declared sharing while still
    # using a copied runtime support object.
    original_metric = phase.metrics[0]
    assert isinstance(original_metric, SliceWiseForceReconstructionMetric)
    assert original_metric.support is not metric.support


def test_adopt_spatial_parameterisations_repoints_policy_support_target() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3)
    )
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(support=shared_support)
            ],
        },
        metrics=[
            SliceWiseForceReconstructionMetric(
                support=shared_support
            )
        ],
        objective_function=VectorFirstResultPassthrough(),
        optimiser=SliceWiseIndependentLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(target=shared_support),
    )

    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    assert phase_runtime.refinement_policy is not None
    old_target = phase_runtime.refinement_policy.target
    updated_spatial_parameterisations = copy.deepcopy(
        phase_runtime.spatial_parameterisations
    )
    phase_runtime.adopt_spatial_parameterisations(updated_spatial_parameterisations)

    new_support = updated_spatial_parameterisations["yield_strength"][0].support
    assert phase_runtime.refinement_policy.target is new_support
    assert phase_runtime.refinement_policy.target is not old_target
    metric = phase_runtime.metrics[0]
    assert isinstance(metric, SliceWiseForceReconstructionMetric)
    assert metric.support is new_support


def test_support_slice_refinement_merges_adjacent_similar_slices() -> None:
    experiment_data = _build_experiment_data()
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
    )
    yield_parameterisation = SliceWiseSpatialParameterisation(support=support)
    hardening_parameterisation = SliceWiseSpatialParameterisation(support=support)
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [yield_parameterisation],
            "hardening_modulus": [hardening_parameterisation],
        },
        metrics=[],
        objective_function=_DummyObjective(),
        optimiser=OptimiserLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(
            target=support,
            merge_parameter_tolerance=0.05,
        ),
    )
    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    assert phase_runtime.refinement_policy is not None
    runtime_support = phase_runtime.resolve_support_target(
        phase_runtime.refinement_policy.target,
    )
    assert isinstance(runtime_support, SupportSlice)

    assert runtime_support.slice_partition is not None
    original_boundaries = runtime_support.slice_partition.boundaries.copy()
    parameter_maps = {
        "yield_strength": np.zeros(
            experiment_data.specimen_geometry.x.shape,
            dtype=np.float64,
        ),
        "hardening_modulus": np.zeros(
            experiment_data.specimen_geometry.x.shape,
            dtype=np.float64,
        ),
    }
    for slice_index, (yield_value, hardening_value) in enumerate(
        zip(
            (100.0, 104.0, 150.0, 220.0),
            (1000.0, 1040.0, 2000.0, 2500.0),
            strict=True,
        )
    ):
        slice_mask = runtime_support.slice_partition.slice_id_map == slice_index
        parameter_maps["yield_strength"][slice_mask] = yield_value
        parameter_maps["hardening_modulus"][slice_mask] = hardening_value

    context = _build_refinement_context(experiment_data, parameter_maps)
    action = phase_runtime.refinement_policy.propose(phase_runtime, context)
    assert action is not None
    action.apply(phase_runtime, context)

    assert runtime_support.slice_partition is None
    assert runtime_support.slice_config is not None
    assert runtime_support.slice_config.boundaries is not None
    assert runtime_support.slice_config.boundaries.shape[0] == original_boundaries.shape[0] - 1
    assert np.allclose(
        runtime_support.slice_config.boundaries,
        np.delete(original_boundaries, 1),
    )


def test_support_slice_refinement_splits_high_error_slices(monkeypatch) -> None:
    experiment_data = _build_experiment_data()
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
    )
    metric = SliceWiseForceReconstructionMetric(support=support)
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [SliceWiseSpatialParameterisation(support=support)],
            "hardening_modulus": [SliceWiseSpatialParameterisation(support=support)],
        },
        metrics=[metric],
        objective_function=_DummyObjective(),
        optimiser=OptimiserLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(
            target=support,
            split_error_threshold=0.1,
        ),
    )
    force_error_ratio = np.array([0.05, 0.12, 0.03, 0.25], dtype=np.float64)

    def fake_force_error(*args, **kwargs):
        return _DummySliceForceResult(force_error_ratio)

    monkeypatch.setattr(
        SliceWiseForceReconstructionMetric,
        "evaluate_force_recon_error",
        fake_force_error,
    )
    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    assert phase_runtime.refinement_policy is not None
    runtime_support = phase_runtime.resolve_support_target(
        phase_runtime.refinement_policy.target,
    )
    assert isinstance(runtime_support, SupportSlice)

    assert runtime_support.slice_partition is not None
    original_boundaries = runtime_support.slice_partition.boundaries.copy()
    empty_maps = {
        "yield_strength": np.zeros(experiment_data.specimen_geometry.x.shape),
        "hardening_modulus": np.zeros(experiment_data.specimen_geometry.x.shape),
    }
    context = _build_refinement_context(experiment_data, empty_maps)
    action = phase_runtime.refinement_policy.propose(phase_runtime, context)
    assert action is not None
    action.apply(phase_runtime, context)

    assert runtime_support.slice_partition is None
    assert runtime_support.slice_config is not None
    assert runtime_support.slice_config.boundaries is not None
    assert np.allclose(
        runtime_support.slice_config.boundaries,
        np.sort(
            np.concatenate(
                (
                    original_boundaries,
                    [
                        0.5 * (original_boundaries[1] + original_boundaries[2]),
                        0.5 * (original_boundaries[3] + original_boundaries[4]),
                    ],
                )
            )
        ),
    )


def test_slice_parameterisation_refits_after_support_slice_count_changes() -> None:
    experiment_data = _build_experiment_data()
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
    )
    parameterisation = SliceWiseSpatialParameterisation(support=support)
    phase = IdentificationPhase(
        spatial_parameterisations={
            "yield_strength": [parameterisation],
            "hardening_modulus": [
                SliceWiseSpatialParameterisation(
                    support=SupportSlice(slice_config=SliceConfig(axis="x", num_slices=4))
                )
            ],
        },
        metrics=[],
        objective_function=_DummyObjective(),
        optimiser=OptimiserLeastSquares(),
        refinement_policy=SliceMergeSplitRefinement(
            target=("yield_strength", 0),
            merge_parameter_tolerance=0.05,
        ),
    )
    phase_runtime = prepare_phase_runtime(phase, experiment_data)
    _, runtime_parameterisation = phase_runtime.get_parameterisation(
        "yield_strength",
        0,
    )
    assert isinstance(runtime_parameterisation, SliceWiseSpatialParameterisation)
    runtime_parameterisation.initialise_from_constitutive_parameter(
        ConstitutiveParameter(
            np.arange(
                experiment_data.specimen_geometry.x.size,
                dtype=np.float64,
            ).reshape(experiment_data.specimen_geometry.x.shape),
            0.0,
            100.0,
        )
    )

    assert runtime_parameterisation.values is not None
    assert len(runtime_parameterisation.values) == 4
    runtime_support = runtime_parameterisation.support
    assert runtime_support.slice_partition is not None
    parameter_map = np.zeros(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.float64,
    )
    for slice_index, value in enumerate((100.0, 104.0, 150.0, 220.0)):
        parameter_map[runtime_support.slice_partition.slice_id_map == slice_index] = value
    parameter_maps = {"yield_strength": parameter_map}
    context = _build_refinement_context(experiment_data, parameter_maps)
    assert phase_runtime.refinement_policy is not None
    action = phase_runtime.refinement_policy.propose(phase_runtime, context)
    assert action is not None
    action.apply(phase_runtime, context)
    runtime_support.prepare(experiment_data)

    runtime_parameterisation.initialise_from_constitutive_parameter(
        ConstitutiveParameter(
            np.arange(
                experiment_data.specimen_geometry.x.size,
                dtype=np.float64,
            ).reshape(experiment_data.specimen_geometry.x.shape),
            0.0,
            100.0,
        )
    )

    assert runtime_parameterisation.values is not None
    assert len(runtime_parameterisation.values) == 3


def test_run_identification_handles_single_shared_support_refinement() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3),
    )
    parameter_map_size = np.array(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.uint32,
    )
    identification = IdentificationConfig(
        constitutive_law=_DummyConstitutiveLaw(),
        parameters={
            "yield_strength": ConstitutiveParameter(
                2.0,
                0.5,
                5.0,
                parameter_map_size,
            ),
            "hardening_modulus": ConstitutiveParameter(
                3.0,
                0.5,
                5.0,
                parameter_map_size,
            ),
        },
        phases=[
            IdentificationPhase(
                spatial_parameterisations={
                    "yield_strength": [
                        SliceWiseSpatialParameterisation(
                            support=shared_support,
                        )
                    ],
                    "hardening_modulus": [
                        SliceWiseSpatialParameterisation(
                            support=shared_support,
                        )
                    ],
                },
                metrics=[
                    SliceWiseForceReconstructionMetric(
                        support=shared_support,
                    )
                ],
                objective_function=VectorFirstResultPassthrough(),
                optimiser=SliceWiseIndependentLeastSquares(),
                refinement_policy=SliceMergeSplitRefinement(
                    target=shared_support,
                    merge_parameter_tolerance=10.0,
                    max_refinements=1,
                ),
            )
        ],
    )

    result = run_identification(
        experiment_data,
        identification,
    )

    assert result.parameter_maps["yield_strength"].shape == (
        experiment_data.specimen_geometry.x.shape
    )
    assert result.parameter_maps["hardening_modulus"].shape == (
        experiment_data.specimen_geometry.x.shape
    )
