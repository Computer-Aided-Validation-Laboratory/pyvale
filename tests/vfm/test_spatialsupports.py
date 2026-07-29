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
from pyvale.vfm.validation import validate_identification_config
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
        region_of_interest,
        1.0,
        pixel_area,
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

    identified_parameters = run_identification(
        experiment_data,
        identification,
    )

    assert identified_parameters["yield_strength"].map.shape == (
        experiment_data.specimen_geometry.x.shape
    )
    assert identified_parameters["hardening_modulus"].map.shape == (
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


def test_validate_identification_config_checks_slicewise_independent_phases() -> None:
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
        validate_identification_config(identification)


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


def test_support_slice_refinement_merges_adjacent_similar_slices() -> None:
    experiment_data = _build_experiment_data()
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
        refine=True,
        merge_parameter_tolerance=0.05,
    )
    yield_parameterisation = SliceWiseSpatialParameterisation(support=support)
    hardening_parameterisation = SliceWiseSpatialParameterisation(support=support)
    support.prepare(experiment_data)

    assert support.slice_partition is not None
    original_boundaries = support.slice_partition.boundaries.copy()
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
        slice_mask = support.slice_partition.slice_id_map == slice_index
        parameter_maps["yield_strength"][slice_mask] = yield_value
        parameter_maps["hardening_modulus"][slice_mask] = hardening_value
    spatial_parameterisations = {
        "yield_strength": [yield_parameterisation],
        "hardening_modulus": [hardening_parameterisation],
    }

    assert support.should_perform_refinement(
        parameter_maps=parameter_maps,
        spatial_parameterisations=spatial_parameterisations,
    )
    support.perform_refinement(
        parameter_maps=parameter_maps,
        spatial_parameterisations=spatial_parameterisations,
    )

    assert support.slice_partition is None
    assert support.slice_config is not None
    assert support.slice_config.boundaries is not None
    assert support.slice_config.boundaries.shape[0] == original_boundaries.shape[0] - 1
    assert np.allclose(
        support.slice_config.boundaries,
        np.delete(original_boundaries, 1),
    )


def test_support_slice_refinement_splits_high_error_slices() -> None:
    experiment_data = _build_experiment_data()
    support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=4),
        refine=True,
        split_error_threshold=0.1,
    )
    support.prepare(experiment_data)

    assert support.slice_partition is not None
    original_boundaries = support.slice_partition.boundaries.copy()
    support.perform_refinement(
        parameter_maps={},
        spatial_parameterisations={},
        force_error_ratio=np.array([0.05, 0.12, 0.03, 0.25], dtype=np.float64),
    )

    assert support.slice_partition is None
    assert support.slice_config is not None
    assert support.slice_config.boundaries is not None
    assert np.allclose(
        support.slice_config.boundaries,
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
        refine=True,
        merge_parameter_tolerance=0.05,
    )
    parameterisation = SliceWiseSpatialParameterisation(support=support)
    support.prepare(experiment_data)
    parameterisation.initialise_from_constitutive_parameter(
        ConstitutiveParameter(
            np.arange(
                experiment_data.specimen_geometry.x.size,
                dtype=np.float64,
            ).reshape(experiment_data.specimen_geometry.x.shape),
            0.0,
            100.0,
        )
    )

    assert parameterisation.values is not None
    assert len(parameterisation.values) == 4
    assert support.slice_partition is not None
    parameter_map = np.zeros(
        experiment_data.specimen_geometry.x.shape,
        dtype=np.float64,
    )
    for slice_index, value in enumerate((100.0, 104.0, 150.0, 220.0)):
        parameter_map[support.slice_partition.slice_id_map == slice_index] = value
    parameter_maps = {"yield_strength": parameter_map}
    spatial_parameterisations = {"yield_strength": [parameterisation]}
    support.perform_refinement(
        parameter_maps=parameter_maps,
        spatial_parameterisations=spatial_parameterisations,
    )
    support.prepare(experiment_data)

    parameterisation.initialise_from_constitutive_parameter(
        ConstitutiveParameter(
            np.arange(
                experiment_data.specimen_geometry.x.size,
                dtype=np.float64,
            ).reshape(experiment_data.specimen_geometry.x.shape),
            0.0,
            100.0,
        )
    )

    assert parameterisation.values is not None
    assert len(parameterisation.values) == 3


def test_run_identification_handles_single_shared_support_refinement() -> None:
    experiment_data = _build_experiment_data()
    shared_support = SupportSlice(
        slice_config=SliceConfig(axis="x", num_slices=3),
        refine=True,
        merge_parameter_tolerance=10.0,
        max_refinements=1,
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
            )
        ],
    )

    identified_parameters = run_identification(
        experiment_data,
        identification,
    )

    assert identified_parameters["yield_strength"].map.shape == (
        experiment_data.specimen_geometry.x.shape
    )
    assert identified_parameters["hardening_modulus"].map.shape == (
        experiment_data.specimen_geometry.x.shape
    )
