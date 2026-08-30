import numpy as np
import pytest

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.loadregimes import LoadRegimeThresholds, ResolvedLoadRegimes
from pyvale.vfm.materialprojection import NativeDofSensitivityAuditConfig
from pyvale.vfm.objectivefuncmaterialinformation import (
    MaterialFeatureReduction,
    MaterialFeatureTerm,
    MaterialInformationObjective,
)
from pyvale.vfm.residualblocks import ResidualBlockSpec
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest
from pyvale.vfm.solvepreparation import build_solve_preparation_context
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


class _MapLaw(IConstitutiveLaw):
    def calculate_stress(self, strain, parameter_maps):
        stress = np.zeros_like(strain)
        stress[:, 0] = parameter_maps["yield_strength"]
        return stress

    def get_required_parameters(self):
        return ("yield_strength",)

    def get_identification_type(self):
        return EIdentificationType.Nonlinear


class _StressMetric(IMetric):
    def initialise(self, experiment_data):
        return None

    def evaluate(
        self,
        stress,
        constitutive_law,
        parameter_map_size,
        spatial_parameterisations,
        experiment_data,
    ):
        return MetricResult(residual=stress[:, 0].reshape(-1).copy())


def _experiment() -> ExperimentData:
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 2))
    roi = VfmRegionOfInterest.from_definition(
        RoiDefinition(
            shapes=(
                RoiShape(
                    shape_type="rectangle",
                    index=0,
                    is_cutting=False,
                    rectangle=(0.0, 0.0, 1.0, 1.0),
                ),
            )
        )
    )
    geometry = SpecimenGeometry(x, y, np.ones_like(x), 1.0, roi)
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    conditions = BoundaryConditions(
        EdgeConditions(free, free, free, free),
        np.zeros((2, 2), dtype=np.float64),
    )
    strain = np.zeros((2, 3, *x.shape), dtype=np.float64)
    return ExperimentData(strain, geometry, conditions, np.array([0.0, 1.0]))


def test_preparation_context_snapshots_dofs_and_restores_candidate_state() -> None:
    experiment = _experiment()
    original = SpatialParameterisationHomogeneous(
        DegreeOfFreedom(4.0, 0.0, 10.0)
    )
    state = PhaseSpatialState({"yield_strength": [original]})
    context = build_solve_preparation_context(
        phase_index=1,
        solve_iteration=3,
        constitutive_law=_MapLaw(),
        parameter_map_size=np.asarray(experiment.specimen_geometry.x.shape),
        spatial_state=state,
        metrics=[_StressMetric()],
        experiment_data=experiment,
    )

    perturbed = context.evaluate_metric_results(np.array([0.8]))

    assert context.phase_index == 1
    assert context.solve_iteration == 3
    assert context.normalised_degrees_of_freedom == pytest.approx([0.4])
    assert context.degrees_of_freedom[0].value == pytest.approx(4.0)
    assert context.degrees_of_freedom[0].parameter_names == ("yield_strength",)
    assert context.degrees_of_freedom[0].role == "homogeneous"
    assert context.degrees_of_freedom[0].owner_type == (
        "SpatialParameterisationHomogeneous"
    )
    assert np.all(context.metric_results[0].residual == 4.0)
    assert np.all(perturbed[0].residual == 8.0)
    assert original.value.value == pytest.approx(4.0)
    assert np.all(context.parameter_maps["yield_strength"] == 4.0)
    assert context.spatial_state.collect_normalised_degrees_of_freedom() == (
        pytest.approx([0.4])
    )


def test_material_objective_persists_opt_in_audit_diagnostics() -> None:
    experiment = _experiment()
    context = build_solve_preparation_context(
        phase_index=0,
        solve_iteration=0,
        constitutive_law=_MapLaw(),
        parameter_map_size=np.asarray(experiment.specimen_geometry.x.shape),
        spatial_state=PhaseSpatialState(
            {
                "yield_strength": [
                    SpatialParameterisationHomogeneous(
                        DegreeOfFreedom(4.0, 0.0, 10.0)
                    )
                ]
            }
        ),
        metrics=[_StressMetric()],
        experiment_data=experiment,
    )
    regimes = ResolvedLoadRegimes(
        thresholds=LoadRegimeThresholds(),
        yielded_fraction=(0.0, 1.0),
        pre_yield=(0,),
        onset=(0,),
        developed=(1,),
        late=(1,),
    )
    objective = MaterialInformationObjective(
        global_objective=None,
        feature_terms=(
            MaterialFeatureTerm(
                "closure",
                0,
                MaterialFeatureReduction.RMS,
            ),
        ),
        alpha=1.0,
        sensitivity_audit=NativeDofSensitivityAuditConfig(
            load_regimes=regimes,
            residual_blocks=(
                ResidualBlockSpec(
                    name="yield-all",
                    metric_index=0,
                    metric_kind="test",
                    load_regime="all",
                ),
            ),
        ),
    )

    objective.prepare_solve(context)

    diagnostics = objective.diagnostics()
    assert diagnostics["sensitivity_audit"]["projection_bases"]["full"][
        "rank"
    ] == 1
    assert diagnostics["sensitivity_audit"]["residual_layout"][
        "blocks"
    ][0]["name"] == "yield-all"


@pytest.mark.parametrize("candidate", [np.array([np.nan]), np.array([-0.1]), np.array([1.1])])
def test_preparation_context_rejects_invalid_candidate_dofs(candidate) -> None:
    experiment = _experiment()
    state = PhaseSpatialState(
        {
            "yield_strength": [
                SpatialParameterisationHomogeneous(
                    DegreeOfFreedom(4.0, 0.0, 10.0)
                )
            ]
        }
    )
    context = build_solve_preparation_context(
        phase_index=0,
        solve_iteration=0,
        constitutive_law=_MapLaw(),
        parameter_map_size=np.asarray(experiment.specimen_geometry.x.shape),
        spatial_state=state,
        metrics=[_StressMetric()],
        experiment_data=experiment,
    )

    with pytest.raises(ValueError):
        context.evaluate_metric_results(candidate)


def test_native_dof_audit_uses_frozen_layout_without_mutating_state() -> None:
    experiment = _experiment()
    original = SpatialParameterisationHomogeneous(
        DegreeOfFreedom(4.0, 0.0, 10.0)
    )
    state = PhaseSpatialState({"yield_strength": [original]})
    context = build_solve_preparation_context(
        phase_index=0,
        solve_iteration=2,
        constitutive_law=_MapLaw(),
        parameter_map_size=np.asarray(experiment.specimen_geometry.x.shape),
        spatial_state=state,
        metrics=[_StressMetric()],
        experiment_data=experiment,
    )
    regimes = ResolvedLoadRegimes(
        thresholds=LoadRegimeThresholds(),
        yielded_fraction=(0.0, 1.0),
        pre_yield=(0,),
        onset=(0,),
        developed=(1,),
        late=(1,),
    )
    config = NativeDofSensitivityAuditConfig(
        load_regimes=regimes,
        residual_blocks=(
            ResidualBlockSpec(
                name="yield-all",
                metric_index=0,
                metric_kind="test",
                load_regime="all",
                noise_scale=2.0,
            ),
        ),
        step=1.0e-4,
    )

    audit = config.prepare(context)

    assert audit.parameter_groups == ("yield",)
    assert audit.projection_bases.full.rank == 1
    assert audit.projection_bases.yield_basis is not None
    assert audit.column_norms[0] == pytest.approx(5.0)
    assert audit.diagnostics()["residual_layout"]["block_count"] == 1
    assert original.value.value == pytest.approx(4.0)
    assert context.normalised_degrees_of_freedom == pytest.approx([0.4])
