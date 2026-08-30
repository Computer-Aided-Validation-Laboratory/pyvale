import numpy as np

from pyvale.vfm.constlaw import EIdentificationType, IConstitutiveLaw
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.identification import prepare_phase_runtime
from pyvale.vfm.identificationconfig import IdentificationPhase
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.phasepreparation import (
    AutomaticEgiSupportPreparation,
    FixedEgiSupportPreparation,
    PhasePreparationContext,
    PhasePreparationResult,
)
from pyvale.vfm.egisupports import PhysicalEgiSupport
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


class _Law(IConstitutiveLaw):
    def calculate_stress(self, strain, parameter_maps):
        stress = np.zeros_like(strain)
        stress[:, 0] = parameter_maps["yield_strength"]
        return stress

    def get_required_parameters(self):
        return ("yield_strength",)

    def get_identification_type(self):
        return EIdentificationType.Nonlinear


class _Metric(IMetric):
    def __init__(self, name):
        self.name = name
        self.initialised = False

    def initialise(self, experiment_data):
        self.initialised = True

    def evaluate(self, *args):
        return MetricResult(residual=np.zeros(1))


class _Preparation:
    def prepare(self, context):
        assert context.phase_index == 1
        assert context.accepted_parameter_maps["yield_strength"] is not None
        return PhasePreparationResult(
            metrics=[_Metric("replacement")],
            diagnostics={"prepared_from_phase": context.phase_index},
        )


def _experiment() -> ExperimentData:
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
    roi = VfmRegionOfInterest.from_definition(RoiDefinition(shapes=(
        RoiShape("rectangle", 0, False, rectangle=(0.0, 0.0, 2.0, 2.0)),
    )))
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    traction_y = Edge(EEdgeCondition.Free, EEdgeCondition.Traction)
    return ExperimentData(
        np.zeros((2, 3, 3, 3)),
        SpecimenGeometry(x, y, np.ones_like(x), 1.0, roi),
        BoundaryConditions(EdgeConditions(free, free, free, free), np.zeros((2, 2))),
        np.array((0.0, 1.0)),
    )


def test_phase_preparation_replaces_only_copied_runtime_metrics() -> None:
    phase = IdentificationPhase(
        spatial_parameterisations={"yield_strength": [SpatialParameterisationHomogeneous()]},
        metrics=[_Metric("configured")],
        objective_function=None,  # Not used by runtime preparation.
        optimiser=None,
        phase_preparation=_Preparation(),
    )
    experiment = _experiment()
    maps = {"yield_strength": np.full((3, 3), 400.0)}

    runtime = prepare_phase_runtime(
        phase,
        experiment,
        phase_index=1,
        constitutive_law=_Law(),
        parameter_map_size=np.array((3, 3), dtype=np.uint32),
        accepted_parameter_maps=maps,
    )

    assert phase.metrics[0].name == "configured"
    assert runtime.metrics[0].name == "replacement"
    assert runtime.metrics[0].initialised
    assert runtime.phase_preparation_diagnostics == {"prepared_from_phase": 1}


def test_fixed_egi_preparation_replaces_templates_in_role_order() -> None:
    support = lambda size: PhysicalEgiSupport(
        requested_side_lengths=(float(size),),
        window_size=(size, size),
        nominal_side_lengths=(float(size), float(size)),
        grid_spacing=(1.0, 1.0),
    )
    preparation = FixedEgiSupportPreparation(
        (("fine", support(3)), ("middle", support(5)), ("broad", support(7)))
    )
    context = type("Context", (), {
        "configured_metrics": (_Metric("force"), EquilibriumGapMetric(window_size=(9, 9))),
    })()

    result = preparation.prepare(context)

    assert isinstance(result.metrics[0], _Metric)
    assert [tuple(metric.window_size) for metric in result.metrics[1:]] == [
        (3, 3), (5, 5), (7, 7)
    ]
    assert result.diagnostics["mode"] == "fixed"


def test_automatic_egi_preparation_selects_and_installs_three_metrics() -> None:
    grid = np.linspace(0.0, 1.0, 15)
    x, y = np.meshgrid(grid, grid)
    roi = VfmRegionOfInterest.from_definition(RoiDefinition(shapes=(
        RoiShape("rectangle", 0, False, rectangle=(0.0, 0.0, 1.0, 1.0)),
    )))
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    traction_y = Edge(EEdgeCondition.Free, EEdgeCondition.Traction)
    experiment = ExperimentData(
        np.zeros((3, 3, 15, 15)),
        SpecimenGeometry(x, y, np.ones_like(x), 1.0, roi),
        BoundaryConditions(
            EdgeConditions(free, free, free, traction_y),
            np.column_stack((np.zeros(3), np.array((1.0, 1.5, 2.0)))),
        ),
        np.array((0.0, 1.0, 2.0)),
    )
    law = _Law()
    maps = {"yield_strength": np.full((15, 15), 400.0)}
    context = PhasePreparationContext(
        phase_index=1,
        experiment_data=experiment,
        constitutive_law=law,
        parameter_map_size=np.array((15, 15), dtype=np.uint32),
        accepted_parameter_maps=maps,
        accepted_stress=law.calculate_stress(experiment.strain, maps),
        configured_metrics=(EquilibriumGapMetric(window_size=(3, 3)),),
    )

    result = AutomaticEgiSupportPreparation(
        yield_parameter_range=100.0,
        residual_noise_scale=1.0e-4,
        local_probe_count=4,
    ).prepare(context)

    assert len(result.metrics) == 3
    assert result.diagnostics["mode"] == "automatic"
    assert result.diagnostics["selection"]["status"] == "three_resolved"
