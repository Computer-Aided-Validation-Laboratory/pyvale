"""Focused engineering checks for the guarded EGI primary objective."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.measurementnoise import (
    MeasurementNoiseFloorConfig,
    MeasurementNoiseMode,
)
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncguardedegiprimary import (
    GuardedEgiPrimaryConfig,
    GuardedEgiPrimaryObjective,
    _guard_reference,
    _passes_limit,
    equal_mean_gated_egi_primary,
)
from pyvale.vfm.optimiser import evaluate_candidate
from pyvale.vfm.residualblocks import ResidualBlockSpec, prepare_canonical_residual_layout
from pyvale.vfm.loadregimes import resolve_load_regimes
from pyvale.vfm.solvepreparation import build_solve_preparation_context
from pyvale.vfm.slicewise_utils import SliceConfig
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def _fre_result(value: float) -> MetricResult:
    values = np.asarray([[value]], dtype=np.float64)
    return MetricResult(
        residual=values,
        additional_fields={
            "normalised_residual": values,
            "temporal_weights": np.ones(1),
        },
    )


def _egi_result(first: float, second: float | None = None) -> MetricResult:
    if second is None:
        second = first
    values = np.asarray([[[first, second]]], dtype=np.float64)
    return MetricResult(
        residual=values,
        additional_fields={
            "normalised_gap": values,
            "temporal_weights": np.ones(1),
        },
    )


def _prepared_objective(*, fre_limit: float = 1.1, broad_limit: float = 1.1):
    objective = GuardedEgiPrimaryObjective(
        GuardedEgiPrimaryConfig(
            fine_noise_scale=2.0,
            broad_noise_scale=4.0,
        )
    )
    regimes = resolve_load_regimes(np.zeros(1))
    gate = np.asarray([[[1.0, 0.0]]])
    objective._fre_layout = prepare_canonical_residual_layout(
        [_fre_result(1.0)], regimes,
        [ResidualBlockSpec("fre", 0, "all", "fre", residual_field="normalised_residual")],
    )
    objective._fine_layout = prepare_canonical_residual_layout(
        [_egi_result(1.0)], regimes,
        [ResidualBlockSpec("fine", 0, "all", "egi", residual_field="normalised_gap", noise_scale=2.0, observation_weights=gate)],
    )
    objective._broad_gated_layout = prepare_canonical_residual_layout(
        [_egi_result(1.0)], regimes,
        [ResidualBlockSpec("broad", 0, "all", "egi", residual_field="normalised_gap", noise_scale=4.0, observation_weights=gate)],
    )
    objective._broad_guard_layout = prepare_canonical_residual_layout(
        [_egi_result(1.0)], regimes,
        [ResidualBlockSpec("broad_guard", 0, "all", "egi", residual_field="normalised_gap")],
    )
    objective._fre_reference = _guard_reference(fre_limit / 1.1, 0.0, 0.1)
    objective._broad_reference = _guard_reference(broad_limit / 1.1, 0.0, 0.1)
    objective._prepared_solve = 0
    objective._recorder.start_solve(1)
    metrics = [
        SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="y", num_slices=3)),
        EquilibriumGapMetric(window_size=(3, 3)),
        EquilibriumGapMetric(window_size=(5, 5)),
    ]
    return objective, metrics


@pytest.mark.parametrize(
    ("parent", "floor", "expected"),
    [(2.0, 1.0, 2.0), (1.0, 2.0, 2.0), (2.0, 2.0, 2.0)],
)
def test_reference_is_maximum_of_parent_and_noise_floor(parent, floor, expected) -> None:
    reference = _guard_reference(parent, floor, 0.10)
    assert reference.reference == expected
    assert reference.limit == pytest.approx(1.10 * expected)


def test_ten_percent_boundary_passes_exactly_and_fails_above() -> None:
    assert _passes_limit(1.10, 1.10)
    assert not _passes_limit(1.10 + 1.0e-9, 1.10)


class _StressMetric(IMetric):
    def initialise(self, experiment_data) -> None:
        pass

    def evaluate(self, stress, constitutive_law, parameter_map_size, spatial_parameterisations, experiment_data):
        values = np.asarray([[float(stress[0, 0, 0, 0])]])
        return MetricResult(residual=values, additional_fields={"normalised_residual": values, "temporal_weights": np.ones(1)})


class _Law:
    def __init__(self) -> None:
        self.calls = 0

    def calculate_stress(self, strain, maps):
        self.calls += 1
        value = float(np.asarray(maps["p"])[0, 0])
        return np.full((1, 3, 1, 1), value)


def test_solve_context_preserves_explicit_accepted_parent() -> None:
    law = _Law()
    state = PhaseSpatialState({
        "p": [SpatialParameterisationHomogeneous(DegreeOfFreedom(1.0, 0.0, 3.0))]
    })
    context = build_solve_preparation_context(
        phase_index=1,
        solve_iteration=2,
        constitutive_law=law,
        parameter_map_size=np.asarray((1, 1), dtype=np.uint32),
        spatial_state=state,
        metrics=[_StressMetric()],
        experiment_data=SimpleNamespace(strain=np.zeros((1, 3, 1, 1))),
        parent_parameter_maps={"p": np.full((1, 1), 2.0)},
    )
    assert context.parameter_maps["p"][0, 0] == pytest.approx(1.0)
    assert context.parent_parameter_maps["p"][0, 0] == pytest.approx(2.0)
    assert context.stress[0, 0, 0, 0] == pytest.approx(1.0)
    assert context.parent_stress[0, 0, 0, 0] == pytest.approx(2.0)


class _PreparationContext:
    def __init__(self, parent_scale: float) -> None:
        self.phase_index = 1
        self.solve_iteration = 0
        self.experiment_data = SimpleNamespace(strain=np.zeros((1, 3, 1, 2)))
        self.stress = np.zeros((1, 3, 1, 2))
        self.constitutive_law = object()
        self.parameter_maps = {
            "yield_strength": np.ones((1, 2)),
            "hardening_modulus": np.ones((1, 2)),
        }
        self.parent_parameter_maps = {
            name: values.copy() for name, values in self.parameter_maps.items()
        }
        self.metric_results = (
            _fre_result(0.4), _egi_result(0.2), _egi_result(0.3),
        )
        self.parent_metric_results = (
            _fre_result(parent_scale),
            _egi_result(parent_scale),
            _egi_result(parent_scale),
        )
        self.metrics = tuple([
            SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="y", num_slices=3)),
            EquilibriumGapMetric(window_size=(3, 3)),
            EquilibriumGapMetric(window_size=(5, 5)),
        ])

    def prepare_residual_layout(self, load_regimes, specs):
        return prepare_canonical_residual_layout(
            self.metric_results, load_regimes, specs
        )


def test_prepare_solve_refreshes_parent_guards_but_freezes_gate(monkeypatch) -> None:
    sensitivity_calls = 0

    def sensitivities(strain, stress, law, maps, names, perturbation_factor):
        nonlocal sensitivity_calls
        sensitivity_calls += 1
        values = np.arange(stress.size, dtype=float).reshape(stress.shape) + 1.0
        return {name: SimpleNamespace(total=values) for name in names}

    monkeypatch.setattr(
        "pyvale.vfm.objectivefuncsensitivitygated.calculate_parameter_stress_sensitivities",
        sensitivities,
    )
    objective = GuardedEgiPrimaryObjective(
        GuardedEgiPrimaryConfig(2.0, 4.0)
    )
    first_context = _PreparationContext(parent_scale=1.0)
    first = objective.prepare_solve(first_context)
    first_reference = objective._fre_reference

    second_context = _PreparationContext(parent_scale=2.0)
    second_context.solve_iteration = 1
    second = objective.prepare_solve(second_context)

    assert first["fre_guard"]["parent"] == pytest.approx(1.0)
    assert second["fre_guard"]["parent"] == pytest.approx(2.0)
    assert first_reference.parent == pytest.approx(1.0)
    assert objective._fre_reference.parent == pytest.approx(2.0)
    assert sensitivity_calls == 1
    assert second["sensitivity_gate"]["refreshed"] is False


def test_only_parallel_candidate_clones_share_active_audit_recorder() -> None:
    objective, _ = _prepared_objective()
    generic_clone = copy.deepcopy(objective)
    candidate_clone = objective.clone_for_candidate_evaluation()

    assert generic_clone._recorder is not objective._recorder
    assert generic_clone._recorder.path is None
    assert candidate_clone._recorder is objective._recorder


def _install_metric_mocks(monkeypatch, *, fre: float, broad: tuple[float, float], fine: tuple[float, float], calls: list[str]):
    def force(self, *args, **kwargs):
        calls.append("fre")
        return _fre_result(fre)

    def egi(self, *args, **kwargs):
        role = "fine" if int(np.prod(self.window_size)) == 9 else "broad"
        calls.append(role)
        values = fine if role == "fine" else broad
        return SimpleNamespace(metric_result=_egi_result(*values))

    monkeypatch.setattr(SliceWiseForceReconstructionMetric, "evaluate", force)
    monkeypatch.setattr(EquilibriumGapMetric, "evaluate_equilibrium_gap", egi)


def test_fre_failure_short_circuits_broad_and_fine(monkeypatch) -> None:
    objective, metrics = _prepared_objective(fre_limit=1.0)
    calls: list[str] = []
    _install_metric_mocks(monkeypatch, fre=1.2, broad=(1.0, 1.0), fine=(1.0, 1.0), calls=calls)
    cost = objective.evaluate_candidate_stress(
        np.zeros((1, 3, 1, 1)), None, np.asarray((1, 1)), {}, metrics, None,
        stress_reconstruction_time_seconds=0.01,
    )
    assert np.isposinf(cost)
    assert calls == ["fre"]
    assert objective.last_result.rejection_reason == "FRE"


def test_broad_failure_is_evaluated_once_and_skips_fine(monkeypatch) -> None:
    objective, metrics = _prepared_objective(broad_limit=0.5)
    calls: list[str] = []
    _install_metric_mocks(monkeypatch, fre=0.5, broad=(1.0, 1.0), fine=(1.0, 1.0), calls=calls)
    cost = objective.evaluate_candidate_stress(
        np.zeros((1, 3, 1, 1)), None, np.asarray((1, 1)), {}, metrics, None,
        stress_reconstruction_time_seconds=0.01,
    )
    assert np.isposinf(cost)
    assert calls == ["fre", "broad"]
    assert objective.last_result.broad_gated_cost == pytest.approx(0.25)
    assert objective.last_result.rejection_reason == "BROAD"


def test_full_pass_reconstructs_stress_once_and_reuses_broad(monkeypatch) -> None:
    objective, metrics = _prepared_objective(fre_limit=2.0, broad_limit=2.0)
    calls: list[str] = []
    _install_metric_mocks(monkeypatch, fre=0.5, broad=(0.8, 0.8), fine=(1.0, 1.0), calls=calls)
    law = _Law()
    state = PhaseSpatialState({
        "p": [SpatialParameterisationHomogeneous(DegreeOfFreedom(1.0, 0.0, 3.0))]
    })
    cost = evaluate_candidate(
        np.asarray((1.0 / 3.0,)), law, np.asarray((1, 1), dtype=np.uint32),
        state, metrics, objective, SimpleNamespace(strain=np.zeros((1, 3, 1, 1))),
    )
    assert law.calls == 1
    assert calls == ["fre", "broad", "fine"]
    assert cost == pytest.approx(0.5 * (0.5 + 0.2))
    assert objective.last_result.rejection_reason == "NONE"


def test_guards_do_not_enter_scalar_and_references_stay_frozen() -> None:
    objective, _ = _prepared_objective(fre_limit=2.0, broad_limit=3.0)
    frozen_fre = objective._fre_reference
    frozen_broad = objective._broad_reference
    first = objective.evaluate([
        _fre_result(0.2), _egi_result(1.0, 9.0), _egi_result(0.8, 0.1),
    ])
    second = objective.evaluate([
        _fre_result(1.8), _egi_result(1.0, 2.0), _egi_result(0.8, 2.0),
    ])
    assert first == pytest.approx(second)
    assert first == pytest.approx(0.5 * (0.5 + 0.2))
    assert objective._fre_reference is frozen_fre
    assert objective._broad_reference is frozen_broad


def test_middle_egi_is_rejected() -> None:
    objective, metrics = _prepared_objective()
    with pytest.raises(ValueError, match="middle EGI is forbidden"):
        objective._validate_metrics(tuple([*metrics, EquilibriumGapMetric(window_size=(7, 7))]))


def test_retrospective_cd_primary_formula_matches_persisted_values() -> None:
    fine_scale = 1.7749920645783005e-6
    broad_scale = 1.974782950165619e-7
    # Persisted/recomputed C BF3 raw gated values from the 2026-09-01 check.
    fine_raw = 0.0002609406632781886 / 100.0
    broad_raw = 5.30631750981118 * broad_scale
    expected = 3.388206162963108
    actual = equal_mean_gated_egi_primary(
        fine_raw / fine_scale,
        broad_raw / broad_scale,
    )
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_noise_floor_uses_signed_perturbation_not_absolute_residual(monkeypatch) -> None:
    noise = MeasurementNoiseFloorConfig(
        mode=MeasurementNoiseMode.USER,
        seeds=(7,),
        strain_std_microstrain=(1.0, 1.0, 1.0),
        force_std_n=1.0,
        strain_filter_sigmas_mm_yx=((1.0, 1.0),) * 3,
    )
    objective = GuardedEgiPrimaryObjective(
        GuardedEgiPrimaryConfig(1.0, 1.0, measurement_noise=noise)
    )
    regimes = resolve_load_regimes(np.zeros(1))
    objective._fre_layout = prepare_canonical_residual_layout(
        [_fre_result(5.0)], regimes,
        [ResidualBlockSpec("fre", 0, "all", "fre", residual_field="normalised_residual")],
    )
    objective._broad_guard_layout = prepare_canonical_residual_layout(
        [_egi_result(10.0)], regimes,
        [ResidualBlockSpec("broad", 0, "all", "egi", residual_field="normalised_gap")],
    )
    metrics = [
        SliceWiseForceReconstructionMetric(slice_config=SliceConfig(axis="y", num_slices=3)),
        EquilibriumGapMetric(window_size=(3, 3)),
        EquilibriumGapMetric(window_size=(5, 5)),
    ]
    monkeypatch.setattr(
        "pyvale.vfm.objectivefuncguardedegiprimary.measurement_noise_realisation",
        lambda experiment, config, seed, force_axis: experiment,
    )
    monkeypatch.setattr(SliceWiseForceReconstructionMetric, "initialise", lambda self, experiment: None)
    monkeypatch.setattr(EquilibriumGapMetric, "initialise", lambda self, experiment: None)
    monkeypatch.setattr(
        SliceWiseForceReconstructionMetric,
        "evaluate_force_recon_error",
        lambda self, stress, experiment: SimpleNamespace(metric_result=_fre_result(7.0)),
    )
    monkeypatch.setattr(
        EquilibriumGapMetric,
        "evaluate_equilibrium_gap",
        lambda self, stress, include_diagnostics=True: SimpleNamespace(metric_result=_egi_result(13.0)),
    )
    context = SimpleNamespace(
        experiment_data=SimpleNamespace(strain=np.zeros((1, 3, 1, 1))),
        constitutive_law=SimpleNamespace(calculate_stress=lambda strain, maps: np.zeros((1, 3, 1, 1))),
        parent_parameter_maps={"p": np.ones((1, 1))},
        metrics=tuple(metrics),
    )
    fre, broad, diagnostics = objective._measurement_noise_floors(
        context,
        (_fre_result(5.0), _egi_result(0.0), _egi_result(10.0)),
    )
    assert fre == pytest.approx(2.0)
    assert broad == pytest.approx(3.0)
    assert diagnostics["floor_definition"] == "empirical Q95 of RMS(r_noisy - r_parent)"


def test_noise_modes_validate_and_persist_metadata() -> None:
    parent = MeasurementNoiseFloorConfig(mode="parent-only")
    calibrated = MeasurementNoiseFloorConfig(
        mode="calibrated", seeds=(1, 2),
        strain_std_microstrain=(10.0, 20.0, 30.0), force_std_n=0.2,
        strain_filter_sigmas_mm_yx=((0.1, 0.1),) * 3,
        model_source="calibration.json",
    )
    user = MeasurementNoiseFloorConfig(
        mode="user", seeds=(3,),
        strain_std_microstrain=(4.0, 5.0, 6.0), force_std_n=0.3,
        strain_filter_sigmas_mm_yx=((0.2, 0.2),) * 3,
        model_source="correlation.json",
    )
    assert parent.metadata()["mode"] == "parent-only"
    assert calibrated.metadata()["number_of_realisations"] == 2
    assert user.metadata()["strain_std_microstrain"] == [4.0, 5.0, 6.0]
    with pytest.raises(ValueError, match="IID fallback"):
        MeasurementNoiseFloorConfig(
            mode="user", seeds=(1,), strain_std_microstrain=(1.0, 1.0, 1.0),
            force_std_n=1.0,
        )
