from __future__ import annotations

import numpy as np

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.identification import run_identification
from pyvale.vfm.identificationconfig import IdentificationConfig, IdentificationPhase
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.objectivefuncvector import VectorWeightedObjective
from optimiserscalarprofile import solve_scalar_profile
from optimiserslicewiseprofile import SliceWiseIndependentProfileOptimiser
from pyvale.vfm.slicewise_utils import SliceConfig
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation, SupportSlice
from vfm.slicewise_validation_case import (
    ensure_rectangle_slicewise_yield_case,
    make_identification_parameters,
)


def test_profile_recovers_interior_minimum_without_using_a_start() -> None:
    result = solve_scalar_profile(
        lambda yield_mpa: np.asarray([(yield_mpa - 620.0) / 25.0, (yield_mpa - 620.0) / 40.0]),
        (200.0, 1200.0),
        scan_points=21,
    )
    assert result.classification == "FINITE_CONDITIONAL_ESTIMATE"
    np.testing.assert_allclose(result.optimum, 620.0, atol=1.0e-3)


def test_all_elastic_high_yield_plateau_is_one_sided() -> None:
    result = solve_scalar_profile(
        lambda yield_mpa: np.asarray([max(0.0, 500.0 - yield_mpa)]),
        (200.0, 1200.0),
        scan_points=21,
    )
    assert result.classification == "ONE_SIDED_OR_ELASTIC_PLATEAU"
    assert result.plateau_onset is not None
    assert result.plateau_onset >= 500.0


def test_truncated_interval_is_bound_limited() -> None:
    result = solve_scalar_profile(
        lambda yield_mpa: np.asarray([(yield_mpa - 900.0) / 50.0]),
        (200.0, 700.0),
        scan_points=21,
    )
    assert result.classification == "BOUND_LIMITED"
    assert result.upper_bound_active
    np.testing.assert_allclose(result.optimum, 700.0)


def test_saved_residual_is_fresh_independent_residual() -> None:
    calls: list[float] = []

    def residual(yield_mpa: float) -> np.ndarray:
        calls.append(yield_mpa)
        return np.asarray([yield_mpa - 550.0, 2.0 * (yield_mpa - 550.0)])

    result = solve_scalar_profile(residual, (200.0, 1200.0), scan_points=21)
    np.testing.assert_allclose(result.residual, residual(result.optimum))
    np.testing.assert_allclose(result.minimum_cost, 0.5 * result.residual @ result.residual)
    assert len(calls) == result.evaluation_count + 1


def test_profile_optimiser_recovers_existing_point_history_fixture() -> None:
    case = ensure_rectangle_slicewise_yield_case()
    experiment = ExperimentData.load_from_file(case.experiment_data_file)
    support = SupportSlice(slice_config=SliceConfig(axis="x", num_slices=4))
    phase = IdentificationPhase(
        spatial_parameterisations={
            "elastic_modulus": [SpatialParameterisationKnown()],
            "poissons_ratio": [SpatialParameterisationKnown()],
            "yield_strength": [SliceWiseSpatialParameterisation(support=support)],
            "hardening_modulus": [SpatialParameterisationKnown()],
        },
        metrics=[SliceWiseForceReconstructionMetric(support=support)],
        objective_function=VectorWeightedObjective(),
        optimiser=SliceWiseIndependentProfileOptimiser(scan_points=21),
    )
    result = run_identification(
        experiment,
        IdentificationConfig(
            constitutive_law=IsotropicVonMisesElastoplasticity(HardeningLinear()),
            parameters=make_identification_parameters(
                np.asarray(experiment.specimen_geometry.x.shape, dtype=np.uint32)
            ),
            phases=[phase],
        ),
    )
    with np.load(case.known_parameter_maps_file) as known:
        np.testing.assert_allclose(
            result.parameter_maps["yield_strength"],
            known["yield_strength"],
            atol=1.0e-3,
        )
    children = result.history.phases[0].solve_results[-1].children
    assert {child.status for child in children} == {"FINITE_CONDITIONAL_ESTIMATE"}
