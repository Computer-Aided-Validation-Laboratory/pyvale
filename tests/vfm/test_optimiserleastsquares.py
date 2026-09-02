from types import SimpleNamespace

import numpy as np
import pytest

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.optimiser import evaluate_candidate
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def test_least_squares_keeps_normalised_dofs_within_bounds(monkeypatch) -> None:
    def residual(candidate, *_args):
        return np.asarray([candidate[0] - 1.5])

    monkeypatch.setattr(
        "pyvale.vfm.optimiserleastsquares.evaluate_candidate",
        residual,
    )
    parameterisation = SpatialParameterisationHomogeneous(
        DegreeOfFreedom(5.0, 0.0, 10.0)
    )
    result = OptimiserLeastSquares(max_evaluations=20).optimise(
        None,
        np.asarray((1, 1), dtype=np.uint32),
        {"parameter": [parameterisation]},
        [],
        None,
        None,
    )

    final_parameterisation = result.spatial_parameterisations["parameter"][0]
    assert isinstance(final_parameterisation, SpatialParameterisationHomogeneous)
    assert isinstance(final_parameterisation.value, DegreeOfFreedom)
    assert final_parameterisation.value.value == pytest.approx(10.0)


def test_candidate_parameter_maps_can_be_projected_to_physical_bounds(
    monkeypatch,
) -> None:
    captured = {}

    class RecordingLaw:
        def calculate_stress(self, strain, parameter_maps):
            captured.update(parameter_maps)
            return np.zeros_like(strain)

    class VectorObjective:
        def evaluate(self, _metric_results):
            return np.asarray([0.0])

    monkeypatch.setattr(
        "pyvale.vfm.optimiser.evaluate_metrics",
        lambda *_args, **_kwargs: [],
    )
    state = PhaseSpatialState({
        "hardening_modulus": [
            SpatialParameterisationHomogeneous(
                DegreeOfFreedom(-5.0, -10.0, 10.0)
            )
        ]
    })

    evaluate_candidate(
        np.asarray([0.0]),
        RecordingLaw(),
        np.asarray((2, 2), dtype=np.uint32),
        state,
        [],
        VectorObjective(),
        SimpleNamespace(strain=np.zeros((1, 3, 2, 2))),
        {"hardening_modulus": (500.0, 10_000.0)},
    )

    np.testing.assert_allclose(captured["hardening_modulus"], 500.0)
