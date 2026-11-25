import numpy as np
import pytest

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.optimiserleastsquares import OptimiserLeastSquares
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
