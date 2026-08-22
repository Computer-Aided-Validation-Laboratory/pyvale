import numpy as np

from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparam import PhaseSpatialState
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def test_phase_spatial_state_round_trips_log_scaled_degree_of_freedom() -> None:
    parameterisation = SpatialParameterisationHomogeneous(
        DegreeOfFreedom(100.0, 1.0, 10_000.0, scaling="log")
    )
    state = PhaseSpatialState({"yield_strength": [parameterisation]})

    normalised = state.collect_normalised_degrees_of_freedom()

    assert np.allclose(normalised, [0.5])
    state.update_from_normalised_degrees_of_freedom(np.asarray([0.75]))
    assert np.isclose(parameterisation.value.value, 1_000.0)
