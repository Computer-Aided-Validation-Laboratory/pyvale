import numpy as np

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparamhomogeneous import SpatialParameterisationHomogeneous


def test_default_homogeneous_dof_uses_parameter_map_mean_and_bounds() -> None:
    parameter = ConstitutiveParameter(
        value=np.full((2, 3), 420.0),
        lower_bound=300.0,
        upper_bound=500.0,
    )
    parameterisation = SpatialParameterisationHomogeneous()

    parameterisation.initialise_from_constitutive_parameter(parameter)

    assert isinstance(parameterisation.value, DegreeOfFreedom)
    assert parameterisation.value.value == 420.0
    assert parameterisation.value.lower_bound == 300.0
    assert parameterisation.value.upper_bound == 500.0
    assert parameterisation.get_num_degrees_of_freedom() == 1


def test_float_homogeneous_value_is_initial_value_for_active_dof() -> None:
    parameter = ConstitutiveParameter(
        value=np.full((2, 3), 420.0),
        lower_bound=300.0,
        upper_bound=500.0,
    )
    parameterisation = SpatialParameterisationHomogeneous(360.0)

    parameterisation.initialise_from_constitutive_parameter(parameter)

    assert isinstance(parameterisation.value, DegreeOfFreedom)
    assert parameterisation.value.value == 360.0
    assert parameterisation.value.lower_bound == 300.0
    assert parameterisation.value.upper_bound == 500.0
    assert parameterisation.get_num_degrees_of_freedom() == 1
    assert np.all(parameterisation.to_map(np.array((2, 3), dtype=np.uint32)) == 360.0)


def test_explicit_homogeneous_dof_retains_its_value_and_bounds() -> None:
    parameter = ConstitutiveParameter(
        value=np.full((2, 3), 420.0),
        lower_bound=300.0,
        upper_bound=500.0,
    )
    parameterisation = SpatialParameterisationHomogeneous(
        DegreeOfFreedom(365.0, 350.0, 380.0)
    )

    parameterisation.initialise_from_constitutive_parameter(parameter)

    assert isinstance(parameterisation.value, DegreeOfFreedom)
    assert parameterisation.value.value == 365.0
    assert parameterisation.value.lower_bound == 350.0
    assert parameterisation.value.upper_bound == 380.0
