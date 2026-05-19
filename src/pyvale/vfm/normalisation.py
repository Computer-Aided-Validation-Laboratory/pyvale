import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.degree_of_freedom import (
    DegreeOfFreedom,
)


def normalise_degree_of_freedom(
    degree_of_freedom: DegreeOfFreedom
) -> float:
    return (
        (degree_of_freedom.value - degree_of_freedom.lower_bound)
        / (degree_of_freedom.upper_bound - degree_of_freedom.lower_bound)
    )

def normalise_degrees_of_freedom(
    degrees_of_freedom: list[DegreeOfFreedom]
) -> npt.NDArray[np.float64]:
    normalised_dofs = []

    for dof in degrees_of_freedom:
        normalised_dofs.append(normalise_degree_of_freedom(dof))

    return np.array(normalised_dofs)

def denormalise_degree_of_freedom(
    normalised_value: float,
    lower_bound: float,
    upper_bound: float
) -> float:
    return ((upper_bound - lower_bound) * normalised_value) + lower_bound

def denormalise_degrees_of_freedom(
    degrees_of_freedom: npt.NDArray[np.float64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return ((upper_bounds - lower_bounds) * degrees_of_freedom) + lower_bounds
