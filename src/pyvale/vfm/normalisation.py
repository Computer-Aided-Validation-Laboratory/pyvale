import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    DegreeOfFreedom,
)


def normalise_degree_of_freedom(
    degree_of_freedom: DegreeOfFreedom
) -> float:
    ...


def normalise_degrees_of_freedom(
    degrees_of_freedom: list[DegreeOfFreedom]
) -> npt.NDArray[np.float64]:
    ...

def denormalise_degree_of_freedom(
    degree_of_freedom: float,
    lower_bound: float,
    upper_bound: float
) -> float:
    ...

def denormalise_degrees_of_freedom(
    degrees_of_freedom: npt.NDArray[np.float64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    ...
