import numpy as np
import numpy.typing as npt

from pyvale.vfm.dof import DegreeOfFreedom


def normalise_degree_of_freedom(
    degree_of_freedom: DegreeOfFreedom,
) -> float:
    """
    Normalise a single degree of freedom to ``[0, 1]``.

    Parameters
    ----------
    degree_of_freedom : DegreeOfFreedom
        DOF with ``value``, ``lower_bound``, and ``upper_bound``

    Returns
    -------
    float
        Normalised value in ``[0, 1]``
    """
    return (
        (degree_of_freedom.value - degree_of_freedom.lower_bound)
        / (degree_of_freedom.upper_bound - degree_of_freedom.lower_bound)
    )


def normalise_degrees_of_freedom(
    degrees_of_freedom: list[DegreeOfFreedom],
) -> npt.NDArray[np.float64]:
    """
    Normalise a list of degrees of freedom to ``[0, 1]``.

    Parameters
    ----------
    degrees_of_freedom : list[DegreeOfFreedom]
        One or more DOFs to normalise

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of normalised values
    """
    normalised_dofs = []

    for dof in degrees_of_freedom:
        normalised_dofs.append(normalise_degree_of_freedom(dof))

    return np.array(normalised_dofs)


def denormalise_degree_of_freedom(
    normalised_value: float,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """
    Reverse the normalisation from ``[0, 1]`` back to physical units

    Parameters
    ----------
    normalised_value : float
        Value in ``[0, 1]``
    lower_bound : float
        Physical lower bound
    upper_bound : float
        Physical upper bound

    Returns
    -------
    float
        Denormalised value in ``[lower_bound, upper_bound]``.
    """
    return ((upper_bound - lower_bound) * normalised_value) + lower_bound


def denormalise_degrees_of_freedom(
    normalised_values: npt.NDArray[np.float64],
    lower_bounds: npt.NDArray[np.float64],
    upper_bounds: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Reverse the normalisation for an array of values

    Parameters
    ----------
    normalised_values : npt.NDArray[np.float64]
        Values in ``[0, 1]``
    lower_bounds : npt.NDArray[np.float64]
        Physical lower bounds per DOF
    upper_bounds : npt.NDArray[np.float64]
        Physical upper bounds per DOF

    Returns
    -------
    npt.NDArray[np.float64]
        1D array of denormalised values
    """
    return ((upper_bounds - lower_bounds) * normalised_values) + lower_bounds
