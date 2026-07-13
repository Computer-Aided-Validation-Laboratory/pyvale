import copy
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.normalisation import denormalise_degrees_of_freedom
from pyvale.vfm.dof import DegreeOfFreedom


class ISpatialParameterisation(ABC):
    """
    Interface (abstract base class) for a spatial parameterisation.

    Maps constitutive parameter values onto the 2D DIC grid. Concrete
    implementations define how the parameter varies in space and how
    those spatial degrees of freedom are collected, updated, and converted
    back to maps
    """

    @abstractmethod
    def get_num_degrees_of_freedom(self) -> int:
        """
        Number of adjustable degrees of freedom in this parameterisation.

        Returns
        -------
        int
            Count of degrees of freedom
        """
        pass

    @abstractmethod
    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter,
    ) -> None:
        """
        Initialise this parameterisation from a constitutive parameter.

        Populates the internal degrees of freedom using the initial value
        and bounds stored in the parameter.

        Parameters
        ----------
        constitutive_parameter : ConstitutiveParameter
            Parameter with initial value, lower bound, and upper bound
        """
        pass

    @abstractmethod
    def to_map(
        self,
        size: npt.NDArray[np.uint32],
    ) -> npt.NDArray[np.float64]:
        """
        Generate a 2D map from the current parameterisation.

        Parameters
        ----------
        size : npt.NDArray[np.uint32]
            Target spatial shape ``(y, x)``

        Returns
        -------
        npt.NDArray[np.float64]
            Parameter map with shape ``(y, x)``
        """
        pass

    @abstractmethod
    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        """
        Export a copy of the degrees of freedom for the optimiser.

        Returns
        -------
        list[DegreeOfFreedom]
            Copies of each degree of freedom
        """
        pass

    @abstractmethod
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        """
        Update internal state from optimiser-supplied degrees of freedom.

        Parameters
        ----------
        degrees_of_freedom : list[DegreeOfFreedom] | npt.NDArray[np.float64]
            New values for each degree of freedom, in the same order as
            returned by `collect_degrees_of_freedom()`.
        """
        pass


def evaluate_parameterisations_to_map(
    spatial_parameterisations: list[ISpatialParameterisation],
    size: npt.NDArray[np.uint32],
) -> npt.NDArray[np.float64]:
    """
    Evaluate a parameter map from a list of spatial parameterisations.

    Each parameterisation is evaluated in list (definition) order and the
    resulting maps are summed together.

    Parameters
    ----------
    spatial_parameterisations : list[ISpatialParameterisation]
        Parameterisations to evaluate and sum, in definition order
    size : npt.NDArray[np.uint32]
        Target spatial shape ``(y, x)``

    Returns
    -------
    npt.NDArray[np.float64]
        Summed parameter map with shape ``(y, x)``
    """
    return np.sum(
        [sp.to_map(size) for sp in spatial_parameterisations],
        axis=0,
    )


def collect_degrees_of_freedom(
    spatial_parameterisations: list[ISpatialParameterisation],
) -> list[DegreeOfFreedom]:
    """
    Collect the degrees of freedom from a list of spatial parameterisations.

    Degrees of freedom are concatenated in list (definition) order.
    """
    degrees_of_freedom: list[DegreeOfFreedom] = []
    for sp in spatial_parameterisations:
        degrees_of_freedom.extend(sp.collect_degrees_of_freedom())
    return degrees_of_freedom


def get_num_degrees_of_freedom(
    spatial_parameterisations: list[ISpatialParameterisation],
) -> int:
    """
    Total number of degrees of freedom across a list of parameterisations.
    """
    return sum(
        sp.get_num_degrees_of_freedom() for sp in spatial_parameterisations
    )


def update_from_degrees_of_freedom(
    spatial_parameterisations: list[ISpatialParameterisation],
    degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
) -> None:
    """
    Distribute a flat degrees of freedom sequence across a list of
    parameterisations.

    The degrees of freedom are consumed in list (definition) order, each
    parameterisation taking as many as it reports via
    ``get_num_degrees_of_freedom()``.
    """
    index = 0
    for sp in spatial_parameterisations:
        num_dofs = sp.get_num_degrees_of_freedom()
        if num_dofs == 0:
            continue
        sp.update_from_degrees_of_freedom(
            degrees_of_freedom[index:index + num_dofs]
        )
        index += num_dofs


def initialise_parameterisations_from_constitutive_parameter(
    spatial_parameterisations: list[ISpatialParameterisation],
    constitutive_parameter: ConstitutiveParameter,
    size: npt.NDArray[np.uint32],
) -> None:
    """
    Initialise a list of spatial parameterisations against a running residual.

    The first parameterisation is initialised from the full constitutive
    parameter map (using its original bounds). Its contribution is subtracted
    from the map and the remaining residual is passed to the next
    parameterisation, and so on in list (definition) order.

    Residual parameterisations after the first are bounded symmetrically by
    ``+/- (upper_bound - lower_bound)`` about zero, since a residual is a
    correction that may be positive or negative.

    Parameters
    ----------
    spatial_parameterisations : list[ISpatialParameterisation]
        Parameterisations to initialise, in definition order
    constitutive_parameter : ConstitutiveParameter
        Parameter with initial value map, lower bound, and upper bound
    size : npt.NDArray[np.uint32]
        Target spatial shape ``(y, x)`` used to evaluate each contribution
    """
    residual_map = np.array(constitutive_parameter.map, dtype=np.float64)
    lower_bound = constitutive_parameter.lower_bound
    upper_bound = constitutive_parameter.upper_bound
    span = upper_bound - lower_bound

    for i, sp in enumerate(spatial_parameterisations):
        if i == 0:
            residual_parameter = ConstitutiveParameter(
                residual_map,
                lower_bound,
                upper_bound,
            )
        else:
            residual_parameter = ConstitutiveParameter(
                residual_map,
                -span,
                span,
            )

        sp.initialise_from_constitutive_parameter(residual_parameter)
        residual_map = residual_map - sp.to_map(size)


def unpack_spatial_parameterisations(
    reference_spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
    normalised_degrees_of_freedom: npt.NDArray[np.float64],
) -> dict[str, list[ISpatialParameterisation]]:
    """
    Create a copy of spatial parameterisations with updated degrees of freedom.

    Denormalises the degrees of freedom vector, copies each reference
    parameterisation list, and applies the new degrees of freedom to the copy.

    Parameters
    ----------
    reference_spatial_parameterisations : dict[str, list[ISpatialParameterisation]]
        Reference parameterisation lists keyed by parameter name
    normalised_degrees_of_freedom : npt.NDArray[np.float64]
        Normalised degrees of freedom

    Returns
    -------
    dict[str, list[ISpatialParameterisation]]
        A copy of the reference spatial parameterisations with
        updated degrees of freedom
    """
    lower_bounds = []
    upper_bounds = []

    for sps in reference_spatial_parameterisations.values():
        for dof in collect_degrees_of_freedom(sps):
            lower_bounds.append(dof.lower_bound)
            upper_bounds.append(dof.upper_bound)

    degrees_of_freedom = denormalise_degrees_of_freedom(
        normalised_degrees_of_freedom,
        np.array(lower_bounds),
        np.array(upper_bounds),
    )

    unpacked_spatial_parameterisations = {}

    index = 0
    for param_name, sps in reference_spatial_parameterisations.items():
        num_dofs = get_num_degrees_of_freedom(sps)

        if num_dofs == 0:
            unpacked_spatial_parameterisations[param_name] = sps
            continue

        unpacked_sps = copy.deepcopy(sps)

        sp_dofs = degrees_of_freedom[index:index + num_dofs]

        update_from_degrees_of_freedom(unpacked_sps, sp_dofs)
        unpacked_spatial_parameterisations[param_name] = unpacked_sps

        index += num_dofs

    return unpacked_spatial_parameterisations
