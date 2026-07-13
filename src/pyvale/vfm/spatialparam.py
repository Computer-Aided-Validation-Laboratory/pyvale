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


def unpack_spatial_parameterisations(
    reference_spatial_parameterisations: dict[str, ISpatialParameterisation],
    normalised_degrees_of_freedom: npt.NDArray[np.float64],
) -> dict[str, ISpatialParameterisation]:
    """
    Create a copy of spatial parameterisations with updated degrees of freedom.

    Denormalises the degrees of freedom vector, copies each reference
    parameterisation, and applies the new degrees of freedom to the copy.

    Parameters
    ----------
    reference_spatial_parameterisations : dict[str, ISpatialParameterisation]
        Reference parameterisations keyed by parameter name
    normalised_degrees_of_freedom : npt.NDArray[np.float64]
        Normalised degrees of freedom

    Returns
    -------
    dict[str, ISpatialParameterisation]
        A copy of the reference spatial parameterisations with
        updated degrees of freedom
    """
    lower_bounds = []
    upper_bounds = []

    for sp in reference_spatial_parameterisations.values():
        for dof in sp.collect_degrees_of_freedom():
            lower_bounds.append(dof.lower_bound)
            upper_bounds.append(dof.upper_bound)

    degrees_of_freedom = denormalise_degrees_of_freedom(
        normalised_degrees_of_freedom,
        np.array(lower_bounds),
        np.array(upper_bounds),
    )

    unpacked_spatial_parameterisations = {}

    index = 0
    for param_name, sp in reference_spatial_parameterisations.items():
        num_dofs = sp.get_num_degrees_of_freedom()

        if num_dofs == 0:
            unpacked_spatial_parameterisations[param_name] = sp
            continue

        unpacked_sp = copy.deepcopy(sp)

        sp_dofs = degrees_of_freedom[index:index + num_dofs]

        unpacked_sp.update_from_degrees_of_freedom(sp_dofs)
        unpacked_spatial_parameterisations[param_name] = unpacked_sp

        index += num_dofs

    return unpacked_spatial_parameterisations
