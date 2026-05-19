import copy
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.normalisation import denormalise_degrees_of_freedom
from pyvale.vfm.spatial_parameterisations.degree_of_freedom import (
    DegreeOfFreedom,
)


# For the update from dof methods, we assume the order of the list/array is
# the same as the order we provided when we collected the dofs
class ISpatialParameterisation(ABC):
    @property
    @abstractmethod
    def num_degrees_of_freedom(self) -> int:
        pass

    @abstractmethod
    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        pass

    @abstractmethod
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        pass

    @abstractmethod
    def update_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        pass


def unpack_spatial_parameterisations(
    reference_spatial_parameterisations: dict[str, ISpatialParameterisation],
    normalised_degrees_of_freedom: npt.NDArray[np.float64],
) -> dict[str, ISpatialParameterisation]:
    lower_bounds = []
    upper_bounds = []

    for sp in reference_spatial_parameterisations.values():
        for dof in sp.collect_degrees_of_freedom():
            lower_bounds.append(dof.lower_bound)
            upper_bounds.append(dof.upper_bound)

    degrees_of_freedom = denormalise_degrees_of_freedom(
        normalised_degrees_of_freedom,
        np.array(lower_bounds),
        np.array(upper_bounds)
    )

    unpacked_spatial_parameterisations = {}

    index = 0
    for param_name, sp in reference_spatial_parameterisations.items():
        num_dofs = sp.num_degrees_of_freedom

        if num_dofs == 0:
            unpacked_spatial_parameterisations[param_name] = sp
            continue

        unpacked_sp = copy.deepcopy(sp)

        sp_dofs = degrees_of_freedom[index:index + num_dofs]

        unpacked_sp.update_from_degrees_of_freedom(sp_dofs)
        unpacked_spatial_parameterisations[param_name] = unpacked_sp

        index += num_dofs

    return unpacked_spatial_parameterisations
