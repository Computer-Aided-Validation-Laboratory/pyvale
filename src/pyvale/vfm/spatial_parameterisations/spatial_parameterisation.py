import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)


@dataclass(slots=True)
class DegreeOfFreedom:
    value: float
    lower_bound: float
    upper_bound: float


# For the update from dof methods, we assume the order of the list/array is
# the same as the order we provided when we collected the dofs
class SpatialParameterisation(ABC):
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
    reference_spatial_parameterisations: dict[str, SpatialParameterisation],
    degrees_of_freedom: npt.NDArray[np.float64],
) -> dict[str, SpatialParameterisation]:
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
