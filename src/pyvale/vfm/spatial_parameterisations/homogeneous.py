from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.spatial_parameterisations.degree_of_freedom import (
    DegreeOfFreedom,
)
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    SpatialParameterisation,
)


@dataclass(slots=True, init=False)
class HomogeneousSpatialParameterisation(SpatialParameterisation):
    value: DegreeOfFreedom

    @property
    def num_degrees_of_freedom(self) -> int:
        return 1

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        return np.full((size[0], size[1]), self.value.value)

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        return [copy(self.value)]

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        # TODO: length list check to match num dofs
        if isinstance(degrees_of_freedom, list):
            self.value = degrees_of_freedom[0]
        else:
            self.value.value = degrees_of_freedom[0]

    def update_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        self.value =  DegreeOfFreedom(
            constitutive_parameter.value[0, 0],
            constitutive_parameter.lower_bound,
            constitutive_parameter.upper_bound,
        )
