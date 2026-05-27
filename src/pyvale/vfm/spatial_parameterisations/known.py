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
    ISpatialParameterisation,
)


@dataclass(slots=True, init=False)
class KnownSpatialParameterisation(ISpatialParameterisation):
    value: npt.NDArray[np.float64] | None = None

    @property
    def num_degrees_of_freedom(self) -> int:
        return 0

    def update_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        self.value = constitutive_parameter.value

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        # TODO: value error if param value not the right size

        return self.value

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        return []

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        return
