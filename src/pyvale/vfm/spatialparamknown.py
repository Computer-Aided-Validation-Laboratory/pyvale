from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class SpatialParameterisationKnown(ISpatialParameterisation):
    value: npt.NDArray[np.float64] | None = None

    def get_num_degrees_of_freedom(self) -> int:
        return 0

    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        self.value = constitutive_parameter.map

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        if self.value is None:
            raise RuntimeError(
                "self.value is None, initialise_from_constitutive_parameter"
                "must be called before to_map"
            )

        if self.value.shape != (size[0], size[1]):
            raise ValueError(
                f"parameter map shape {self.value.shape} does not match "
                f"expected size ({size[0]}, {size[1]})"
            )

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