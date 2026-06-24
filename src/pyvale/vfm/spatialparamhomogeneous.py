from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class HomogeneousSpatialParameterisation(ISpatialParameterisation):
    value: float | DegreeOfFreedom | None = None

    def get_num_degrees_of_freedom(self) -> int:
        if isinstance(self.value, DegreeOfFreedom):
            return 1
        else:
            return 0

    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        if self.value is None or isinstance(self.value, DegreeOfFreedom):
            self.value =  DegreeOfFreedom(
                float(np.nanmean(constitutive_parameter.map)),
                constitutive_parameter.lower_bound,
                constitutive_parameter.upper_bound,
            )
        else:
            self.value = float(np.nanmean(constitutive_parameter.map))

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        if self.value is None:
            raise RuntimeError(
                "self.value is None, initialise_from_constitutive_parameter"
                "must be called before to_map"
            )

        if isinstance(self.value, DegreeOfFreedom):
            value = self.value.value
        else:
            value = self.value

        return np.full((size[0], size[1]), value)

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        if isinstance(self.value, DegreeOfFreedom):
            return [copy(self.value)]
        else:
            return []

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        if len(degrees_of_freedom) != self.get_num_degrees_of_freedom():
            raise ValueError(
                f"expected {self.get_num_degrees_of_freedom()} degrees of "
                f"freedom, got {len(degrees_of_freedom)}"
            )

        if isinstance(degrees_of_freedom, list):
            if isinstance(self.value, DegreeOfFreedom):
                self.value = degrees_of_freedom[0]
        else:
            if isinstance(self.value, DegreeOfFreedom):
                self.value.value = degrees_of_freedom[0]

