from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    DegreeOfFreedom,
    SpatialParameterisation,
)


@dataclass(slots=True)
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

    def pack_degrees_of_freedom(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]
    ]:
        return (
            np.array([self.value.value]),
            np.array([self.value.lower_bound]),
            np.array([self.value.upper_bound])
        )

    def update_from_packed_degrees_of_freedom(
        self,
        degrees_of_freedom: npt.NDArray[np.float64]
    ) -> None:
        self.value.value = degrees_of_freedom[0]
