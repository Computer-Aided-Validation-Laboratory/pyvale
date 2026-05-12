from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    DegreeOfFreedom,
    SpatialParameterisation,
)


@dataclass(slots=True)
class KnownSpatialParameterisation(SpatialParameterisation):
    value: npt.NDArray[np.float64]

    @property
    def num_degrees_of_freedom(self) -> int:
        return 0

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
        degrees_of_freedom: list[DegreeOfFreedom]
    ) -> None:
        return

    def pack_degrees_of_freedom(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]
    ]:
        return (np.array([]), np.array([]), np.array([]))

    def update_from_packed_degrees_of_freedom(
        self,
        degrees_of_freedom: npt.NDArray[np.float64]
    ) -> None:
        return
