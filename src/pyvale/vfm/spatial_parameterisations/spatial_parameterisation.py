from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class DegreeOfFreedom:
    value: float
    lower_bound: float
    upper_bound: float


# For both update from dof methods, we assume the order of the list/array is
# the same as the order we provided when we collected/packed the dofs
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
        degrees_of_freedom: list[DegreeOfFreedom]
    ) -> None:
        pass

    # returns a tuple of (array of dofs, array of lower bounds, array of upper bounds)
    @abstractmethod
    def pack_degrees_of_freedom(
        self,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64]
    ]:
        pass

    @abstractmethod
    def update_from_packed_degrees_of_freedom(
        self,
        degrees_of_freedom: npt.NDArray[np.float64]
    ) -> None:
        pass
