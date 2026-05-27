from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.spatial_parameterisations.degree_of_freedom import (
    DegreeOfFreedom,
)


# TODO: maybe scrap this and use duck typing with an ORd type
class IBasisFunctionKernel(ABC):
    @property
    @abstractmethod
    def num_degrees_of_freedom(self) -> int:
        pass

    @abstractmethod
    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        pass

    # We assume the order of the list/array is
    # the same as the order we provided when we collected the dofs
    @abstractmethod
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        pass


@dataclass(slots=True)
class UnivariateBasisFunctionKernel(IBasisFunctionKernel):
    x: float | DegreeOfFreedom
    y: float | DegreeOfFreedom
    height: float | DegreeOfFreedom
    variance: float | DegreeOfFreedom

    @property
    def num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.x, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.y, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.height, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.variance, DegreeOfFreedom):
            num_dofs += 1

        return num_dofs

    def collect_degrees_of_freedom(
        self
    ) -> list[DegreeOfFreedom]:
        dofs = []

        if isinstance(self.x, DegreeOfFreedom):
            dofs.append(copy(self.x))
        if isinstance(self.y, DegreeOfFreedom):
            dofs.append(copy(self.y))
        if isinstance(self.height, DegreeOfFreedom):
            dofs.append(copy(self.height))
        if isinstance(self.variance, DegreeOfFreedom):
            dofs.append(copy(self.variance))

        return dofs

    # TODO: length check to match num dofs
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        dof_index = 0

        if isinstance(self.x, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.x.value = dof.value
            else:
                self.x.value = dof

            dof_index += 1
        if isinstance(self.y, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.y.value = dof.value
            else:
                self.y.value = dof

            dof_index += 1
        if isinstance(self.height, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.height.value = dof.value
            else:
                self.height.value = dof

            dof_index += 1
        if isinstance(self.variance, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.variance.value = dof.value
            else:
                self.variance.value = dof


@dataclass(slots=True)
class BivariateBasisFunctionKernel(IBasisFunctionKernel):
    x: float | DegreeOfFreedom
    y: float | DegreeOfFreedom
    height: float | DegreeOfFreedom
    variance_x: float | DegreeOfFreedom
    variance_y: float | DegreeOfFreedom
    angle: float | DegreeOfFreedom

    @property
    def num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.x, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.y, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.height, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.variance_x, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.variance_y, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.angle, DegreeOfFreedom):
            num_dofs += 1

        return num_dofs

    def collect_degrees_of_freedom(
        self
    ) -> list[DegreeOfFreedom]:
        dofs = []

        if isinstance(self.x, DegreeOfFreedom):
            dofs.append(copy(self.x))
        if isinstance(self.y, DegreeOfFreedom):
            dofs.append(copy(self.y))
        if isinstance(self.height, DegreeOfFreedom):
            dofs.append(copy(self.height))
        if isinstance(self.variance_x, DegreeOfFreedom):
            dofs.append(copy(self.variance_x))
        if isinstance(self.variance_y, DegreeOfFreedom):
            dofs.append(copy(self.variance_y))
        if isinstance(self.angle, DegreeOfFreedom):
            dofs.append(copy(self.angle))

        return dofs

    # TODO: length check to match num dofs
    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        dof_index = 0

        if isinstance(self.x, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.x.value = dof.value
            else:
                self.x.value = dof

            dof_index += 1
        if isinstance(self.y, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.y.value = dof.value
            else:
                self.y.value = dof

            dof_index += 1
        if isinstance(self.height, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.height.value = dof.value
            else:
                self.height.value = dof

            dof_index += 1
        if isinstance(self.variance_x, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.variance_x.value = dof.value
            else:
                self.variance_x.value = dof

            dof_index += 1
        if isinstance(self.variance_y, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.variance_y.value = dof.value
            else:
                self.variance_y.value = dof

            dof_index += 1
        if isinstance(self.angle, DegreeOfFreedom):
            dof = degrees_of_freedom[dof_index]

            if isinstance(dof, DegreeOfFreedom):
                self.angle.value = dof.value
            else:
                self.angle.value = dof
