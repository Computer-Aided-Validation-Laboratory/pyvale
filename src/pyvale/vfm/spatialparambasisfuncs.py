from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.spatialparam import ISpatialParameterisation


# TODO: maybe scrap this and use duck typing with an ORd type
class IBasisFunctionKernel(ABC):
    @abstractmethod
    def get_num_degrees_of_freedom(self) -> int:
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

    def get_num_degrees_of_freedom(self) -> int:
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

    def get_num_degrees_of_freedom(self) -> int:
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


# Initialising a new basis function:
#   need to take:
#   prev parameter map
# TODO: need a global floor term dof
@dataclass(slots=True)
class BasisFunctionSpatialParameterisation(ISpatialParameterisation):
    floor: float | DegreeOfFreedom | None = None
    kernels: list[IBasisFunctionKernel] = field(default_factory=list)

    def get_num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.floor, DegreeOfFreedom):
            num_dofs += 1

        for kernel in self.kernels:
            num_dofs += kernel.get_num_degrees_of_freedom()

        return num_dofs

    # TODO: create our initial basis functions with fitting
    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        initial_kernel_count = 1

        

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        ...
        # map = np.zeros((size[0], size[1]))

        # for kernel in self.kernels:
            # dx = x - kernel.x_dof.value
            # dy = y - kernel.y_dof.value

            # variance_x = max(kernel.variance_x_dof.value, 1.0e-12)
            # variance_y = variance_x
            # angle = 0.0

            # if kernel.variance_y_dof is not None:
            #     variance_y = max(kernel.variance_y_dof.value, 1.0e-12)
            # if kernel.angle_dof is not None:
            #     angle = kernel.angle_dof.value
            #     if angle_units.lower().startswith("deg"):
            #         angle = np.deg2rad(angle)

            # cos_theta = np.cos(angle)
            # sin_theta = np.sin(angle)

            # # Rotate coordinates into the kernel's principal directions.
            # local_x = cos_theta * dx + sin_theta * dy
            # local_y = -sin_theta * dx + cos_theta * dy

            # exponent = -0.5 * (
            #     (local_x * local_x) / variance_x
            #     + (local_y * local_y) / variance_y
            # )

            # coefficient = 1.0
            # if normalised:
            #     determinant = variance_x * variance_y
            #     coefficient = 1.0 / np.sqrt(determinant * (2.0 * np.pi) ** 2)

            # return coefficient * kernel.height_dof.value * np.exp(exponent)

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        dofs = []

        if isinstance(self.floor, DegreeOfFreedom):
            dofs.append(copy(self.floor))

        for kernel in self.kernels:
            dofs.append(kernel.collect_degrees_of_freedom())

        return dofs

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        index = 0

        if isinstance(self.floor, DegreeOfFreedom):
            dof = degrees_of_freedom[index]

            if isinstance(dof, DegreeOfFreedom):
                self.floor.value = dof.value
            else:
                self.floor.value = dof

            index += 1

        for kernel in self.kernels:
            num_dofs = kernel.get_num_degrees_of_freedom()

            dofs = degrees_of_freedom[index:index + num_dofs]

            kernel.update_from_degrees_of_freedom(dofs)

            index += num_dofs
