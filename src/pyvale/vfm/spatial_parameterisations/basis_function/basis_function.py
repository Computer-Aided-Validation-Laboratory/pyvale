from copy import copy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.spatial_parameterisations.basis_function.kernels import (
    IBasisFunctionKernel,
)
from pyvale.vfm.spatial_parameterisations.degree_of_freedom import (
    DegreeOfFreedom,
)
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    ISpatialParameterisation,
)


# Initialising a new basis function:
#   need to take:
#   prev parameter map
# TODO: need a global floor term dof
@dataclass(slots=True)
class BasisFunctionSpatialParameterisation(ISpatialParameterisation):
    floor: float | DegreeOfFreedom | None = None
    kernels: list[IBasisFunctionKernel] = field(default_factory=list)

    @property
    def num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.floor, DegreeOfFreedom):
            num_dofs += 1

        for kernel in self.kernels:
            num_dofs += kernel.num_degrees_of_freedom

        return num_dofs

    # TODO: create our initial basis functions with fitting
    def update_from_constitutive_parameter(
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
            num_dofs = kernel.num_degrees_of_freedom

            dofs = degrees_of_freedom[index:index + num_dofs]

            kernel.update_from_degrees_of_freedom(dofs)

            index += num_dofs
