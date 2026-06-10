from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter
from scipy.optimize import Bounds, minimize

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.normalisation import (
    denormalise_degrees_of_freedom,
    normalise_degrees_of_freedom,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation


@dataclass(slots=True)
class BasisFunctionKernelUnivariate:
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
class BasisFunctionKernelBivariate:
    x: float | DegreeOfFreedom
    y: float | DegreeOfFreedom
    height: float | DegreeOfFreedom
    variance_x: float | DegreeOfFreedom
    variance_y: float | DegreeOfFreedom
    # radians
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


BasisFunctionKernel = (
    BasisFunctionKernelUnivariate
    | BasisFunctionKernelBivariate
)


@dataclass(slots=True)
class SpatialParameterisationBasisFunction(ISpatialParameterisation):
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    kernels: list[BasisFunctionKernel]

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ) -> None:
        self.x = x
        self.y = y
        self.kernels = []

    def get_num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        for kernel in self.kernels:
            num_dofs += kernel.get_num_degrees_of_freedom()

        return num_dofs

    # TODO: in future, we might want to randomise poisitions of all rbfs
    #   when a new rbf is added
    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        target_map = constitutive_parameter.map
        map_size = np.array(target_map.shape, dtype=np.uint32)

        # place initial basis function kernel in the center of the map
        min_x = np.min(self.x)
        max_x = np.max(self.x)
        min_y = np.min(self.y)
        max_y = np.max(self.y)

        range_x = max_x - min_x
        range_y = max_y - min_y

        centre_x = min_x + (range_x / 2.0)
        centre_y = min_y + (range_y / 2.0)

        dof_x = DegreeOfFreedom(
            centre_x,
            min_x - (0.5 * range_x),
            max_x + (0.5 * range_x)
        )
        dof_y = DegreeOfFreedom(
            centre_y,
            min_y - (0.5 * range_y),
            max_y + (0.5 * range_y)
        )

        constitutive_parameter_range = (
            constitutive_parameter.upper_bound
            - constitutive_parameter.lower_bound
        )

        dof_height = DegreeOfFreedom(
            float(np.mean(np.abs(target_map))),
            -constitutive_parameter_range,
            constitutive_parameter_range
        )

        # TODO: figure out how to set initial value and range
        dof_variance = DegreeOfFreedom(
            10.0,
            0.0,
            100.0
        )

        self.kernels.append(
            BasisFunctionKernelUnivariate(
                dof_x,
                dof_y,
                dof_height,
                dof_variance
            )
        )

        prev_rmspe = 1

        for _ in range(10):
            # perform fitting on the existing kernels
            updated_dofs = self.collect_degrees_of_freedom()
            normalised_dofs = normalise_degrees_of_freedom(updated_dofs)

            # TODO: maybe move to pattern search, or Powell
            # TODO: keep track of the rmspe every iteration of the for loop
            # TODO: keep track of the rmspe every iteration of the optimisation
            #   loop, and break when converged enough
            res = minimize(
                lambda x: self._calc_rmspe_from_dofs(x, target_map),
                normalised_dofs,
                method="L-BFGS-B",
                bounds=Bounds(0.0, 1.0)
            )

            optimised_dofs = res.x

            rmspe = self._calc_rmspe_from_dofs(optimised_dofs, target_map)

            # TODO: what is the right constant for this break condition?
            if (prev_rmspe - rmspe) < 0.005:
                break

            prev_rmspe = rmspe

            lower_bounds = []
            upper_bounds = []

            for dof in self.collect_degrees_of_freedom():
                lower_bounds.append(dof.lower_bound)
                upper_bounds.append(dof.upper_bound)

            updated_dofs = denormalise_degrees_of_freedom(
                optimised_dofs,
                np.array(lower_bounds),
                np.array(upper_bounds)
            )

            self.update_from_degrees_of_freedom(updated_dofs)

            map = self.to_map(np.array(target_map.shape))

            error_map = target_map - map

            smoothed_error_map = uniform_filter(error_map, size=4)

            abs_smoothed_error = np.abs(smoothed_error_map)
            max_idx = np.unravel_index(
                np.argmax(abs_smoothed_error), abs_smoothed_error.shape
            )

            # place a new univariate basis function at the centre
            # of the highest error
            centre_x = self.x[max_idx]
            centre_y = self.y[max_idx]

            dof_x = DegreeOfFreedom(
                centre_x,
                min_x - (0.5 * range_x),
                max_x + (0.5 * range_x)
            )
            dof_y = DegreeOfFreedom(
                centre_y,
                min_y - (0.5 * range_y),
                max_y + (0.5 * range_y)
            )

            dof_height = DegreeOfFreedom(
                float(error_map[max_idx]),
                -constitutive_parameter_range,
                constitutive_parameter_range
            )

            dof_variance = DegreeOfFreedom(
                10.0,
                0.0,
                100.0
            )

            self.kernels.append(
                BasisFunctionKernelUnivariate(
                    dof_x,
                    dof_y,
                    dof_height,
                    dof_variance
                )
            )


    def _calc_rmspe(
        self,
        map: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64]
    ):
        return np.sqrt(np.mean(((target_map - map) / target_map ) ** 2))

    def _calc_rmspe_from_dofs(
        self,
        degrees_of_freedom: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64]
    ):
        lower_bounds = []
        upper_bounds = []

        for dof in self.collect_degrees_of_freedom():
            lower_bounds.append(dof.lower_bound)
            upper_bounds.append(dof.upper_bound)

        dofs = denormalise_degrees_of_freedom(
            degrees_of_freedom,
            np.array(lower_bounds),
            np.array(upper_bounds)
        )

        updated_parameterisation = deepcopy(self)
        updated_parameterisation.update_from_degrees_of_freedom(dofs)

        map = updated_parameterisation.to_map(
            np.array(target_map.shape)
        )

        return self._calc_rmspe(map, target_map)

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        if not self.kernels:
            raise RuntimeError(
                "self.kernels is empty, initialise_from_constitutive_parameter"
                "must be called before to_map"
            )

        parameter_map = np.zeros((size[0], size[1]), dtype=np.float64)

        for kernel in self.kernels:
            x = (
                kernel.x.value
                if isinstance(kernel.x, DegreeOfFreedom)
                else kernel.x
            )
            y = (
                kernel.y.value
                if isinstance(kernel.y, DegreeOfFreedom)
                else kernel.y
            )
            height = (
                kernel.height.value
                if isinstance(kernel.height, DegreeOfFreedom)
                else kernel.height
            )

            dx = self.x - x
            dy = self.y - y

            match kernel:
                case BasisFunctionKernelUnivariate(variance=v):
                    variance = (
                        v.value
                        if isinstance(v, DegreeOfFreedom)
                        else v
                    )

                    exponent = -0.5 * (
                        (dx ** 2) / variance
                        + (dy ** 2) / variance
                    )
                case BasisFunctionKernelBivariate(
                    variance_x=v_x,
                    variance_y=v_y,
                    angle=a
                ):
                    variance_x = (
                        v_x.value
                        if isinstance(v_x, DegreeOfFreedom)
                        else v_x
                    )
                    variance_y = (
                        v_y.value
                        if isinstance(v_y, DegreeOfFreedom)
                        else v_y
                    )
                    angle = (
                        a.value
                        if isinstance(a, DegreeOfFreedom)
                        else a
                    )

                    # rotate coordinates into the kernel's principal directions
                    cos_theta = np.cos(angle)
                    sin_theta = np.sin(angle)

                    local_x = cos_theta * dx + sin_theta * dy
                    local_y = -sin_theta * dx + cos_theta * dy

                    exponent = -0.5 * (
                        (local_x ** 2 ) / variance_x
                        + (local_y ** 2) / variance_y
                    )

            parameter_map += height * np.exp(exponent)

        return parameter_map

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        dofs = []

        for kernel in self.kernels:
            dofs.extend(kernel.collect_degrees_of_freedom())

        return dofs

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        if len(degrees_of_freedom) != self.get_num_degrees_of_freedom():
            raise ValueError(
                f"expected {self.get_num_degrees_of_freedom()} degrees of "
                f"freedom, got {len(degrees_of_freedom)}"
            )

        index = 0

        for kernel in self.kernels:
            num_dofs = kernel.get_num_degrees_of_freedom()

            dofs = degrees_of_freedom[index:index + num_dofs]

            kernel.update_from_degrees_of_freedom(dofs)

            index += num_dofs
