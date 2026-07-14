from copy import copy, deepcopy
from dataclasses import dataclass

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


# Lower bound on a kernel's Gaussian feature size, expressed as a number of
# data points. This sets the minimum allowed kernel variance. Exposed here as a
# tunable knob (TODO: tune).
MIN_FEATURE_SIZE_POINTS = 3

# Upper bound on a kernel's Gaussian feature size, expressed as a multiple of
# the domain diagonal. This sets the maximum allowed kernel variance. Exposed
# here as a tunable knob (TODO: tune).
MAX_FEATURE_SIZE_DOMAIN_MULTIPLE = 2.0

CONVERGENCE_THRESHOLD = 0.01
MAX_ADDITIONAL_KERNELS = 10


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

        constitutive_parameter_range = (
            constitutive_parameter.upper_bound
            - constitutive_parameter.lower_bound
        )

        self.kernels.append(
            self._initialise_kernal(
                target_map,
                constitutive_parameter_range
            )
        )

        self._fit_kernels(target_map)

        rmse = _calc_rmse(
            self.to_map(np.array(target_map.shape)),
            target_map
        )

        if rmse < CONVERGENCE_THRESHOLD:
            return

        prev_rmses = []
        for _ in range(MAX_ADDITIONAL_KERNELS):
            self.kernels.append(
                self._initialise_kernal(
                    target_map,
                    constitutive_parameter_range
                )
            )

            self._fit_kernels(target_map)

            rmse = _calc_rmse(
                self.to_map(np.array(target_map.shape)),
                target_map
            )

            prev_rmses.append(rmse)

            if rmse < CONVERGENCE_THRESHOLD:
                return

        print("done")


    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
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

    def should_perform_refinement(self) -> bool:
        return False

    def _fit_kernels(self, target_map: npt.NDArray[np.float64]) -> None:
        dofs = self.collect_degrees_of_freedom()
        normalised_dofs = normalise_degrees_of_freedom(dofs)

        prev_rmses = []
        converged_dofs = np.zeros_like(normalised_dofs)

        def callback(xk: npt.NDArray[np.float64]) -> None:
            if np.array_equal(xk, converged_dofs):
                raise StopIteration()

        try:
            # TODO: maybe move to pattern search, or Powell
            res = minimize(
                lambda x: self._evaluate_dofs(
                    x,
                    target_map,
                    prev_rmses,
                    converged_dofs
                ),
                normalised_dofs,
                method="L-BFGS-B",
                bounds=Bounds(0.0, 1.0),
                callback=callback
            )
            fitted_dofs = res.x
        except StopIteration:
            fitted_dofs = converged_dofs

        lower_bounds = []
        upper_bounds = []

        for dof in self.collect_degrees_of_freedom():
            lower_bounds.append(dof.lower_bound)
            upper_bounds.append(dof.upper_bound)

        updated_dofs = denormalise_degrees_of_freedom(
            fitted_dofs,
            np.array(lower_bounds),
            np.array(upper_bounds)
        )

        self.update_from_degrees_of_freedom(updated_dofs)

    def _evaluate_dofs(
        self,
        normalised_dofs: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64],
        prev_rmses: list[float],
        converged_dofs: npt.NDArray[np.float64],
    ) -> float:
        rmse = self._calc_rmse_from_dofs(normalised_dofs, target_map)

        if prev_rmses:
            reached_convergence_threshold = (
                rmse
                < (CONVERGENCE_THRESHOLD * prev_rmses[-1])
            )

            if reached_convergence_threshold:
                converged_dofs[:] = normalised_dofs

        prev_rmses.append(rmse)

        return rmse

    def _initialise_kernal(
        self,
        target_map: npt.NDArray[np.float64],
        const_param_range: float,
    ) -> BasisFunctionKernel:
            min_x = np.min(self.x)
            max_x = np.max(self.x)

            min_y = np.min(self.y)
            max_y = np.max(self.y)

            range_x = max_x - min_x
            range_y = max_y - min_y

            map = self.to_map(np.array(target_map.shape))
            diff_map = target_map - map

            smoothed_diff_map = uniform_filter(diff_map, size=5)
            abs_smoothed_diff_map = np.abs(smoothed_diff_map)

            peak_idx = np.unravel_index(
                np.argmax(abs_smoothed_diff_map), abs_smoothed_diff_map.shape
            )

            # Place kernel at the location of the peak map value
            centre_x = self.x[peak_idx]
            centre_y = self.y[peak_idx]

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
                float(diff_map[peak_idx]),
                -const_param_range,
                const_param_range
            )

            variance_range = _compute_variance_range(self.x, self.y)

            # seed the variance from the small end of the range so kernels start
            # narrow and grow only as needed
            dof_variance = DegreeOfFreedom(
                float(variance_range[0]),
                variance_range[0],
                variance_range[1]
            )

            return BasisFunctionKernelUnivariate(
                dof_x,
                dof_y,
                dof_height,
                dof_variance
            )

    def _calc_rmse_from_dofs(
        self,
        degrees_of_freedom: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64],
    ) -> float:
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

        return _calc_rmse(map, target_map)


def _calc_rmse(
    map: npt.NDArray[np.float64],
    target_map: npt.NDArray[np.float64]
) -> float:
    # Normalise by the peak magnitude of the target rather than by the
    # per-point target value. A per-point relative error is dominated by the
    # near-zero background between features, which drives the optimiser to park
    # kernels outside the domain (where they contribute nothing) instead of
    # placing them on the features.
    scale = np.max(np.abs(target_map))
    if scale == 0.0:
        scale = 1.0
    return np.sqrt(np.mean(((target_map - map) / scale) ** 2))


def _compute_variance_range(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    feature_threshold = 0.1

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    point_spacing = np.hypot(dx, dy)

    # smallest feature spans at least MIN_FEATURE_SIZE_POINTS data points
    min_feature_size = MIN_FEATURE_SIZE_POINTS * point_spacing

    domain_diagonal = np.hypot(
        np.max(x) - np.min(x),
        np.max(y) - np.min(y)
    )

    # largest feature can span several times the domain, allowing very broad
    # (near-flat) kernels
    max_feature_size = MAX_FEATURE_SIZE_DOMAIN_MULTIPLE * domain_diagonal

    threshold_factor = np.sqrt(-2.0 * np.log(feature_threshold))

    min_sigma = (min_feature_size / 2.0) / threshold_factor
    max_sigma = (max_feature_size / 2.0) / threshold_factor

    return np.array([min_sigma ** 2, max_sigma ** 2])
