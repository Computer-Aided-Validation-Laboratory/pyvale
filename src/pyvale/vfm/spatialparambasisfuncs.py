from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.ndimage import uniform_filter
from scipy.optimize import Bounds, minimize

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.dof import DegreeOfFreedom
from pyvale.vfm.experimentdata import ExperimentData
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
MAX_FEATURE_SIZE_DOMAIN_MULTIPLE = 1.0

CONVERGENCE_THRESHOLD = 0.01
MAX_ADDITIONAL_KERNELS = 10


def _resolve_dof_value(value: float | DegreeOfFreedom) -> float:
    return float(value.value if isinstance(value, DegreeOfFreedom) else value)


@dataclass(slots=True)
class BasisFunctionKernelUnivariate:
    x: float | DegreeOfFreedom
    y: float | DegreeOfFreedom
    variance: float | DegreeOfFreedom

    def get_num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.x, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.y, DegreeOfFreedom):
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
    variance_x: float | DegreeOfFreedom
    variance_y: float | DegreeOfFreedom
    # radians
    angle: float | DegreeOfFreedom

    def canonical_values(self) -> tuple[float, float, float]:
        """Return an axis-ordered, pi-periodic representation of the ellipse."""
        variance_x = _resolve_dof_value(self.variance_x)
        variance_y = _resolve_dof_value(self.variance_y)
        angle = _resolve_dof_value(self.angle)
        if variance_y > variance_x:
            variance_x, variance_y = variance_y, variance_x
            angle += 0.5 * np.pi
        angle = (angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi
        return variance_x, variance_y, angle

    def get_num_degrees_of_freedom(self) -> int:
        num_dofs = 0

        if isinstance(self.x, DegreeOfFreedom):
            num_dofs += 1
        if isinstance(self.y, DegreeOfFreedom):
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
class SupportBasis:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    kernels: list[BasisFunctionKernel] | None = None

    def __post_init__(self) -> None:
        if self.kernels is None:
            self.kernels = []

    def prepare(
        self,
        experiment_data: ExperimentData,
    ) -> None:
        return

    def get_num_degrees_of_freedom(self) -> int:
        assert self.kernels is not None
        return sum(
            kernel.get_num_degrees_of_freedom()
            for kernel in self.kernels
        )

    def collect_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        assert self.kernels is not None
        degrees_of_freedom: list[DegreeOfFreedom] = []
        for kernel in self.kernels:
            degrees_of_freedom.extend(kernel.collect_degrees_of_freedom())
        return degrees_of_freedom

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
    ) -> None:
        assert self.kernels is not None
        index = 0
        for kernel in self.kernels:
            num_dofs = kernel.get_num_degrees_of_freedom()
            if num_dofs == 0:
                continue
            kernel.update_from_degrees_of_freedom(
                degrees_of_freedom[index:index + num_dofs]
            )
            index += num_dofs


@dataclass(slots=True)
class SpatialParameterisationBasisFunction(ISpatialParameterisation):
    """
    Smoothly varying parameterisation built from a weighted sum of basis
    functions over the specimen grid

    Attributes
    ----------
    support : SupportBasis
        The support structure for the basis functions.
    heights : list[float | DegreeOfFreedom | None]
        The heights of the basis functions.
    x : npt.NDArray[np.float64]
        The x-coordinates of the specimen grid.
    y : npt.NDArray[np.float64]
        The y-coordinates of the specimen grid.
    kernels : list[BasisFunctionKernel]
        The basis function kernels.
    initial_kernels_max : int
        The maximum number of initial kernels to add during phase initialisation.
    initial_height_fraction : float
        The fraction of the initial height.
    """

    support: SupportBasis
    heights: list[float | DegreeOfFreedom | None]
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    kernels: list[BasisFunctionKernel]
    initial_kernels_max: int
    initial_height_fraction: float
    kernel_type: Literal["univariate", "bivariate"]
    centre_bounds_span_factor: float

    def __init__(
        self,
        x: npt.NDArray[np.float64] | None = None,
        y: npt.NDArray[np.float64] | None = None,
        support: SupportBasis | None = None,
        heights: list[float | DegreeOfFreedom | None] | None = None,
        kernels: list[BasisFunctionKernel] | None = None,
        *,
        initial_kernels_max: int = 1,
        initial_height_fraction: float = 0.01,
        kernel_type: Literal["univariate", "bivariate"] = "univariate",
        centre_bounds_span_factor: float = 1.0,
    ) -> None:
        if initial_kernels_max < 1:
            raise ValueError("initial_kernels_max must be at least one.")
        if not 0.0 < initial_height_fraction <= 1.0:
            raise ValueError(
                "initial_height_fraction must lie in (0, 1]."
            )
        if kernel_type not in {"univariate", "bivariate"}:
            raise ValueError(
                "kernel_type must be 'univariate' or 'bivariate'."
            )
        if centre_bounds_span_factor < 1.0:
            raise ValueError("centre_bounds_span_factor must be at least 1.0.")

        if support is None:
            if x is None or y is None:
                raise ValueError("Provide either support or both x and y.")
            self.support = SupportBasis(x=x, y=y, kernels=kernels)
        else:
            if x is not None or y is not None or kernels is not None:
                raise ValueError(
                    "Provide either support or x/y with kernels, not both."
                )
            self.support = support

        self.x = self.support.x
        self.y = self.support.y
        assert self.support.kernels is not None
        self.kernels = self.support.kernels
        self.heights = [] if heights is None else heights

        self.initial_kernels_max = initial_kernels_max
        self.initial_height_fraction = initial_height_fraction
        self.kernel_type = kernel_type
        self.centre_bounds_span_factor = centre_bounds_span_factor

        self._ensure_heights_match_support()

    def create_kernel(
        self,
        x: float | DegreeOfFreedom,
        y: float | DegreeOfFreedom,
        variance: float | DegreeOfFreedom,
    ) -> BasisFunctionKernel:
        """Create a Gaussian kernel matching this parameterisation's type."""

        if self.kernel_type == "bivariate":
            return BasisFunctionKernelBivariate(
                x=x,
                y=y,
                variance_x=variance,
                variance_y=copy(variance),
                angle=DegreeOfFreedom(
                    0.0,
                    -0.5 * np.pi,
                    0.5 * np.pi,
                ),
            )
        return BasisFunctionKernelUnivariate(x=x, y=y, variance=variance)

    def get_centre_bounds(self) -> tuple[float, float, float, float]:
        """Return configured ``x_min, x_max, y_min, y_max`` centre bounds."""
        data_min_x = float(np.nanmin(self.x))
        data_max_x = float(np.nanmax(self.x))
        data_min_y = float(np.nanmin(self.y))
        data_max_y = float(np.nanmax(self.y))
        x_padding = 0.5 * (self.centre_bounds_span_factor - 1.0) * (
            data_max_x - data_min_x
        )
        y_padding = 0.5 * (self.centre_bounds_span_factor - 1.0) * (
            data_max_y - data_min_y
        )
        return (
            data_min_x - x_padding,
            data_max_x + x_padding,
            data_min_y - y_padding,
            data_max_y + y_padding,
        )

    def get_num_degrees_of_freedom(self) -> int:
        return sum(
            isinstance(height, DegreeOfFreedom) or height is None
            for height in self.heights
        )

    # TODO: in future, we might want to randomise poisitions of all rbfs
    #   when a new rbf is added
    def initialise_from_constitutive_parameter(
        self,
        constitutive_parameter: ConstitutiveParameter
    ) -> None:
        """
        Initialise the basis function parameterisation from a
        constitutive parameter's map, fitting the heights of the
        basis functions to match the target map. If no kernels are
        present, new kernels will be added.
        """
        target_map = constitutive_parameter.map

        constitutive_parameter_range = (
            constitutive_parameter.upper_bound
            - constitutive_parameter.lower_bound
        )

        # Structural initialisation is phase-level policy. With no kernels there
        # are no basis DOFs to initialise.
        if len(self.kernels) == 0:
            return

        # Explicit DOFs are caller-supplied seeds or previously identified
        # values. Preserve them across phase preparation; only unresolved
        # ``None`` height slots need fitting from the incoming map.
        self._ensure_heights_match_support()
        if all(isinstance(height, DegreeOfFreedom) for height in self.heights):
            return

        self._initialise_heights_from_support(
            constitutive_parameter,
            target_map,
        )

    def fit_to_parameter_map(
        self,
        constitutive_parameter: ConstitutiveParameter,
        *,
        max_basis_functions: int | None = None,
        minimum_relative_improvement: float | None = None,
    ) -> None:
        """Add and fit Gaussian bases to a supplied parameter map.

        This is the explicit counterpart to automatic initialisation, for
        phase-level policies that choose when map fitting is appropriate.
        """

        self.fit_to_map(
            constitutive_parameter.map,
            parameter_range=(
                constitutive_parameter.upper_bound
                - constitutive_parameter.lower_bound
            ),
            max_basis_functions=max_basis_functions,
            minimum_relative_improvement=minimum_relative_improvement,
        )

    def fit_to_map(
        self,
        target_map: npt.NDArray[np.float64],
        *,
        parameter_range: float,
        max_basis_functions: int | None = None,
        minimum_relative_improvement: float | None = None,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        """Add and fit Gaussian bases to a map with the given value range."""

        if self.kernels:
            raise ValueError("Parameter-map fitting requires an empty basis support.")
        if parameter_range <= 0.0:
            raise ValueError("parameter_range must be positive.")
        if max_basis_functions is not None and max_basis_functions < 1:
            raise ValueError("max_basis_functions must be at least one.")
        if (
            minimum_relative_improvement is not None
            and minimum_relative_improvement < 0.0
        ):
            raise ValueError("minimum_relative_improvement must be non-negative.")

        self._initialise_support_from_map(
            target_map,
            parameter_range,
            max_basis_functions=max_basis_functions,
            minimum_relative_improvement=minimum_relative_improvement,
            fit_mask=fit_mask,
        )

    def to_map(
        self,
        size: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.float64]:
        parameter_map = np.zeros((size[0], size[1]), dtype=np.float64)

        self._ensure_heights_match_support()
        for kernel, height in zip(self.kernels, self.heights, strict=True):
            parameter_map += (
                _resolve_height_value(height)
                * self._evaluate_kernel_response(kernel)
            )

        return parameter_map

    def collect_degrees_of_freedom(
        self,
    ) -> list[DegreeOfFreedom]:
        self._ensure_heights_match_support()
        degrees_of_freedom: list[DegreeOfFreedom] = []
        for height in self.heights:
            if isinstance(height, DegreeOfFreedom):
                degrees_of_freedom.append(copy(height))
            elif height is None:
                raise ValueError(
                    "SpatialParameterisationBasisFunction must be initialised "
                    "before collecting height DOFs."
                )
        return degrees_of_freedom

    def update_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64]
    ) -> None:
        self._ensure_heights_match_support()
        if len(degrees_of_freedom) != self.get_num_degrees_of_freedom():
            raise ValueError(
                f"expected {self.get_num_degrees_of_freedom()} degrees of "
                f"freedom, got {len(degrees_of_freedom)}"
            )

        dof_index = 0
        for height_index, height in enumerate(self.heights):
            if not isinstance(height, DegreeOfFreedom):
                continue

            updated_value = degrees_of_freedom[dof_index]
            if isinstance(updated_value, DegreeOfFreedom):
                self.heights[height_index] = updated_value
            else:
                height.value = float(updated_value)
            dof_index += 1

    def _ensure_heights_match_support(self) -> None:
        if len(self.kernels) == 0:
            if len(self.heights) != 0:
                raise ValueError(
                    "SupportBasis has no kernels, so heights must be empty."
                )
            return

        if len(self.heights) == 0:
            self.heights = [None] * len(self.kernels)
            return

        if len(self.heights) != len(self.kernels):
            raise ValueError(
                f"Expected {len(self.kernels)} basis heights, got "
                f"{len(self.heights)}."
            )

    def _evaluate_kernel_response(
        self,
        kernel: BasisFunctionKernel,
    ) -> npt.NDArray[np.float64]:
        x_coord = (
            kernel.x.value
            if isinstance(kernel.x, DegreeOfFreedom)
            else kernel.x
        )
        y_coord = (
            kernel.y.value
            if isinstance(kernel.y, DegreeOfFreedom)
            else kernel.y
        )

        dx = self.x - x_coord
        dy = self.y - y_coord

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
                angle=a,
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

                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)
                local_x = cos_theta * dx + sin_theta * dy
                local_y = -sin_theta * dx + cos_theta * dy

                exponent = -0.5 * (
                    (local_x ** 2) / variance_x
                    + (local_y ** 2) / variance_y
                )

        return np.exp(exponent)

    def _collect_internal_degrees_of_freedom(
        self,
        include_support_degrees_of_freedom: bool,
    ) -> list[DegreeOfFreedom]:
        degrees_of_freedom: list[DegreeOfFreedom] = []
        if include_support_degrees_of_freedom:
            degrees_of_freedom.extend(self.support.collect_degrees_of_freedom())
        degrees_of_freedom.extend(self.collect_degrees_of_freedom())
        return degrees_of_freedom

    def _update_internal_from_degrees_of_freedom(
        self,
        degrees_of_freedom: list[DegreeOfFreedom] | npt.NDArray[np.float64],
        include_support_degrees_of_freedom: bool,
    ) -> None:
        index = 0
        if include_support_degrees_of_freedom:
            num_support_dofs = self.support.get_num_degrees_of_freedom()
            self.support.update_from_degrees_of_freedom(
                degrees_of_freedom[:num_support_dofs]
            )
            index += num_support_dofs

        num_parameterisation_dofs = self.get_num_degrees_of_freedom()
        self.update_from_degrees_of_freedom(
            degrees_of_freedom[index:index + num_parameterisation_dofs]
        )

    def _fit_internal_dofs(
        self,
        target_map: npt.NDArray[np.float64],
        include_support_degrees_of_freedom: bool,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        degrees_of_freedom = self._collect_internal_degrees_of_freedom(
            include_support_degrees_of_freedom
        )
        if not degrees_of_freedom:
            return

        normalised_dofs = normalise_degrees_of_freedom(degrees_of_freedom)

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
                    converged_dofs,
                    include_support_degrees_of_freedom,
                    fit_mask,
                ),
                normalised_dofs,
                method="L-BFGS-B",
                bounds=Bounds(0.0, 1.0),
                callback=callback
            )
            fitted_dofs = res.x
        except StopIteration:
            fitted_dofs = converged_dofs

        updated_dofs = denormalise_degrees_of_freedom(
            fitted_dofs,
            np.asarray(
                [dof.lower_bound for dof in degrees_of_freedom],
                dtype=np.float64,
            ),
            np.asarray(
                [dof.upper_bound for dof in degrees_of_freedom],
                dtype=np.float64,
            ),
        )

        self._update_internal_from_degrees_of_freedom(
            updated_dofs,
            include_support_degrees_of_freedom,
        )

    def _initialise_support_from_map(
        self,
        target_map: npt.NDArray[np.float64],
        const_param_range: float,
        *,
        max_basis_functions: int | None = None,
        minimum_relative_improvement: float | None = None,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> None:
        kernel, height = self._initialise_kernel(
            target_map,
            const_param_range,
            fit_mask=fit_mask,
        )
        self.kernels.append(kernel)
        self.heights.append(height)
        if fit_mask is None:
            self._fit_internal_dofs(
                target_map,
                include_support_degrees_of_freedom=True,
            )
        else:
            self._fit_internal_dofs(
                target_map,
                include_support_degrees_of_freedom=True,
                fit_mask=fit_mask,
            )

        rmse = _calc_rmse(
            self.to_map(np.array(target_map.shape)),
            target_map,
            fit_mask,
        )
        if rmse < CONVERGENCE_THRESHOLD:
            return

        maximum = (
            MAX_ADDITIONAL_KERNELS + 1
            if max_basis_functions is None
            else max_basis_functions
        )
        for _ in range(maximum - 1):
            previous_rmse = rmse
            accepted_kernels = deepcopy(self.kernels)
            accepted_heights = deepcopy(self.heights)
            kernel, height = self._initialise_kernel(
                target_map,
                const_param_range,
                fit_mask=fit_mask,
            )
            self.kernels.append(kernel)
            self.heights.append(height)
            if fit_mask is None:
                self._fit_internal_dofs(
                    target_map,
                    include_support_degrees_of_freedom=True,
                )
            else:
                self._fit_internal_dofs(
                    target_map,
                    include_support_degrees_of_freedom=True,
                    fit_mask=fit_mask,
                )

            rmse = _calc_rmse(
                self.to_map(np.array(target_map.shape)),
                target_map,
                fit_mask,
            )
            if (
                minimum_relative_improvement is not None
                and (previous_rmse - rmse) / max(previous_rmse, 1.0e-12)
                < minimum_relative_improvement
            ):
                # Fitting a candidate updates every existing kernel and height.
                # Restore the complete accepted model, not only the new basis.
                self.kernels[:] = accepted_kernels
                self.heights[:] = accepted_heights
                return
            if rmse < CONVERGENCE_THRESHOLD:
                return

    def _initialise_heights_from_support(
        self,
        constitutive_parameter: ConstitutiveParameter,
        target_map: npt.NDArray[np.float64],
    ) -> None:
        self._ensure_heights_match_support()
        if len(self.kernels) == 0:
            return

        finite_mask = np.isfinite(target_map)
        target_vector = target_map[finite_mask]

        fixed_contribution = np.zeros_like(target_vector, dtype=np.float64)
        free_kernel_indices: list[int] = []
        free_kernel_responses: list[npt.NDArray[np.float64]] = []

        for kernel_index, (kernel, height) in enumerate(
            zip(self.kernels, self.heights, strict=True)
        ):
            kernel_response = self._evaluate_kernel_response(kernel)[finite_mask]
            if isinstance(height, DegreeOfFreedom) or height is None:
                free_kernel_indices.append(kernel_index)
                free_kernel_responses.append(kernel_response)
                continue

            fixed_contribution += float(height) * kernel_response

        if not free_kernel_indices:
            return

        response_matrix = np.column_stack(free_kernel_responses)
        solved_heights, _, _, _ = np.linalg.lstsq(
            response_matrix,
            target_vector - fixed_contribution,
            rcond=None,
        )

        for solved_height, kernel_index in zip(
            solved_heights,
            free_kernel_indices,
            strict=True,
        ):
            existing_height = self.heights[kernel_index]
            if isinstance(existing_height, DegreeOfFreedom):
                lower_bound = existing_height.lower_bound
                upper_bound = existing_height.upper_bound
            else:
                lower_bound = constitutive_parameter.lower_bound
                upper_bound = constitutive_parameter.upper_bound

            clipped_height = float(
                np.clip(solved_height, lower_bound, upper_bound)
            )
            self.heights[kernel_index] = DegreeOfFreedom(
                clipped_height,
                lower_bound,
                upper_bound,
            )

    def _evaluate_dofs(
        self,
        normalised_dofs: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64],
        prev_rmses: list[float],
        converged_dofs: npt.NDArray[np.float64],
        include_support_degrees_of_freedom: bool,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> float:
        rmse = self._calc_rmse_from_dofs(
            normalised_dofs,
            target_map,
            include_support_degrees_of_freedom,
            fit_mask,
        )

        if prev_rmses:
            reached_convergence_threshold = (
                rmse
                < (CONVERGENCE_THRESHOLD * prev_rmses[-1])
            )

            if reached_convergence_threshold:
                converged_dofs[:] = normalised_dofs

        prev_rmses.append(rmse)

        return rmse

    def _initialise_kernel(
        self,
        target_map: npt.NDArray[np.float64],
        const_param_range: float,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[BasisFunctionKernel, DegreeOfFreedom]:
        min_x, max_x, min_y, max_y = self.get_centre_bounds()

        parameter_map = self.to_map(np.array(target_map.shape))
        diff_map = target_map - parameter_map
        if fit_mask is not None:
            diff_map = np.where(fit_mask, diff_map, 0.0)

        smoothed_diff_map = uniform_filter(diff_map, size=5)
        abs_smoothed_diff_map = np.abs(smoothed_diff_map)

        peak_idx = np.unravel_index(
            np.argmax(abs_smoothed_diff_map),
            abs_smoothed_diff_map.shape,
        )

        # Place the kernel at the dominant unresolved feature.
        centre_x = self.x[peak_idx]
        centre_y = self.y[peak_idx]

        dof_x = DegreeOfFreedom(
            centre_x,
            min_x,
            max_x,
        )

        dof_y = DegreeOfFreedom(
            centre_y,
            min_y,
            max_y,
        )

        dof_height = DegreeOfFreedom(
            float(diff_map[peak_idx]),
            -const_param_range,
            const_param_range,
        )

        variance_range = _compute_variance_range(self.x, self.y)

        # Seed the variance from the small end of the range so kernels start
        # narrow and grow only as needed.
        dof_variance = DegreeOfFreedom(
            float(variance_range[0]),
            variance_range[0],
            variance_range[1],
            scaling="log",
        )

        return self.create_kernel(dof_x, dof_y, dof_variance), dof_height

    def _calc_rmse_from_dofs(
        self,
        degrees_of_freedom: npt.NDArray[np.float64],
        target_map: npt.NDArray[np.float64],
        include_support_degrees_of_freedom: bool,
        fit_mask: npt.NDArray[np.bool_] | None = None,
    ) -> float:
        internal_dofs = self._collect_internal_degrees_of_freedom(
            include_support_degrees_of_freedom
        )

        dofs = denormalise_degrees_of_freedom(
            degrees_of_freedom,
            np.asarray(
                [dof.lower_bound for dof in internal_dofs],
                dtype=np.float64,
            ),
            np.asarray(
                [dof.upper_bound for dof in internal_dofs],
                dtype=np.float64,
            ),
        )

        updated_parameterisation = deepcopy(self)
        updated_parameterisation._update_internal_from_degrees_of_freedom(
            dofs,
            include_support_degrees_of_freedom,
        )

        map = updated_parameterisation.to_map(
            np.array(target_map.shape)
        )

        return _calc_rmse(map, target_map, fit_mask)


def _resolve_height_value(
    height: float | DegreeOfFreedom | None,
) -> float:
    if isinstance(height, DegreeOfFreedom):
        return float(height.value)
    if height is None:
        raise ValueError(
            "SpatialParameterisationBasisFunction must be initialised before "
            "converting to a map."
        )
    return float(height)


def _calc_rmse(
    map: npt.NDArray[np.float64],
    target_map: npt.NDArray[np.float64],
    fit_mask: npt.NDArray[np.bool_] | None = None,
) -> float:
    # Normalise by the peak magnitude of the target rather than by the
    # per-point target value. A per-point relative error is dominated by the
    # near-zero background between features, which drives the optimiser to park
    # kernels outside the domain (where they contribute nothing) instead of
    # placing them on the features.
    if fit_mask is None:
        fit_mask = np.ones(target_map.shape, dtype=bool)
    if fit_mask.shape != target_map.shape:
        raise ValueError("fit_mask must have the same shape as target_map")
    scale = np.max(np.abs(target_map[fit_mask]))
    if scale == 0.0:
        scale = 1.0
    return np.sqrt(np.mean(((target_map[fit_mask] - map[fit_mask]) / scale) ** 2))


def _compute_variance_range(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    feature_threshold = 0.1

    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    point_spacing = min(abs(dx), abs(dy))

    # smallest feature spans at least MIN_FEATURE_SIZE_POINTS data points
    min_feature_size = MIN_FEATURE_SIZE_POINTS * point_spacing

    domain_diagonal = np.hypot(
        np.max(x) - np.min(x),
        np.max(y) - np.min(y)
    )

    max_feature_size = max(
        MAX_FEATURE_SIZE_DOMAIN_MULTIPLE * domain_diagonal,
        min_feature_size * (1.0 + 1.0e-6),
    )

    threshold_factor = np.sqrt(-2.0 * np.log(feature_threshold))

    min_sigma = (min_feature_size / 2.0) / threshold_factor
    max_sigma = (max_feature_size / 2.0) / threshold_factor

    return np.array([min_sigma ** 2, max_sigma ** 2])
