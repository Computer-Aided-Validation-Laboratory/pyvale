from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from pyvale.vfm.project_definition import (
    ParameterDefinition,
    ParameterisationSpec,
    PhaseResult,
    TestData,
    resolve_parameter_initial_value_scalar,
)
from pyvale.vfm.spatial_parameterisation import BaseParameterisation, ParameterisationDof


@dataclass(slots=True)
class BasisFunctionKernel:
    """One Gaussian basis function and its optimisation DOFs."""

    x_dof: ParameterisationDof
    y_dof: ParameterisationDof
    height_dof: ParameterisationDof
    variance_x_dof: ParameterisationDof
    variance_y_dof: ParameterisationDof | None = None
    angle_dof: ParameterisationDof | None = None

    def collect_dofs(self) -> list[ParameterisationDof]:
        dofs = [
            self.x_dof,
            self.y_dof,
            self.height_dof,
            self.variance_x_dof,
        ]
        if self.variance_y_dof is not None:
            dofs.append(self.variance_y_dof)
        if self.angle_dof is not None:
            dofs.append(self.angle_dof)
        return dofs


@dataclass(slots=True)
class BasisFunctionParameterisation(BaseParameterisation):
    """Sum of Gaussian basis functions defined directly on the test grid."""

    parameter_name: str
    parameter_definition: ParameterDefinition
    kernels: list[BasisFunctionKernel]
    options: dict[str, Any] = field(default_factory=dict)
    kernel_shape: str = "univariate"
    angle_units: str = "radians"
    normalised: bool = False
    kind: str = "basis_function"

    def prepare(self, test_data: TestData) -> None:
        self._ensure_kernels(test_data)

    def collect_dofs(self) -> list[ParameterisationDof]:
        dofs: list[ParameterisationDof] = []
        for kernel in self.kernels:
            dofs.extend(kernel.collect_dofs())
        return dofs

    def initialise(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """Seed kernel heights from an existing map when one is available."""

        self._ensure_kernels(test_data)

        if source_map is None:
            return

        for kernel in self.kernels:
            kernel.height_dof.value = _sample_map_at_point(
                source_map,
                test_data,
                kernel.x_dof.value,
                kernel.y_dof.value,
            )

    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        self._ensure_kernels(test_data)
        size_y, size_x = test_data.x.shape[0], test_data.x.shape[1]
        parameter_map = np.zeros((size_y, size_x), dtype=np.float64)

        for kernel in self.kernels:
            contribution = _evaluate_kernel(
                x=test_data.x,
                y=test_data.y,
                kernel=kernel,
                angle_units=self.angle_units,
                normalised=self.normalised,
            )
            parameter_map = parameter_map + contribution

        parameter_map[~test_data.specimen_mask] = np.nan
        return parameter_map

    def _ensure_kernels(self, test_data: TestData) -> None:
        if self.kernels:
            return

        self.kernels = _build_default_kernels(
            parameter_name=self.parameter_name,
            parameter_definition=self.parameter_definition,
            kernel_shape=self.kernel_shape,
            initial_count=int(self.options.get("initial_count", 1)),
            active_groups=set(self.options.get("active_groups", [])),
            test_data=test_data,
        )


def build_basis_function_parameterisation(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_spec: ParameterisationSpec,
    previous_result: PhaseResult | None = None,
) -> BaseParameterisation:
    """Build a Gaussian basis-function parameterisation from YAML-style options."""

    options = parameterisation_spec.options
    kernel_shape = str(options.get("kernel_shape", options.get("shape", "univariate")))
    angle_units = str(options.get("angle_units", "radians"))
    normalised = bool(options.get("normalised", False))

    kernels_data = options.get("kernels")
    kernels: list[BasisFunctionKernel] = []
    if kernels_data:
        kernels = [
            _build_kernel(
                parameter_name=parameter_name,
                parameter_definition=parameter_definition,
                kernel_shape=kernel_shape,
                kernel_index=kernel_index,
                kernel_data=kernel_data,
                active_groups=set(options.get("active_groups", [])),
            )
            for kernel_index, kernel_data in enumerate(kernels_data)
        ]

    return BasisFunctionParameterisation(
        parameter_name=parameter_name,
        parameter_definition=parameter_definition,
        kernels=kernels,
        options=options,
        kernel_shape=kernel_shape,
        angle_units=angle_units,
        normalised=normalised,
    )


def _build_kernel(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    kernel_shape: str,
    kernel_index: int,
    kernel_data: dict[str, Any],
    active_groups: set[str],
) -> BasisFunctionKernel:
    kernel_uid = f"{parameter_name}.basis_function.kernel_{kernel_index}"
    is_bivariate = kernel_shape.lower().startswith("bi")
    default_height_value = resolve_parameter_initial_value_scalar(parameter_definition)

    x_dof = _build_dof(
        uid=f"{kernel_uid}.x",
        group="rbf_centres",
        value=kernel_data.get("x"),
        lower_bound=kernel_data.get("x_lower_bound", kernel_data.get("x")),
        upper_bound=kernel_data.get("x_upper_bound", kernel_data.get("x")),
        active_groups=active_groups,
    )
    y_dof = _build_dof(
        uid=f"{kernel_uid}.y",
        group="rbf_centres",
        value=kernel_data.get("y"),
        lower_bound=kernel_data.get("y_lower_bound", kernel_data.get("y")),
        upper_bound=kernel_data.get("y_upper_bound", kernel_data.get("y")),
        active_groups=active_groups,
    )
    height_dof = _build_dof(
        uid=f"{kernel_uid}.height",
        group="rbf_heights",
        value=kernel_data.get("height", default_height_value),
        lower_bound=kernel_data.get(
            "height_lower_bound",
            parameter_definition.lower_bound,
        ),
        upper_bound=kernel_data.get(
            "height_upper_bound",
            parameter_definition.upper_bound,
        ),
        active_groups=active_groups,
    )
    variance_x_dof = _build_dof(
        uid=f"{kernel_uid}.variance_x",
        group="rbf_variances",
        value=kernel_data.get("variance_x", kernel_data.get("variance", 1.0)),
        lower_bound=kernel_data.get(
            "variance_x_lower_bound",
            kernel_data.get("variance_lower_bound", 1.0e-12),
        ),
        upper_bound=kernel_data.get(
            "variance_x_upper_bound",
            kernel_data.get("variance_upper_bound", 1.0e12),
        ),
        active_groups=active_groups,
    )

    variance_y_dof: ParameterisationDof | None = None
    angle_dof: ParameterisationDof | None = None
    if is_bivariate:
        variance_y_dof = _build_dof(
            uid=f"{kernel_uid}.variance_y",
            group="rbf_variances",
            value=kernel_data.get("variance_y", kernel_data.get("variance_x", 1.0)),
            lower_bound=kernel_data.get("variance_y_lower_bound", 1.0e-12),
            upper_bound=kernel_data.get("variance_y_upper_bound", 1.0e12),
            active_groups=active_groups,
        )
        angle_dof = _build_dof(
            uid=f"{kernel_uid}.angle",
            group="rbf_angles",
            value=kernel_data.get("angle", 0.0),
            lower_bound=kernel_data.get("angle_lower_bound", -np.pi),
            upper_bound=kernel_data.get("angle_upper_bound", np.pi),
            active_groups=active_groups,
        )

    return BasisFunctionKernel(
        x_dof=x_dof,
        y_dof=y_dof,
        height_dof=height_dof,
        variance_x_dof=variance_x_dof,
        variance_y_dof=variance_y_dof,
        angle_dof=angle_dof,
    )


def _build_dof(
    uid: str,
    group: str,
    value: float | None,
    lower_bound: float | None,
    upper_bound: float | None,
    active_groups: set[str],
) -> ParameterisationDof:
    if value is None:
        raise ValueError(f"Basis-function DOF '{uid}' requires a value.")
    if lower_bound is None or upper_bound is None:
        raise ValueError(f"Basis-function DOF '{uid}' requires bounds.")

    active = True
    if active_groups:
        active = group in active_groups

    return ParameterisationDof(
        uid=uid,
        group=group,
        value=float(value),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        active=active,
    )


def _build_default_kernels(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    kernel_shape: str,
    initial_count: int,
    active_groups: set[str],
    test_data: TestData,
) -> list[BasisFunctionKernel]:
    initial_count = max(1, initial_count)
    valid_mask = test_data.specimen_mask

    x_values = test_data.x[valid_mask]
    y_values = test_data.y[valid_mask]
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    grid_x = int(np.ceil(np.sqrt(initial_count)))
    grid_y = int(np.ceil(initial_count / grid_x))

    x_centres = np.linspace(x_min, x_max, grid_x)
    y_centres = np.linspace(y_min, y_max, grid_y)
    centre_pairs = [
        (float(x_coord), float(y_coord))
        for y_coord in y_centres
        for x_coord in x_centres
    ][:initial_count]

    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)
    default_variance_x = max((span_x / max(grid_x, 1)) ** 2, 1.0e-6)
    default_variance_y = max((span_y / max(grid_y, 1)) ** 2, 1.0e-6)

    resolved_initial_value = resolve_parameter_initial_value_scalar(parameter_definition)
    if resolved_initial_value is None:
        default_height = 0.0
    else:
        default_height = float(resolved_initial_value) / initial_count

    kernels: list[BasisFunctionKernel] = []
    for kernel_index, (x_coord, y_coord) in enumerate(centre_pairs):
        kernel_data: dict[str, Any] = {
            "x": x_coord,
            "x_lower_bound": x_min,
            "x_upper_bound": x_max,
            "y": y_coord,
            "y_lower_bound": y_min,
            "y_upper_bound": y_max,
            "height": default_height,
            "height_lower_bound": (
                parameter_definition.lower_bound
                if parameter_definition.lower_bound is not None
                else default_height
            ),
            "height_upper_bound": (
                parameter_definition.upper_bound
                if parameter_definition.upper_bound is not None
                else default_height
            ),
            "variance_x": default_variance_x,
            "variance_x_lower_bound": default_variance_x / 10.0,
            "variance_x_upper_bound": default_variance_x * 10.0,
        }
        if kernel_shape.lower().startswith("bi"):
            kernel_data["variance_y"] = default_variance_y
            kernel_data["variance_y_lower_bound"] = default_variance_y / 10.0
            kernel_data["variance_y_upper_bound"] = default_variance_y * 10.0
            kernel_data["angle"] = 0.0

        kernels.append(
            _build_kernel(
                parameter_name=parameter_name,
                parameter_definition=parameter_definition,
                kernel_shape=kernel_shape,
                kernel_index=kernel_index,
                kernel_data=kernel_data,
                active_groups=active_groups,
            )
        )

    return kernels


def _evaluate_kernel(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    kernel: BasisFunctionKernel,
    angle_units: str,
    normalised: bool,
) -> npt.NDArray[np.float64]:
    dx = x - kernel.x_dof.value
    dy = y - kernel.y_dof.value

    variance_x = max(kernel.variance_x_dof.value, 1.0e-12)
    variance_y = variance_x
    angle = 0.0

    if kernel.variance_y_dof is not None:
        variance_y = max(kernel.variance_y_dof.value, 1.0e-12)
    if kernel.angle_dof is not None:
        angle = kernel.angle_dof.value
        if angle_units.lower().startswith("deg"):
            angle = np.deg2rad(angle)

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # Rotate coordinates into the kernel's principal directions.
    local_x = cos_theta * dx + sin_theta * dy
    local_y = -sin_theta * dx + cos_theta * dy

    exponent = -0.5 * (
        (local_x * local_x) / variance_x
        + (local_y * local_y) / variance_y
    )

    coefficient = 1.0
    if normalised:
        determinant = variance_x * variance_y
        coefficient = 1.0 / np.sqrt(determinant * (2.0 * np.pi) ** 2)

    return coefficient * kernel.height_dof.value * np.exp(exponent)


def _sample_map_at_point(
    source_map: npt.NDArray[np.float64],
    test_data: TestData,
    x_coord: float,
    y_coord: float,
) -> float:
    valid_mask = test_data.specimen_mask & np.isfinite(source_map)
    if not np.any(valid_mask):
        return 0.0

    distance = (test_data.x - x_coord) ** 2 + (test_data.y - y_coord) ** 2
    distance = np.where(valid_mask, distance, np.inf)
    flat_index = int(np.argmin(distance))
    return float(np.ravel(source_map)[flat_index])
