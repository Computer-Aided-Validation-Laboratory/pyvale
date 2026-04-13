from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.project_definition import (
    ParameterDefinition,
    ParameterisationSpec,
    PhaseResult,
    TestData,
    resolve_parameter_initial_value,
)
from pyvale.vfm.spatial_parameterisation import BaseParameterisation, ParameterisationDof


@dataclass(slots=True)
class KnownParameterisation(BaseParameterisation):
    """Fixed scalar or fixed map contribution."""

    parameter_name: str
    value: float | npt.NDArray[np.float64]
    kind: str = "known"

    def collect_dofs(self) -> list[ParameterisationDof]:
        return []

    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        value_array = np.asarray(self.value, dtype=np.float64)

        if value_array.ndim == 0:
            parameter_map = np.full(
                (test_data.size_y, test_data.size_x),
                float(value_array),
                dtype=np.float64,
            )
        else:
            if value_array.shape != (test_data.size_y, test_data.size_x):
                raise ValueError(
                    f"Known parameter '{self.parameter_name}' has shape "
                    f"{value_array.shape}, expected "
                    f"{(test_data.size_y, test_data.size_x)}."
                )
            parameter_map = value_array.copy()

        parameter_map[~test_data.specimen_mask] = np.nan
        return parameter_map


@dataclass(slots=True)
class HomogeneousParameterisation(BaseParameterisation):
    """Single scalar DOF applied everywhere inside the specimen mask."""

    parameter_name: str
    dof: ParameterisationDof
    initialise_from: str = "initial_value"
    kind: str = "homogeneous"

    def collect_dofs(self) -> list[ParameterisationDof]:
        return [self.dof]

    def initialise(
        self,
        test_data: TestData,
        source_map: npt.NDArray[np.float64] | None = None,
    ) -> None:
        if source_map is None:
            return
        if self.initialise_from == "initial_value":
            return
        self.dof.value = float(np.nanmean(source_map[test_data.specimen_mask]))

    def to_map(self, test_data: TestData) -> npt.NDArray[np.float64]:
        parameter_map = np.full(
            (test_data.size_y, test_data.size_x),
            self.dof.value,
            dtype=np.float64,
        )
        parameter_map[~test_data.specimen_mask] = np.nan
        return parameter_map


def build_homogeneous_parameterisation(
    parameter_name: str,
    parameter_definition: ParameterDefinition,
    parameterisation_spec: ParameterisationSpec,
    previous_result: PhaseResult | None = None,
) -> BaseParameterisation:
    options = parameterisation_spec.options
    initial_value = options.get("value")
    if initial_value is None:
        initial_value = resolve_parameter_initial_value(parameter_definition)

    if initial_value is None:
        raise ValueError(
            f"Parameter '{parameter_name}' needs an initial value for a "
            f"'{parameterisation_spec.kind}' parameterisation."
        )

    if parameterisation_spec.kind == "known":
        return KnownParameterisation(
            parameter_name=parameter_name,
            value=initial_value,
        )

    initial_scalar = (
        float(np.nanmean(initial_value))
        if isinstance(initial_value, np.ndarray)
        else float(initial_value)
    )
    if initial_scalar is None:
        raise ValueError(
            f"Parameter '{parameter_name}' needs a scalar initial value for "
            "a homogeneous parameterisation."
        )

    lower_bound = options.get("lower_bound", parameter_definition.lower_bound)
    upper_bound = options.get("upper_bound", parameter_definition.upper_bound)
    if lower_bound is None or upper_bound is None:
        raise ValueError(
            f"Parameter '{parameter_name}' needs lower and upper bounds for "
            "a homogeneous parameterisation."
        )

    return HomogeneousParameterisation(
        parameter_name=parameter_name,
        dof=ParameterisationDof(
            uid=f"{parameter_name}.homogeneous.value",
            group="value",
            value=float(initial_scalar),
            lower_bound=float(lower_bound),
            upper_bound=float(upper_bound),
            active=True,
        ),
        initialise_from=str(options.get("initialise_from", "initial_value")),
    )
