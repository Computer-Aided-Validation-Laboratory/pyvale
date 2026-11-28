# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Feasibility limits on simulated sensor performance."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.sensoropt.designspace import IConstrainOut
from pyvale.sensoropt.objective import (
    EErrMetric,
    ExperimentData,
    IObjFunc,
    _reduce_error,
)
from pyvale.sensorsim.enums import EExpSimKeys
from pyvale.sensorsim.sensordata import SensorData


@dataclass(slots=True, kw_only=True)
class ConstrainObjVal(IConstrainOut):
    """Require every raw value returned by a minimisation objective to meet a cap."""

    objective: IObjFunc
    max_value: float

    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        return self.objective.get_required_outputs()

    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        if not np.isfinite(self.max_value):
            raise ValueError("max_value must be finite.")
        values = self.objective.eval(
            sensor_data=sensor_data,
            experiment_data=experiment_data,
            scenario_keys=scenario_keys,
        )
        if not isinstance(values, np.ndarray) or values.size == 0:
            raise ValueError("A constrained objective must return a nonempty array.")
        return np.asarray(values - self.max_value, dtype=np.float64).ravel()


@dataclass(slots=True, kw_only=True)
class ConstrainMeasErr(IConstrainOut):
    """Limit measurement error versus nominal truth for the specified arrays.

    Applies independently to each selected array and realisation, reducing over
    sensor, component and time. Use arrays with comparable component units.
    A layout with none of the specified arrays selected fails the constraint.
    """

    sensor_keys: tuple[str, ...]
    max_error: float
    metric: EErrMetric = EErrMetric.RMSE

    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        return frozenset({EExpSimKeys.MEASUREMENT, EExpSimKeys.TRUTH})

    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        if not self.sensor_keys:
            raise ValueError("Specify at least one array for the measurement error limit.")
        if not np.isfinite(self.max_error) or self.max_error < 0:
            raise ValueError("max_error must be finite and nonnegative.")

        residuals: list[npt.NDArray[np.float64]] = []
        for key in self.sensor_keys:
            if key not in sensor_data:
                continue
            for scenario in scenario_keys:
                measured = experiment_data[scenario, key, EExpSimKeys.MEASUREMENT]
                truth = experiment_data[scenario, key, EExpSimKeys.TRUTH]
                if measured.ndim != 4 or measured.size == 0 or measured.shape != truth.shape:
                    raise ValueError("Measurement and truth must have matching 4D histories.")
                error = _reduce_error(measured - truth, metric=self.metric, axes=(1, 2, 3))
                residuals.append(error - self.max_error)
        if not residuals:
            return np.ones(1, dtype=np.float64)
        return np.concatenate(residuals)
