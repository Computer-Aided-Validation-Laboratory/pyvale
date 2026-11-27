"""Objectives computed from a proposed sensor layout and shared experiment data."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.sensorsim.enums import EExpSimKeys
from pyvale.sensorsim.sensordata import SensorData

ExperimentData = Mapping[tuple[str, str, EExpSimKeys], npt.NDArray[np.float64]]


class IObjFunc(ABC):
    """A criterion to minimise, evaluated for one proposed sensor layout."""

    @abstractmethod
    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        """Declare which experiment outputs this objective reads."""

    @abstractmethod
    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        """Return a complete scalar or losses with axes (scenario, realisation).

        Experiment keys contain a scenario, sensor array name and output kind.
        Measurement, truth and error arrays have axes (realisation, sensor,
        component, time). Nominal time and position arrays have no realisation axis.
        """


@dataclass(slots=True, kw_only=True)
class ObjSignal(IObjFunc):
    """Maximise the RMS of the nominal signal over sensors, components and time.

    Combine arrays only when their components have comparable units. The returned
    loss is the negative RMS of all selected observations in each scenario.
    """

    sensor_keys: tuple[str, ...]

    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        return frozenset({EExpSimKeys.TRUTH})

    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        if not self.sensor_keys:
            raise ValueError("ObjSignal requires at least one sensor array key.")

        losses: list[npt.NDArray[np.float64]] = []
        for scenario_key in scenario_keys:
            sum_squares = None
            num_observations = 0
            for sensor_key in self.sensor_keys:
                truth = experiment_data[scenario_key, sensor_key, EExpSimKeys.TRUTH]
                squared_signal = np.sum(np.square(truth), axis=(1, 2, 3))
                sum_squares = (
                    squared_signal
                    if sum_squares is None
                    else sum_squares + squared_signal
                )
                num_observations += int(np.prod(truth.shape[1:]))

            if sum_squares is None or num_observations == 0:
                raise ValueError("ObjSignal requires at least one observation.")
            losses.append(-np.sqrt(sum_squares / num_observations))

        return np.stack(losses, axis=0)
