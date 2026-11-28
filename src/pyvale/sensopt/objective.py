# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Objectives computed from a proposed sensor layout and shared experiment data."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from math import log, log1p

import numpy as np
import numpy.typing as npt

from pyvale.sensopt.designspace import _calculate_sensor_cost
from pyvale.sensopt.evalstrat import EvalStrat
from pyvale.sensopt.fieldrecon import FieldReconNearest, IFieldRecon
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
        Arrays with no selected sensors are absent from the layout and histories.
        """


class IObjNormalisation(ABC):
    """Transform one scalar objective before weighting."""

    @abstractmethod
    def eval(self, score: float) -> float:
        """Return the normalised score."""


@dataclass(slots=True, kw_only=True)
class ObjNormAffine(IObjNormalisation):
    """Apply a positive gain and an optional offset."""

    gain: float
    offset: float = 0.0

    def eval(self, score: float) -> float:
        if not np.isfinite(self.gain) or self.gain <= 0 or not np.isfinite(self.offset):
            raise ValueError("Affine normalisation requires positive gain and finite parameters.")
        return self.gain * score + self.offset


@dataclass(slots=True, kw_only=True)
class ObjNormLog(IObjNormalisation):
    """Take the natural logarithm of a positive loss relative to a reference."""

    reference: float

    def eval(self, score: float) -> float:
        if not np.isfinite(self.reference) or self.reference <= 0 or score <= 0:
            raise ValueError("Log normalisation requires a positive score and reference.")
        return log(score) - log(self.reference)


@dataclass(slots=True, kw_only=True)
class ObjNormLog1p(IObjNormalisation):
    """Log-scale a nonnegative loss while preserving zero."""

    reference: float

    def eval(self, score: float) -> float:
        if not np.isfinite(self.reference) or self.reference <= 0 or score < 0:
            raise ValueError(
                "Log1p normalisation requires a nonnegative score and positive reference."
            )
        return log1p(score / self.reference)


@dataclass(slots=True, kw_only=True)
class ObjSpec:
    """Pair an objective with its evaluation strategy."""

    objective: IObjFunc
    eval_strat: EvalStrat


@dataclass(slots=True, kw_only=True)
class ObjTerm:
    """One objective, its optional fixed normalisation and its weight.

    If no input array has selected sensors, use empty_score before normalising.
    Built-in signal and reconstruction objectives supply their own sensor keys.
    Custom objectives using empty_score must specify sensor_keys on the term.
    """

    objective: IObjFunc | ObjSpec
    weight: float = 1.0
    normalisation: IObjNormalisation | None = None
    empty_score: float | None = None
    sensor_keys: tuple[str, ...] | None = None


@dataclass(slots=True, kw_only=True)
class ObjWeighted:
    """Combine scalar objective terms into a single minimisation score."""

    terms: tuple[ObjTerm, ...]


@dataclass(slots=True, kw_only=True)
class ObjSensorCost(IObjFunc):
    """Sum installed sensor counts times their array-specific unit costs."""

    unit_costs: Mapping[str, float]

    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        return frozenset()

    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        cost = _calculate_sensor_cost(sensor_data, self.unit_costs)
        return np.array(cost, dtype=np.float64)


class ESignalAxis(Enum):
    """Axes within one sensor observation history."""

    POSITION = auto()
    COMPONENT = auto()
    TIME = auto()


class ESignalReduce(Enum):
    """Statistics for reducing a nominal signal."""

    L2 = auto()
    RMS = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
    MAX_ABS = auto()


@dataclass(slots=True, kw_only=True)
class SignalReduction:
    """Reduce one named axis or a group of axes jointly."""

    axis: ESignalAxis | tuple[ESignalAxis, ...]
    method: ESignalReduce


_SIGNAL_AXES = (ESignalAxis.POSITION, ESignalAxis.COMPONENT, ESignalAxis.TIME)


@dataclass(slots=True, kw_only=True)
class ObjSignal(IObjFunc):
    """Maximise a nominal signal statistic using the supplied reduction order.

    An optional time index selects one sample before reduction. Every remaining
    signal axis must be reduced. Scenarios and realisations are preserved.
    Multiple arrays must use comparable units and agree on the axes that remain
    when positions are pooled. A joint reduction over all three axes pools
    individual readings even when the histories have different shapes.
    """

    sensor_keys: tuple[str, ...]
    reductions: tuple[SignalReduction, ...]
    time_index: int | None = None

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

        axes = list(_SIGNAL_AXES)
        if self.time_index is not None:
            axes.remove(ESignalAxis.TIME)

        steps: list[tuple[tuple[int, ...], bool, ESignalReduce]] = []
        for reduction in self.reductions:
            named_axes = reduction.axis
            if isinstance(named_axes, ESignalAxis):
                named_axes = (named_axes,)
            if not named_axes or len(set(named_axes)) != len(named_axes):
                raise ValueError("A signal reduction needs distinct axes.")
            if any(axis not in axes for axis in named_axes):
                raise ValueError("Each available signal axis can be reduced only once.")
            indices = tuple(axes.index(axis) + 1 for axis in named_axes)
            steps.append(
                (indices, ESignalAxis.POSITION in named_axes, reduction.method)
            )
            for axis in named_axes:
                axes.remove(axis)
        if axes:
            raise ValueError("Reduce all signal axes or select a single time sample.")

        losses: list[npt.NDArray[np.float64]] = []
        for scenario_key in scenario_keys:
            histories: list[npt.NDArray[np.float64]] = []
            for sensor_key in self.sensor_keys:
                output_key = (scenario_key, sensor_key, EExpSimKeys.TRUTH)
                if output_key not in experiment_data:
                    continue
                truth = experiment_data[output_key]
                if truth.ndim != 4 or truth.size == 0:
                    raise ValueError("Signal histories must be nonempty 4D arrays.")
                if self.time_index is not None:
                    truth = truth[..., self.time_index]
                histories.append(truth)
            if not histories:
                raise ValueError("Signal scoring requires at least one selected sensor.")

            for indices, pool_positions, method in steps:
                if pool_positions and len(histories) > 1:
                    pooled: list[npt.NDArray[np.float64]] = []
                    for history in histories:
                        remaining = tuple(
                            index
                            for index in range(history.ndim)
                            if index not in indices
                        )
                        ordered = history.transpose(remaining + indices)
                        kept_shape = ordered.shape[: len(remaining)]
                        pooled.append(ordered.reshape(*kept_shape, -1))
                    histories = [np.concatenate(pooled, axis=-1)]
                    indices = (-1,)
                histories = [
                    _reduce_signal(history, axes=indices, method=method)
                    for history in histories
                ]
            losses.append(-histories[0])

        return np.stack(losses, axis=0)


def _reduce_signal(
    values: npt.NDArray[np.float64],
    *,
    axes: tuple[int, ...],
    method: ESignalReduce,
) -> npt.NDArray[np.float64]:
    match method:
        case ESignalReduce.L2:
            return np.sqrt(np.sum(np.square(values), axis=axes))
        case ESignalReduce.RMS:
            return np.sqrt(np.mean(np.square(values), axis=axes))
        case ESignalReduce.MEAN:
            return np.mean(values, axis=axes)
        case ESignalReduce.MAX:
            return np.max(values, axis=axes)
        case ESignalReduce.MIN:
            return np.min(values, axis=axes)
        case ESignalReduce.MAX_ABS:
            return np.max(np.abs(values), axis=axes)
    raise ValueError(f"Unsupported signal reduction {method!r}.")


@dataclass(slots=True, kw_only=True)
class ObjSignalRMS(IObjFunc):
    """Maximise the RMS statistic pooled across all selected readings."""

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
        return ObjSignal(
            sensor_keys=self.sensor_keys,
            reductions=(SignalReduction(axis=_SIGNAL_AXES, method=ESignalReduce.RMS),),
        ).eval(
            sensor_data=sensor_data,
            experiment_data=experiment_data,
            scenario_keys=scenario_keys,
        )


@dataclass(slots=True, kw_only=True)
class ObjSignalPeakAbs(IObjFunc):
    """Maximise the PeakAbs statistic pooled across all selected readings."""

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
        return ObjSignal(
            sensor_keys=self.sensor_keys,
            reductions=(
                SignalReduction(axis=_SIGNAL_AXES, method=ESignalReduce.MAX_ABS),
            ),
        ).eval(
            sensor_data=sensor_data,
            experiment_data=experiment_data,
            scenario_keys=scenario_keys,
        )


class EErrMetric(Enum):
    """Metrics for reducing errors over observation axes."""

    RMSE = auto()
    MAE = auto()
    MAX_ERR = auto()  # Maximum absolute error.


class EReconRef(Enum):
    """Reference field used to score an individual reconstruction."""

    SIM_SCENARIO = auto()
    MEAN_SCENARIO = auto()
    MEDIAN_SCENARIO = auto()


def _reduce_error(
    error: npt.NDArray[np.float64], *, metric: EErrMetric, axes: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    match metric:
        case EErrMetric.RMSE:
            return np.sqrt(np.mean(np.square(error), axis=axes))
        case EErrMetric.MAE:
            return np.mean(np.abs(error), axis=axes)
        case EErrMetric.MAX_ERR:
            return np.max(np.abs(error), axis=axes)
        case _:
            raise ValueError(f"Unsupported error metric {metric!r}.")


@dataclass(slots=True, kw_only=True)
class ObjFieldRecon(IObjFunc):
    """Score reconstruction at user-supplied target coordinates.

    Reference values are keyed by scenario and have shape (target, time).
    Each target and time has equal weight. Losses have shape
    (scenario, realisation), in the units of the selected component.
    """

    target_positions: npt.NDArray[np.float64]
    target_values: Mapping[str, npt.NDArray[np.float64]] | None = None
    sensor_keys: tuple[str, ...]
    component_index: int
    metric: EErrMetric = EErrMetric.RMSE
    reconstructor: IFieldRecon | None = None
    reference_field: str | None = None
    reference_mode: EReconRef = EReconRef.SIM_SCENARIO
    reference_weights: Mapping[str, float] | None = None

    def get_required_outputs(self) -> frozenset[EExpSimKeys]:
        return frozenset({EExpSimKeys.MEASUREMENT, EExpSimKeys.SENSOR_TIMES})

    def eval(
        self,
        *,
        sensor_data: Mapping[str, SensorData],
        experiment_data: ExperimentData,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        if self.target_values is None:
            raise ValueError("Reference field values must be resolved before evaluation.")
        active_keys: tuple[str, ...] = tuple(
            key for key in self.sensor_keys if key in sensor_data
        )
        if not active_keys:
            raise ValueError("Field reconstruction requires an active sensor array.")

        positions: list[npt.NDArray[np.float64]] = []
        for key in active_keys:
            sensor_positions = sensor_data[key].positions
            if sensor_positions is None:
                raise ValueError(f"Sensor array {key!r} requires positions.")
            positions.append(sensor_positions)

        sensor_positions = np.concatenate(positions, axis=0)
        reconstructor = self.reconstructor
        if reconstructor is None:
            reconstructor = FieldReconNearest()

        losses: list[npt.NDArray[np.float64]] = []
        for scenario_key in scenario_keys:
            if len(active_keys) > 1:
                times = experiment_data[
                    scenario_key, active_keys[0], EExpSimKeys.SENSOR_TIMES
                ]
                for key in active_keys[1:]:
                    if not np.array_equal(
                        times, experiment_data[scenario_key, key, EExpSimKeys.SENSOR_TIMES]
                    ):
                        raise ValueError("Pooled reconstruction requires matching sample times.")
            measurements: npt.NDArray[np.float64] = np.concatenate(
                [
                    experiment_data[scenario_key, key, EExpSimKeys.MEASUREMENT]
                    for key in active_keys
                ],
                axis=1,
            )
            # The selected component uses a common time grid and equal nodal weights.
            sensor_values = np.moveaxis(
                measurements[:, :, self.component_index, :], 1, 0
            )
            nodal_values = reconstructor.reconstruct(
                sensor_positions=sensor_positions,
                sensor_values=sensor_values,
                target_positions=self.target_positions,
            )
            predicted = np.moveaxis(nodal_values, 0, 1)
            reference = self.target_values[scenario_key]
            if reference.shape != predicted.shape[1:]:
                raise ValueError(
                    "Reference values must match the target and time axes."
                )
            losses.append(
                _reduce_error(predicted - reference, metric=self.metric, axes=(1, 2))
            )

        return np.stack(losses, axis=0)
