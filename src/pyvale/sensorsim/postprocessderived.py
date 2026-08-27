# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Multi-sensor fusion processors and derived quantity calculators."""

from typing import Callable, Any, Sequence
import numpy as np
from scipy import integrate

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessor import IMeasurementProcessor
from pyvale.sensorsim.sensordescriptor import SensorDescriptor


class ProcessStiffness(IMeasurementProcessor):
    """Calculates dynamic secant stiffness k(t) = F(t) / u(t) from paired
    force and displacement measurements.
    """

    __slots__ = ("_force_key", "_disp_key", "_eps", "_units")

    def __init__(
        self,
        force: str,
        disp: str,
        eps: float = 1e-9,
        units: str = "N/mm",
    ) -> None:
        self._force_key = force
        self._disp_key = disp
        self._eps = float(eps)
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._force_key, self._disp_key)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return ("stiffness",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        f_data = inputs[self._force_key]
        u_data = inputs[self._disp_key]

        f_vals = f_data.values
        u_vals = u_data.values

        denom = np.where(np.abs(u_vals) < self._eps, self._eps, u_vals)
        k_vals = f_vals / denom

        desc = SensorDescriptor(
            name="Dynamic Stiffness", tag="STIFF", units=self._units
        )

        return MeasurementData(
            values=k_vals,
            sample_times=f_data.sample_times,
            positions=f_data.positions,
            components=("stiffness",),
            units=self._units,
            descriptor=desc,
        )


class ProcessWork(IMeasurementProcessor):
    """Calculates mechanical work / dissipated strain energy
    W(t) = ∫ F(t) u_dot(t) dt from force and displacement measurements.
    """

    __slots__ = ("_force_key", "_disp_key", "_units")

    def __init__(
        self,
        force: str,
        disp: str,
        units: str = "mJ",
    ) -> None:
        self._force_key = force
        self._disp_key = disp
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._force_key, self._disp_key)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return ("work",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        f_data = inputs[self._force_key]
        u_data = inputs[self._disp_key]

        f_vals = f_data.values
        u_vals = u_data.values
        times = f_data.sample_times

        t_1d = times[0] if times.ndim > 1 else times
        u_dot = np.gradient(u_vals, t_1d, axis=-1)

        power = f_vals * u_dot
        work = integrate.cumulative_trapezoid(
            power, x=t_1d, axis=-1, initial=0.0
        )

        desc = SensorDescriptor(
            name="Mechanical Work", tag="WORK", units=self._units
        )

        return MeasurementData(
            values=work,
            sample_times=f_data.sample_times,
            positions=f_data.positions,
            components=("work",),
            units=self._units,
            descriptor=desc,
        )


class ProcessRelativeDifference(IMeasurementProcessor):
    """Calculates relative difference Δy(t) = y_B(t) - y_A(t) between two
    sensor measurement datasets.
    """

    __slots__ = ("_source_a", "_source_b", "_label", "_units")

    def __init__(
        self,
        source_a: str,
        source_b: str,
        label: str = "difference",
        units: str | None = None,
    ) -> None:
        self._source_a = source_a
        self._source_b = source_b
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source_a, self._source_b)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return (self._label,)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data_a = inputs[self._source_a]
        data_b = inputs[self._source_b]

        diff_vals = data_b.values - data_a.values
        units_out = self._units if self._units is not None else data_a.units

        desc = SensorDescriptor(
            name=self._label, tag="DIFF", units=units_out
        )

        return MeasurementData(
            values=diff_vals,
            sample_times=data_a.sample_times,
            positions=data_b.positions,
            components=(self._label,),
            units=units_out,
            descriptor=desc,
        )


class ProcessRatio(IMeasurementProcessor):
    """Calculates ratio r(t) = y_B(t) / y_A(t) between two sensor datasets."""

    __slots__ = ("_source_a", "_source_b", "_eps", "_label", "_units")

    def __init__(
        self,
        source_a: str,
        source_b: str,
        eps: float = 1e-9,
        label: str = "ratio",
        units: str = "",
    ) -> None:
        self._source_a = source_a
        self._source_b = source_b
        self._eps = float(eps)
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source_a, self._source_b)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return (self._label,)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data_a = inputs[self._source_a]
        data_b = inputs[self._source_b]

        denom = np.where(
            np.abs(data_a.values) < self._eps, self._eps, data_a.values
        )
        ratio_vals = data_b.values / denom

        desc = SensorDescriptor(
            name=self._label, tag="RATIO", units=self._units
        )

        return MeasurementData(
            values=ratio_vals,
            sample_times=data_a.sample_times,
            positions=data_b.positions,
            components=(self._label,),
            units=self._units,
            descriptor=desc,
        )


class ProcessMagnitude(IMeasurementProcessor):
    """Computes Euclidean magnitude norm of vector components:
    ||v|| = sqrt(sum(v_i^2)).
    """

    __slots__ = ("_source", "_label", "_units")

    def __init__(
        self,
        source: str,
        label: str = "magnitude",
        units: str | None = None,
    ) -> None:
        self._source = source
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return (self._label,)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        mag = np.linalg.norm(vals, axis=-2, keepdims=True)

        units_out = self._units if self._units is not None else data.units
        desc = SensorDescriptor(
            name=self._label, tag="MAG", units=units_out
        )

        return MeasurementData(
            values=mag,
            sample_times=data.sample_times,
            positions=data.positions,
            components=(self._label,),
            units=units_out,
            descriptor=desc,
        )


class ProcessCustom(IMeasurementProcessor):
    """Wraps an arbitrary user-defined callable functional into the
    IMeasurementProcessor interface.
    """

    __slots__ = ("_sources", "_func", "_output_components", "_kwargs")

    def __init__(
        self,
        sources: Sequence[str] | dict[str, str],
        func: Callable[..., MeasurementData | np.ndarray],
        output_components: tuple[str, ...] = ("custom",),
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        sources : Sequence[str] | dict[str, str]
            Names of input sources required by the custom function.
        func : Callable[..., MeasurementData | np.ndarray]
            User-defined function accepting inputs dictionary and kwargs.
        output_components : tuple[str, ...], optional
            Names of output components. Default is ('custom',).
        """
        if isinstance(sources, dict):
            self._sources = tuple(sources.values())
        else:
            self._sources = tuple(sources)

        self._func = func
        self._output_components = output_components
        self._kwargs = kwargs

    def get_source_keys(self) -> tuple[str, ...]:
        return self._sources

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return self._output_components

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        res = self._func(inputs, **self._kwargs)
        if isinstance(res, MeasurementData):
            return res

        ref_data = next(iter(inputs.values()))
        return MeasurementData(
            values=np.asarray(res),
            sample_times=ref_data.sample_times,
            positions=ref_data.positions,
            components=self._output_components,
            units=ref_data.units,
            descriptor=ref_data.descriptor,
        )
