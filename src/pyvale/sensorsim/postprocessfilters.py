# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Pre-smoothing and noise filtering signal processors for simulated sensor
data.
"""

import numpy as np
from scipy import signal, ndimage

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessor import IMeasurementProcessor


class ProcessFilterSavitzkyGolay(IMeasurementProcessor):
    """Applies Savitzky-Golay polynomial filtering along the time axis."""

    __slots__ = ("_source", "_window_length", "_polyorder", "_deriv", "_delta")

    def __init__(
        self,
        source: str,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0,
    ) -> None:
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1

        self._source = source
        self._window_length = window_length
        self._polyorder = polyorder
        self._deriv = deriv
        self._delta = delta

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is not None:
            return data.components
        return ("filtered",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        n_times = vals.shape[-1]
        w_len = min(self._window_length, n_times)
        if w_len % 2 == 0:
            w_len = max(3, w_len - 1)
        p_order = min(self._polyorder, w_len - 1)

        filtered = signal.savgol_filter(
            vals,
            window_length=w_len,
            polyorder=p_order,
            deriv=self._deriv,
            delta=self._delta,
            axis=-1,
            mode="interp",
        )

        return MeasurementData(
            values=filtered,
            sample_times=data.sample_times,
            positions=data.positions,
            components=data.components,
            units=data.units,
            descriptor=data.descriptor,
        )


class ProcessFilterGaussian(IMeasurementProcessor):
    """Applies 1D Gaussian smoothing convolution along the time axis."""

    __slots__ = ("_source", "_sigma", "_truncate")

    def __init__(
        self,
        source: str,
        sigma: float = 1.0,
        truncate: float = 4.0,
    ) -> None:
        self._source = source
        self._sigma = float(sigma)
        self._truncate = float(truncate)

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is not None:
            return data.components
        return ("filtered",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values

        filtered = ndimage.gaussian_filter1d(
            vals,
            sigma=self._sigma,
            axis=-1,
            truncate=self._truncate,
            mode="nearest",
        )

        return MeasurementData(
            values=filtered,
            sample_times=data.sample_times,
            positions=data.positions,
            components=data.components,
            units=data.units,
            descriptor=data.descriptor,
        )


class ProcessFilterMovingAverage(IMeasurementProcessor):
    """Applies a sliding uniform boxcar moving average filter along the time
    axis.
    """

    __slots__ = ("_source", "_window_length")

    def __init__(
        self,
        source: str,
        window_length: int = 5,
    ) -> None:
        self._source = source
        self._window_length = int(window_length)

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is not None:
            return data.components
        return ("filtered",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        w_len = max(1, self._window_length)

        filtered = ndimage.uniform_filter1d(
            vals,
            size=w_len,
            axis=-1,
            mode="nearest",
        )

        return MeasurementData(
            values=filtered,
            sample_times=data.sample_times,
            positions=data.positions,
            components=data.components,
            units=data.units,
            descriptor=data.descriptor,
        )


class ProcessFilterMedian(IMeasurementProcessor):
    """Applies a sliding 1D median filter along the time axis to remove
    impulse noise and outlier spikes.
    """

    __slots__ = ("_source", "_size")

    def __init__(
        self,
        source: str,
        size: int = 5,
    ) -> None:
        self._source = source
        self._size = int(size)

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is not None:
            return data.components
        return ("filtered",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values

        filtered = ndimage.median_filter(
            vals,
            size=(1,) * (vals.ndim - 1) + (self._size,),
            mode="nearest",
        )

        return MeasurementData(
            values=filtered,
            sample_times=data.sample_times,
            positions=data.positions,
            components=data.components,
            units=data.units,
            descriptor=data.descriptor,
        )


class ProcessFilterButterworth(IMeasurementProcessor):
    """Applies forward-backward zero-phase digital Butterworth IIR filtering
    along the time axis.
    """

    __slots__ = ("_source", "_cutoff", "_order", "_btype", "_sampling_rate")

    def __init__(
        self,
        source: str,
        cutoff: float | tuple[float, float],
        order: int = 4,
        btype: str = "lowpass",
        sampling_rate: float | None = None,
    ) -> None:
        self._source = source
        self._cutoff = cutoff
        self._order = int(order)
        self._btype = str(btype)
        self._sampling_rate = sampling_rate

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is not None:
            return data.components
        return ("filtered",)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        times = data.sample_times

        if self._sampling_rate is not None:
            fs = self._sampling_rate
        else:
            if times.ndim == 1 and len(times) > 1:
                dt = float(np.mean(np.diff(times)))
                fs = 1.0 / dt if dt > 0.0 else 1.0
            else:
                fs = 1.0

        sos = signal.butter(
            self._order,
            self._cutoff,
            btype=self._btype,
            fs=fs,
            output="sos",
        )
        filtered = signal.sosfiltfilt(sos, vals, axis=-1)

        return MeasurementData(
            values=filtered,
            sample_times=data.sample_times,
            positions=data.positions,
            components=data.components,
            units=data.units,
            descriptor=data.descriptor,
        )
