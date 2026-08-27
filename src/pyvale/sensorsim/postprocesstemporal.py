# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Temporal differentiation and integration signal processors for simulated
measurements.
"""

from typing import Literal
import numpy as np
from scipy import interpolate, integrate, signal

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessor import IMeasurementProcessor
from pyvale.sensorsim.sensordescriptor import SensorDescriptor


class ProcessDifferentiateTime(IMeasurementProcessor):
    """Computes numerical time derivatives (velocity, acceleration, strain rate)
    of measured signals.
    """

    __slots__ = (
        "_source",
        "_order",
        "_method",
        "_label",
        "_units",
        "_savgol_window",
        "_savgol_polyorder",
    )

    def __init__(
        self,
        source: str,
        order: int = 1,
        method: Literal["finite_diff", "spline", "savgol"] = "finite_diff",
        label: str | None = None,
        units: str = "",
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
    ) -> None:
        self._source = source
        self._order = int(order)
        self._method = method
        self._label = label
        self._units = units
        self._savgol_window = savgol_window
        self._savgol_polyorder = savgol_polyorder

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is None:
            prefix = self._label if self._label else "d_dt"
            return (f"{prefix}",)

        if self._label is not None:
            if len(data.components) == 1:
                return (self._label,)
            return tuple(f"{self._label}_{c}" for c in data.components)

        suffix = "dot" if self._order == 1 else f"d{self._order}"
        return tuple(f"{c}_{suffix}" for c in data.components)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        times = data.sample_times

        if times is None:
            t_1d = np.linspace(0.0, 1.0, vals.shape[-1])
        else:
            t_1d = times[0] if times.ndim > 1 else times

        if self._method == "savgol":
            n_times = vals.shape[-1]
            w_len = min(self._savgol_window, n_times)
            if w_len % 2 == 0:
                w_len = max(3, w_len - 1)
            p_order = min(self._savgol_polyorder, w_len - 1)
            dt = float(np.mean(np.diff(t_1d))) if len(t_1d) > 1 else 1.0
            dt = dt if dt > 0.0 else 1.0
            deriv_vals = signal.savgol_filter(
                vals,
                window_length=w_len,
                polyorder=p_order,
                deriv=self._order,
                delta=dt,
                axis=-1,
                mode="interp",
            )

        elif self._method == "spline":
            orig_shape = vals.shape
            flat_vals = vals.reshape(-1, orig_shape[-1])
            deriv_flat = np.empty_like(flat_vals)
            for ii in range(flat_vals.shape[0]):
                spl = interpolate.CubicSpline(t_1d, flat_vals[ii, :])
                spl_deriv = spl.derivative(nu=self._order)
                deriv_flat[ii, :] = spl_deriv(t_1d)
            deriv_vals = deriv_flat.reshape(orig_shape)

        else:
            current_vals = vals
            for _ in range(self._order):
                current_vals = np.gradient(current_vals, t_1d, axis=-1)
            deriv_vals = current_vals

        out_comps = self.get_output_components({self._source: data})
        desc = (
            SensorDescriptor(name=self._label, tag="DERIV", units=self._units)
            if self._label
            else data.descriptor
        )

        return MeasurementData(
            values=deriv_vals,
            sample_times=data.sample_times,
            positions=data.positions,
            components=out_comps,
            units=self._units if self._units else data.units,
            descriptor=desc,
        )


class ProcessIntegrateTime(IMeasurementProcessor):
    """Computes cumulative temporal integrals (displacement from velocity,
    velocity from acceleration, impulse from force) with initial conditions.
    """

    __slots__ = (
        "_source",
        "_initial_value",
        "_method",
        "_label",
        "_units",
    )

    def __init__(
        self,
        source: str,
        initial_value: float | np.ndarray = 0.0,
        method: Literal["trapezoid", "simpson"] = "trapezoid",
        label: str | None = None,
        units: str = "",
    ) -> None:
        self._source = source
        self._initial_value = initial_value
        self._method = method
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        data = input_metadata.get(self._source)
        if data is None:
            prefix = self._label if self._label else "int_dt"
            return (f"{prefix}",)

        if self._label is not None:
            if len(data.components) == 1:
                return (self._label,)
            return tuple(f"{self._label}_{c}" for c in data.components)

        return tuple(f"{c}_int" for c in data.components)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        times = data.sample_times

        if times is None:
            t_1d = np.linspace(0.0, 1.0, vals.shape[-1])
        else:
            t_1d = times[0] if times.ndim > 1 else times

        cum_integral = integrate.cumulative_trapezoid(
            vals, x=t_1d, axis=-1, initial=0.0
        )

        if isinstance(self._initial_value, np.ndarray):
            init_val = self._initial_value
            while init_val.ndim < cum_integral.ndim:
                init_val = np.expand_dims(init_val, axis=0)
            cum_integral = cum_integral + init_val
        else:
            cum_integral = cum_integral + float(self._initial_value)

        out_comps = self.get_output_components({self._source: data})
        desc = (
            SensorDescriptor(name=self._label, tag="INT", units=self._units)
            if self._label
            else data.descriptor
        )

        return MeasurementData(
            values=cum_integral,
            sample_times=data.sample_times,
            positions=data.positions,
            components=out_comps,
            units=self._units if self._units else data.units,
            descriptor=desc,
        )
