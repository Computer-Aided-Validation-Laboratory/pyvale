# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Direct sampling of analytic field functions."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt

from pyvale.sensorsim.fieldsamplerinterface import FieldFunc, IFieldSampler


@dataclass(slots=True, kw_only=True, frozen=True)
class FieldSampleBounds:
    """Inclusive bounds in the field's coordinate and time units."""

    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None
    time: tuple[float, float] | None = None


class FieldSampleAnalytic(IFieldSampler):
    """Evaluate a world-frame function at sensor positions and times.

    Parameters
    ----------
    evaluate : FieldFunc
        Receives XYZ points shaped (point, 3) and times shaped (time,).
        Returns world-frame values shaped (point, component, time).
    default_times : numpy.ndarray
        Strictly increasing times used when no times are requested.
    num_components : int
        Size of the component axis, including for out-of-bounds calls.
    bounds : FieldSampleBounds or None, optional
        Inclusive limits in the same units as the input positions and times.
        Samples outside these limits are NaN.
    """

    def __init__(
        self,
        *,
        evaluate: FieldFunc,
        default_times: npt.NDArray[np.float64],
        num_components: int,
        bounds: FieldSampleBounds | None = None,
    ) -> None:
        if not callable(evaluate):
            raise TypeError("evaluate must be a callable.")
        if (
            not isinstance(num_components, Integral)
            or isinstance(num_components, bool)
            or num_components < 1
        ):
            raise ValueError("num_components must be a positive integer.")
        times = np.array(default_times, dtype=np.float64, copy=True)
        if (
            times.ndim != 1
            or not times.size
            or not np.all(np.isfinite(times))
            or np.any(times[1:] <= times[:-1])
        ):
            raise ValueError("Default times must be finite and strictly increasing.")
        if bounds is not None:
            for interval in (bounds.x, bounds.y, bounds.z, bounds.time):
                if interval is not None:
                    lower, upper = interval
                    if (
                        not np.isfinite(lower)
                        or not np.isfinite(upper)
                        or lower > upper
                    ):
                        raise ValueError(
                            "Field sample bounds must be finite and ordered."
                        )
        self._evaluate = evaluate
        self._times = times
        self._num_components = int(num_components)
        self._bounds = bounds

    def get_time_steps(self) -> npt.NDArray[np.float64]:
        return self._times.copy()

    def get_num_components(self) -> int:
        return self._num_components

    def sample(
        self,
        points: npt.NDArray[np.float64],
        times: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        points = np.asarray(points, dtype=np.float64)
        times = self._times if times is None else np.asarray(times, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or times.ndim != 1:
            raise ValueError(
                "Expected points shaped (point, 3) and times shaped (time,)."
            )

        valid_points = np.all(np.isfinite(points), axis=1)
        valid_times = np.isfinite(times)
        bounds = self._bounds
        if bounds is not None:
            for axis, interval in enumerate((bounds.x, bounds.y, bounds.z)):
                if interval is not None:
                    lower, upper = interval
                    valid_points &= (lower <= points[:, axis]) & (
                        points[:, axis] <= upper
                    )
            if bounds.time is not None:
                lower, upper = bounds.time
                valid_times &= (lower <= times) & (times <= upper)

        result = np.full((len(points), self._num_components, len(times)), np.nan)
        if np.any(valid_points) and np.any(valid_times):
            values = np.asarray(
                self._evaluate(points[valid_points], times[valid_times]),
                dtype=np.float64,
            )
            expected = (
                np.count_nonzero(valid_points),
                self._num_components,
                np.count_nonzero(valid_times),
            )
            if values.shape != expected:
                raise ValueError(
                    f"Field function must return samples with shape {expected}."
                )
            result[
                np.ix_(valid_points, np.arange(self._num_components), valid_times)
            ] = values
        return result
