# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Candidate sampling times for discrete optimisation."""

from itertools import combinations
from numbers import Integral

import numpy as np
import numpy.typing as npt


def _validate_sample_times(times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    schedule = np.asarray(times, dtype=np.float64)
    if (
        schedule.ndim != 1
        or not schedule.size
        or not np.all(np.isfinite(schedule))
        or np.any(schedule[1:] <= schedule[:-1])
    ):
        raise ValueError("Sample times must be finite and strictly increasing.")
    return schedule


def gen_sample_time_combinations(
    *,
    candidate_times: npt.NDArray[np.float64],
    num_samples: int,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Enumerate complete schedules with exactly num_samples distinct times."""
    times = _validate_sample_times(candidate_times)
    if (
        not isinstance(num_samples, Integral)
        or isinstance(num_samples, bool)
        or not 1 <= num_samples <= len(times)
    ):
        raise ValueError("num_samples must be a positive integer within the time pool.")
    return tuple(
        np.array(schedule, dtype=np.float64)
        for schedule in combinations(times, num_samples)
    )


def gen_sample_times_from_periods(
    *,
    periods: npt.NDArray[np.float64],
    time_window: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Generate one regular schedule per period, starting at the left endpoint."""
    start, stop = time_window
    if not np.isfinite(start) or not np.isfinite(stop) or start > stop:
        raise ValueError("time_window must have finite, ordered endpoints.")
    periods = np.asarray(periods, dtype=np.float64)
    if (
        periods.ndim != 1
        or not periods.size
        or not np.all(np.isfinite(periods))
        or np.any(periods <= 0)
    ):
        raise ValueError("Periods must be a nonempty array of finite positive values.")

    schedules: list[npt.NDArray[np.float64]] = []
    for period in periods:
        steps = (stop - start) / period
        if not np.isfinite(steps):
            raise ValueError(
                "The time window contains too many samples for this period."
            )
        nearest_step = np.rint(steps)
        # Snap a mathematically on-grid endpoint affected by division roundoff.
        on_grid = abs(steps - nearest_step) <= 4 * np.spacing(steps)
        num_intervals = int(nearest_step) if on_grid else int(np.floor(steps))
        schedule = start + np.arange(num_intervals + 1, dtype=np.float64) * period
        if on_grid and num_intervals:
            schedule[-1] = stop
        schedules.append(_validate_sample_times(schedule))
    return tuple(schedules)


def gen_sample_times_from_freqs(
    *,
    frequencies: npt.NDArray[np.float64],
    time_window: tuple[float, float],
) -> tuple[npt.NDArray[np.float64], ...]:
    """Generate one regular schedule per frequency in inverse time units."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("Frequencies must be finite and positive.")
    with np.errstate(over="ignore"):
        periods = np.reciprocal(frequencies)
    return gen_sample_times_from_periods(periods=periods, time_window=time_window)
