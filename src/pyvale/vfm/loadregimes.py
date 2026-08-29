"""Resolve and freeze load-step regimes from predicted yielded fractions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True, frozen=True)
class LoadRegimeThresholds:
    onset: float = 0.02
    developed: float = 0.20
    late: float = 0.65

    def __post_init__(self) -> None:
        values = (self.onset, self.developed, self.late)
        if not 0.0 <= values[0] < values[1] < values[2] <= 1.0:
            raise ValueError("Require 0 <= onset < developed < late <= 1.")


@dataclass(slots=True, frozen=True)
class ResolvedLoadRegimes:
    thresholds: LoadRegimeThresholds
    yielded_fraction: tuple[float, ...]
    pre_yield: tuple[int, ...]
    onset: tuple[int, ...]
    developed: tuple[int, ...]
    late: tuple[int, ...]
    resolution: str = "absolute_yielded_fraction"
    normalised_progress: tuple[float, ...] | None = None

    def indices(self, name: str) -> tuple[int, ...]:
        if name not in {"pre_yield", "onset", "developed", "late"}:
            raise KeyError(f"Unknown load regime {name!r}.")
        return getattr(self, name)

    def diagnostics(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("pre_yield", "onset", "developed", "late"):
            result[name] = list(result[name])
        result["yielded_fraction"] = list(result["yielded_fraction"])
        if result["normalised_progress"] is not None:
            result["normalised_progress"] = list(result["normalised_progress"])
        return result


def resolve_load_regimes(
    yielded_fraction: npt.ArrayLike,
    thresholds: LoadRegimeThresholds = LoadRegimeThresholds(),
    *,
    ensure_nonempty: bool = True,
) -> ResolvedLoadRegimes:
    """Partition frames using frozen Phase-0 yielded fractions."""

    fraction = np.asarray(yielded_fraction, dtype=np.float64)
    if fraction.ndim != 1 or fraction.size == 0:
        raise ValueError("yielded_fraction must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(fraction)) or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("yielded_fraction values must be finite and within [0, 1].")
    bins = {
        "pre_yield": np.flatnonzero(fraction < thresholds.onset),
        "onset": np.flatnonzero((fraction >= thresholds.onset) & (fraction < thresholds.developed)),
        "developed": np.flatnonzero((fraction >= thresholds.developed) & (fraction < thresholds.late)),
        "late": np.flatnonzero(fraction >= thresholds.late),
    }
    if ensure_nonempty:
        bins = _fill_empty_bins(fraction, bins, thresholds)
    return ResolvedLoadRegimes(
        thresholds=thresholds,
        yielded_fraction=tuple(float(value) for value in fraction),
        **{name: tuple(int(index) for index in indices) for name, indices in bins.items()},
    )


def resolve_relative_load_regimes(
    yielded_fraction: npt.ArrayLike,
    thresholds: LoadRegimeThresholds = LoadRegimeThresholds(
        onset=0.05,
        developed=0.50,
        late=0.80,
    ),
    *,
    minimum_frames: int = 2,
) -> ResolvedLoadRegimes:
    """Partition frames by progress relative to the observed Phase-0 range.

    This avoids unreachable specimen-independent absolute yielded-fraction
    thresholds. The resulting blocks are ordered and disjoint. If threshold
    crossings leave a block too small, contiguous boundaries are adjusted
    while retaining at least ``minimum_frames`` per block when possible.
    """

    fraction = np.asarray(yielded_fraction, dtype=np.float64)
    if fraction.ndim != 1 or fraction.size == 0:
        raise ValueError("yielded_fraction must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(fraction)) or np.any((fraction < 0.0) | (fraction > 1.0)):
        raise ValueError("yielded_fraction values must be finite and within [0, 1].")
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive.")
    if fraction.size < 4 * minimum_frames:
        raise ValueError(
            "At least four times minimum_frames load steps are required for "
            "four disjoint regimes."
        )
    monotone = np.maximum.accumulate(fraction)
    span = float(monotone[-1] - monotone[0])
    if span <= np.finfo(np.float64).eps:
        progress = np.linspace(0.0, 1.0, fraction.size)
    else:
        progress = (monotone - monotone[0]) / span
    proposed = [
        int(np.searchsorted(progress, thresholds.onset, side="left")),
        int(np.searchsorted(progress, thresholds.developed, side="left")),
        int(np.searchsorted(progress, thresholds.late, side="left")),
    ]
    boundaries = _regularise_boundaries(
        proposed,
        frame_count=fraction.size,
        minimum_frames=minimum_frames,
    )
    first, second, third = boundaries
    bins = {
        "pre_yield": np.arange(0, first),
        "onset": np.arange(first, second),
        "developed": np.arange(second, third),
        "late": np.arange(third, fraction.size),
    }
    return ResolvedLoadRegimes(
        thresholds=thresholds,
        yielded_fraction=tuple(float(value) for value in fraction),
        pre_yield=tuple(int(index) for index in bins["pre_yield"]),
        onset=tuple(int(index) for index in bins["onset"]),
        developed=tuple(int(index) for index in bins["developed"]),
        late=tuple(int(index) for index in bins["late"]),
        resolution="relative_monotone_yield_progress",
        normalised_progress=tuple(float(value) for value in progress),
    )


def _regularise_boundaries(
    proposed: list[int],
    *,
    frame_count: int,
    minimum_frames: int,
) -> tuple[int, int, int]:
    boundaries: list[int] = []
    previous = 0
    for boundary_index, value in enumerate(proposed):
        remaining_blocks = 3 - boundary_index
        lower = previous + minimum_frames
        upper = frame_count - remaining_blocks * minimum_frames
        resolved = min(max(int(value), lower), upper)
        boundaries.append(resolved)
        previous = resolved
    return tuple(boundaries)  # type: ignore[return-value]


def _fill_empty_bins(
    fraction: npt.NDArray[np.float64],
    bins: dict[str, npt.NDArray[np.int64]],
    thresholds: LoadRegimeThresholds,
) -> dict[str, npt.NDArray[np.int64]]:
    targets = {
        "pre_yield": 0.0,
        "onset": 0.5 * (thresholds.onset + thresholds.developed),
        "developed": 0.5 * (thresholds.developed + thresholds.late),
        "late": 1.0,
    }
    resolved = dict(bins)
    for name, indices in bins.items():
        if indices.size == 0:
            resolved[name] = np.asarray([int(np.argmin(np.abs(fraction - targets[name])))])
    return resolved
