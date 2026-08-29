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

    def indices(self, name: str) -> tuple[int, ...]:
        if name not in {"pre_yield", "onset", "developed", "late"}:
            raise KeyError(f"Unknown load regime {name!r}.")
        return getattr(self, name)

    def diagnostics(self) -> dict[str, object]:
        result = asdict(self)
        for name in ("pre_yield", "onset", "developed", "late"):
            result[name] = list(result[name])
        result["yielded_fraction"] = list(result["yielded_fraction"])
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
