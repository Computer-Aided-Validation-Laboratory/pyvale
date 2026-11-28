# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Site selectors for discrete sensor placement."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from pyvale.sensopt.designspace import IConstrainInp
from pyvale.sensorsim.sensordata import SensorData


@dataclass(slots=True, kw_only=True)
class SensSiteBounds:
    """Inclusive XYZ intervals in the design space's coordinate units."""

    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None


@dataclass(slots=True, kw_only=True)
class SensSiteGridIdxs:
    """Zero-based indices along the original grid axes. None selects an axis in full."""

    x: range | tuple[int, ...] | None = None
    y: range | tuple[int, ...] | None = None
    z: range | tuple[int, ...] | None = None


@dataclass(slots=True, kw_only=True)
class SensSiteAll:
    """Select the intersection of one or more site selectors."""

    selectors: tuple[SensSiteSelector, ...]


@dataclass(slots=True, kw_only=True)
class SensSiteAny:
    """Select the union of one or more site selectors."""

    selectors: tuple[SensSiteSelector, ...]


@dataclass(slots=True, kw_only=True)
class SensSiteNot:
    """Select candidate sites outside the child selector."""

    selector: SensSiteSelector


SensSiteSelector: TypeAlias = (
    SensSiteBounds
    | SensSiteGridIdxs
    | npt.NDArray[np.float64]
    | SensSiteAll
    | SensSiteAny
    | SensSiteNot
)

_SiteMask = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.bool_]]


class ConstrainSensSites(IConstrainInp):
    """Allow selected sites that match the default or their array's override.

    Grid selectors use the original grid of shape (nx, ny, nz, 3). Grid indices
    and explicit site rows are matched by exact coordinates. Selectors are
    compiled at construction, before candidate filtering or evaluation.
    """

    def __init__(
        self,
        *,
        default: SensSiteSelector | None = None,
        by_array: Mapping[str, SensSiteSelector] | None = None,
        grid_xyz: npt.NDArray[np.float64] | None = None,
    ) -> None:
        grid = grid_xyz
        if grid is not None and (
            grid.ndim != 4
            or grid.shape[-1] != 3
            or grid.size == 0
            or not np.all(np.isfinite(grid))
        ):
            raise ValueError("grid_xyz must be finite with shape (nx, ny, nz, 3).")
        self._default_mask = (
            None if default is None else _compile_site_mask(default, grid)
        )
        self._array_masks: dict[str, _SiteMask] = {}
        if by_array is not None:
            self._array_masks = {
                key: _compile_site_mask(selector, grid)
                for key, selector in by_array.items()
            }

    def eval(self, sensor_data: Mapping[str, SensorData]) -> npt.NDArray[np.float64]:
        residuals: list[npt.NDArray[np.float64]] = []
        for key, data in sensor_data.items():
            mask = self._array_masks.get(key, self._default_mask)
            if mask is None:
                continue
            positions = data.positions
            if positions is None:
                raise ValueError(f"Sensor array {key!r} requires positions.")
            residuals.append((~mask(positions)).astype(np.float64))
        if not residuals:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(residuals)


def _compile_site_mask(
    selector: SensSiteSelector, grid: npt.NDArray[np.float64] | None
) -> _SiteMask:
    if isinstance(selector, SensSiteBounds):
        bounds = tuple(
            (axis, interval)
            for axis, interval in enumerate((selector.x, selector.y, selector.z))
            if interval is not None
        )
        for _, (lower, upper) in bounds:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                raise ValueError("Site bounds must be finite and ordered.")

        def match_bounds(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
            matches = np.ones(len(positions), dtype=np.bool_)
            for axis, (lower, upper) in bounds:
                matches &= (positions[:, axis] >= lower) & (positions[:, axis] <= upper)
            return matches

        return match_bounds

    if isinstance(selector, SensSiteGridIdxs):
        if grid is None:
            raise ValueError("grid_xyz is required for SensSiteGridIdxs.")
        grid_indices: list[npt.NDArray[np.intp]] = []
        axes = (selector.x, selector.y, selector.z)
        for indices, size in zip(axes, grid.shape[:3], strict=True):
            if indices is None:
                grid_indices.append(np.arange(size, dtype=np.intp))
                continue
            if any(
                not isinstance(index, Integral)
                or isinstance(index, bool)
                or not 0 <= index < size
                for index in indices
            ):
                raise ValueError("Site grid indices must be integers within the grid.")
            grid_indices.append(np.asarray(indices, dtype=np.intp))
        selector = grid[np.ix_(*grid_indices)].reshape(-1, 3)

    if isinstance(selector, np.ndarray):
        if (
            selector.ndim != 2
            or selector.shape[1] != 3
            or not np.all(np.isfinite(selector))
        ):
            raise ValueError("Allowed sites must be finite with shape (site, 3).")
        rows = frozenset(tuple(position) for position in selector)

        def match_rows(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
            return np.fromiter(
                (tuple(position) in rows for position in positions),
                dtype=np.bool_,
                count=len(positions),
            )

        return match_rows

    if isinstance(selector, SensSiteNot):
        child = _compile_site_mask(selector.selector, grid)

        def negate(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
            return ~child(positions)

        return negate

    if not isinstance(selector, (SensSiteAll, SensSiteAny)):
        raise TypeError("Unsupported site selector.")
    if not selector.selectors:
        raise ValueError("SensSiteAll and SensSiteAny require at least one selector.")
    masks = tuple(_compile_site_mask(child, grid) for child in selector.selectors)
    match_all = isinstance(selector, SensSiteAll)

    def combine(positions: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        matches = masks[0](positions)
        for mask in masks[1:]:
            if match_all:
                matches &= mask(positions)
            else:
                matches |= mask(positions)
        return matches

    return combine
