"""Load assembled MatchID DIC archives into the VFM input-data format."""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import EdgeConditions
from pyvale.vfm.roi import (
    RoiDefinition,
    RoiShape,
    VfmRegionOfInterest,
    convert_mask_to_physical_roi,
)


@dataclass(slots=True)
class MatchIDAssembledConfig:
    """Inputs for an assembled MatchID DIC archive and its force history."""

    assembled_file: Path
    """HDF5 file containing MatchID strain fields and reference coordinates."""

    force_history_file: Path
    """CSV force history containing ``time_s`` and ``force_N`` columns."""

    thickness: float
    """Out-of-plane specimen thickness (mm)."""

    edge_conditions: EdgeConditions
    """Boundary conditions applied to the four specimen edges."""

    force_direction: Literal["x", "y"] = "y"
    """Global direction in which the measured force is applied."""


@dataclass(slots=True, frozen=True)
class MatchIDAssembledData:
    """Normalised arrays and ROI loaded from a MatchID assembled archive."""

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    region_of_interest: VfmRegionOfInterest


def load_matchid_assembled_data(
    config: MatchIDAssembledConfig,
) -> MatchIDAssembledData:
    """Load a sampled assembled MatchID archive.

    The assembled archive stores physical coordinates from camera calibration,
    which can contain small non-axis-aligned distortions. VFM currently
    requires an axis-aligned uniform grid, so the coordinate axes are fitted
    to a regular grid while retaining their physical extent. The ROI is the
    intersection of points valid in every selected frame, preventing missing
    DIC values from entering the analysis domain.
    """

    # Read the spatial grid, strain history, and MatchID validity mask.
    with h5py.File(config.assembled_file, "r") as archive:
        x_raw = _read_dataset(archive, "x_ref")
        y_raw = _read_dataset(archive, "y_ref")
        strain_components = [
            _read_dataset(archive, name)
            for name in ("exx", "eyy", "exy")
        ]
        measurement_valid_mask = _read_dataset(
            archive,
            "measurement_valid_mask",
        ).astype(bool, copy=False)

    # Ensure every field describes the same DIC grid and set of frames.
    _validate_archive_shapes(
        x_raw,
        y_raw,
        strain_components,
        measurement_valid_mask,
    )

    # VFM stores strain as (timesteps, components, y, x), ordered xx, yy, xy.
    strain = np.stack(strain_components, axis=1).astype(np.float64, copy=False)

    # Only retain points measured in every selected frame and with finite data.
    common_valid_mask = np.all(measurement_valid_mask, axis=0)
    common_valid_mask &= np.isfinite(x_raw) & np.isfinite(y_raw)
    common_valid_mask &= np.all(np.isfinite(strain), axis=(0, 1))
    if not np.any(common_valid_mask):
        raise ValueError("MatchID archive has no points valid in every frame.")

    # Convert the slightly distorted calibrated coordinates to VFM's regular grid.
    x_axis = _fit_uniform_axis(x_raw, axis=0, name="x")
    y_axis = _fit_uniform_axis(y_raw, axis=1, name="y")
    x, y = np.meshgrid(x_axis, y_axis)

    # Keep invalid values explicit so downstream code never treats them as
    # physical measurements; the ROI below excludes the same points.
    strain[:, :, ~common_valid_mask] = np.nan
    region_of_interest = VfmRegionOfInterest.from_definition(
        convert_mask_to_physical_roi(
            common_valid_mask,
            x,
            y,
            simplification_pixels=0.0,
        )
    )

    # Align the force history with the DIC frames and place it in [Fx, Fy].
    force, time = _load_force_history(config)
    if force.shape[0] != strain.shape[0]:
        raise ValueError(
            "force-history row count does not match MatchID archive frame count: "
            f"{force.shape[0]} != {strain.shape[0]}."
        )

    return MatchIDAssembledData(
        x=x,
        y=y,
        strain=strain,
        force=force,
        time=time,
        region_of_interest=region_of_interest,
    )


def _read_dataset(
    archive: h5py.File,
    name: str,
) -> npt.NDArray[np.float64]:
    """Read one required dataset from an assembled MatchID archive."""

    if name not in archive:
        raise ValueError(
            f"MatchID assembled archive '{archive.filename}' is missing dataset '{name}'."
        )
    return np.asarray(archive[name])


def _validate_archive_shapes(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strain_components: list[npt.NDArray[np.float64]],
    measurement_valid_mask: npt.NDArray[np.bool_],
) -> None:
    """Check that archive arrays use compatible grid and frame dimensions."""

    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x_ref and y_ref must be matching two-dimensional grids.")
    for name, component in zip(("exx", "eyy", "exy"), strain_components):
        if component.ndim != 3 or component.shape[1:] != x.shape:
            raise ValueError(
                f"{name} must have shape (timesteps, {x.shape[0]}, {x.shape[1]}), "
                f"got {component.shape}."
            )
    if measurement_valid_mask.shape != strain_components[0].shape:
        raise ValueError(
            "measurement_valid_mask must match the strain component shape, "
            f"got {measurement_valid_mask.shape} and {strain_components[0].shape}."
        )


def _fit_uniform_axis(
    coordinates: npt.NDArray[np.float64],
    *,
    axis: int,
    name: str,
) -> npt.NDArray[np.float64]:
    """Fit one regular physical coordinate axis from the calibrated grid."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        coordinate_axis = np.nanmedian(coordinates, axis=axis)
    finite = np.isfinite(coordinate_axis)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"Could not determine a physical {name}-axis from MatchID coordinates.")

    indices = np.arange(coordinate_axis.size, dtype=np.float64)
    slope, intercept = np.polyfit(indices[finite], coordinate_axis[finite], deg=1)
    if slope <= 0.0:
        raise ValueError(f"MatchID {name}-axis must increase with its grid index.")
    return (intercept + slope * indices).astype(np.float64, copy=False)


def _load_force_history(
    config: MatchIDAssembledConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Load force/time columns and return VFM's force and time arrays."""

    force_history = np.atleast_1d(np.genfromtxt(
        config.force_history_file,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    ))
    required_columns = {"time_s", "force_N"}
    names = set(force_history.dtype.names or ())
    missing_columns = sorted(required_columns - names)
    if missing_columns:
        raise ValueError(
            f"Force history '{config.force_history_file}' is missing columns {missing_columns}."
        )

    time = np.asarray(force_history["time_s"], dtype=np.float64)
    time -= time[0]
    force = np.zeros((time.size, 2), dtype=np.float64)
    force[:, 0 if config.force_direction == "x" else 1] = np.asarray(
        force_history["force_N"],
        dtype=np.float64,
    )
    return force, time
