"""Load solver-independent assembled strain-data bundles for VFM preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pyvale.vfm.experimentdata import EdgeConditions
from pyvale.vfm.roi import VfmRegionOfInterest, convert_mask_to_physical_roi


@dataclass(slots=True, frozen=True)
class AssembledDataConfig:
    """Configuration for a directory containing standard assembled ``.npy`` arrays."""

    data_dir: Path
    thickness: float
    edge_conditions: EdgeConditions


@dataclass(slots=True, frozen=True)
class AssembledData:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    strain: npt.NDArray[np.float64]
    force: npt.NDArray[np.float64]
    time: npt.NDArray[np.float64]
    region_of_interest: VfmRegionOfInterest


def load_assembled_data(config: AssembledDataConfig) -> AssembledData:
    """Load and validate a generic assembled FE/DIC handoff directory.

    Required arrays are ``x.npy``, ``y.npy``, ``strain.npy``, ``force.npy``
    and ``time.npy``.  An optional ``specimen_mask.npy`` controls the physical
    ROI; otherwise finite strain values define it.
    """

    data_dir = Path(config.data_dir)
    names = ("x", "y", "strain", "force", "time")
    missing = [f"{name}.npy" for name in names if not (data_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"Assembled data directory is missing: {missing}")
    x, y, strain, force, time = (np.load(data_dir / f"{name}.npy") for name in names)
    x, y, strain, force, time = (
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
        np.asarray(strain, dtype=np.float64), np.asarray(force, dtype=np.float64), np.asarray(time, dtype=np.float64),
    )
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("x.npy and y.npy must be matching two-dimensional arrays.")
    if strain.ndim != 4 or strain.shape[1:] != (3, *x.shape):
        raise ValueError(f"strain.npy must have shape (time, 3, {x.shape[0]}, {x.shape[1]}), got {strain.shape}.")
    if force.shape != (strain.shape[0], 2):
        raise ValueError("force.npy must have shape (time, 2).")
    if time.shape != (strain.shape[0],) or not np.all(np.diff(time) >= 0.0):
        raise ValueError("time.npy must be a non-decreasing one-dimensional time history.")
    mask_path = data_dir / "specimen_mask.npy"
    mask = np.load(mask_path).astype(bool) if mask_path.is_file() else np.all(np.isfinite(strain), axis=(0, 1))
    if mask.shape != x.shape or not np.any(mask):
        raise ValueError("specimen_mask.npy must match x/y and contain at least one valid point.")
    if not np.all(np.isfinite(x[mask])) or not np.all(np.isfinite(y[mask])):
        raise ValueError("Coordinates must be finite inside the specimen mask.")
    strain = strain.copy()
    strain[:, :, ~mask] = np.nan
    roi = VfmRegionOfInterest.from_definition(convert_mask_to_physical_roi(mask, x, y, simplification_pixels=0.0))
    return AssembledData(x, y, strain, force, time, roi)
