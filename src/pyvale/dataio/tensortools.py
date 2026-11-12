# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Symmetric tensor assembly and small-strain invariants in Cartesian axes."""

from enum import Enum, auto

import numpy as np


class EStrainShear(Enum):
    TENSOR = auto()
    ENGINEERING = auto()


def assemble_strain_tensor(
    normal: np.ndarray,
    shear: np.ndarray,
    *,
    shear_convention: EStrainShear,
) -> np.ndarray:
    """Assemble (..., 3, 3) from normal (..., xx/yy/zz) and shear (..., xy/xz/yz).

    Engineering shear gamma_ij equals 2 epsilon_ij. Missing out-of-plane strain
    must be supplied explicitly; no plane-stress or plane-strain assumption is made.
    """
    if not isinstance(shear_convention, EStrainShear):
        raise TypeError("Select an EStrainShear convention.")
    normal = np.asarray(normal, dtype=np.float64)
    shear = np.asarray(shear, dtype=np.float64)
    if normal.ndim < 1 or normal.shape[-1] != 3 or normal.shape != shear.shape:
        raise ValueError("Normal and shear components must share shape (..., 3).")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(shear)):
        raise ValueError("Tensor components must be finite.")
    tensor = np.empty((*normal.shape[:-1], 3, 3), dtype=np.float64)
    for index in range(3):
        tensor[..., index, index] = normal[..., index]
    factor = 0.5 if shear_convention is EStrainShear.ENGINEERING else 1.0
    for component, (row, column) in enumerate(((0, 1), (0, 2), (1, 2))):
        tensor[..., row, column] = factor * shear[..., component]
        tensor[..., column, row] = tensor[..., row, column]
    return tensor


def _validate_strain_tensor(tensor: np.ndarray) -> np.ndarray:
    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError("Strain tensors must have shape (..., 3, 3).")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("Strain tensors must be finite.")
    if not np.array_equal(tensor, np.swapaxes(tensor, -1, -2)):
        raise ValueError("Strain tensors must be symmetric.")
    return tensor


def calc_principal_strains(tensor: np.ndarray) -> np.ndarray:
    """Return (..., 3) principal strains in descending order (first = maximum)."""
    tensor = _validate_strain_tensor(tensor)
    return np.linalg.eigvalsh(tensor)[..., ::-1]


def calc_equivalent_strain(tensor: np.ndarray) -> np.ndarray:
    """Return sqrt(2/3 epsilon_dev:epsilon_dev), the deviatoric equivalent strain.

    This small-strain measure excludes volumetric strain. It is not the
    stress-style sqrt(3/2 s:s) expression or accumulated equivalent plastic strain.
    """
    tensor = _validate_strain_tensor(tensor)
    mean_strain = np.trace(tensor, axis1=-2, axis2=-1) / 3.0
    deviatoric = tensor - mean_strain[..., None, None] * np.eye(3)
    return np.sqrt((2.0 / 3.0) * np.sum(deviatoric * deviatoric, axis=(-2, -1)))
