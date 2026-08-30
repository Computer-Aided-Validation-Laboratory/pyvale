# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Camera data for three dimensional and planar rendering."""

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from scipy.spatial.transform import Rotation


class EDistortionModel(IntEnum):
    """Lens distortion models matching Riley's C backend.

    Values correspond to the ``distortion_model`` field in Riley's CCameraInput.
    """

    NONE = 0
    BROWN_CONRADY = 1
    BROWN_CONRADY_EXT = 2
    POLYNOMIAL = 3
    BROWN_CONRADY_POLYNOMIAL = 4
    BROWN_CONRADY_EXT_POLYNOMIAL = 5


class EPSFType(IntEnum):
    """Point spread function types matching Riley's PsfType enum.

    Values correspond to Riley's PsfType enum.
    """

    PIXEL_BOX = 0
    GAUSSIAN = 1
    ANISOTROPIC_GAUSSIAN = 2


@dataclass(slots=True, kw_only=True)
class Camera:
    """A perspective camera data structure.

    Parameters
    ----------
    pixels_num : np.ndarray
        Image pixel counts with shape ``(2,)`` and dtype ``int32`` in
        ``(width, height)`` order.
    pixels_size : np.ndarray
        Physical pixel dimensions with shape ``(2,)`` and dtype ``float64``
        in ``(width, height)`` order, in world length units.
    pos_world : np.ndarray
        Camera position in world coordinates with shape ``(3,)`` and dtype
        ``float64`` representing (X, Y, Z) coordinates.
    rot_world : scipy.spatial.transform.Rotation
        Camera orientation in world coordinates.
    roi_cent_world : np.ndarray
        Region of interest centre in world coordinates with shape ``(3,)``
        and dtype ``float64`` representing (X, Y, Z) coordinates.
    focal_length : float
        Camera focal length in the same units as world coordinates.
    subsample : int, optional
        Number of sub pixel samples per pixel direction.
    distortion_model : EDistortionModel, optional
        Lens distortion model identifier.
    distortion_k1, distortion_k2, distortion_k3, distortion_k4 : float, optional
        Radial distortion coefficients.
    distortion_k5, distortion_k6 : float, optional
        Additional radial distortion coefficients.
    distortion_p1, distortion_p2 : float, optional
        Tangential distortion coefficients.
    c0, c1 : float or None, optional
        Optical centre coordinates in pixels. ``None`` centres the coordinate
        in the image.
    fstop : float or None, optional
        Lens f number. ``None`` leaves depth of field disabled.
    psf_type : EPSFType, optional
        Point spread function model identifier.
    psf_sigma_x, psf_sigma_y : float, optional
        Point spread function standard deviations in pixel coordinates.
    psf_theta : float, optional
        Point spread function rotation in radians.
    psf_support_rad : float, optional
        Point spread function support radius in pixels.
    """

    pixels_num: np.ndarray
    pixels_size: np.ndarray
    pos_world: np.ndarray
    rot_world: Rotation
    roi_cent_world: np.ndarray
    focal_length: float
    subsample: int = 1
    distortion_model: EDistortionModel = EDistortionModel.NONE
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    distortion_k4: float = 0.0
    distortion_k5: float = 0.0
    distortion_k6: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0
    c0: float | None = None
    c1: float | None = None
    fstop: float | None = None
    psf_type: EPSFType = EPSFType.PIXEL_BOX
    psf_sigma_x: float = 0.0
    psf_sigma_y: float = 0.0
    psf_theta: float = 0.0
    psf_support_rad: float = 0.0

    def __post_init__(self) -> None:
        """Normalise arrays and supply an unset optical centre."""
        self.pixels_num = np.asarray(self.pixels_num, dtype=np.int32)
        self.pixels_size = np.asarray(self.pixels_size, dtype=np.float64)
        self.pos_world = np.asarray(self.pos_world, dtype=np.float64)
        self.roi_cent_world = np.asarray(self.roi_cent_world, dtype=np.float64)

        if self.c0 is None:
            self.c0 = float(self.pixels_num[0]) / 2.0

        if self.c1 is None:
            self.c1 = float(self.pixels_num[1]) / 2.0

        if isinstance(self.distortion_model, int):
            self.distortion_model = EDistortionModel(self.distortion_model)

        if isinstance(self.psf_type, int):
            self.psf_type = EPSFType(self.psf_type)


__all__ = ["Camera", "EDistortionModel", "EPSFType"]
