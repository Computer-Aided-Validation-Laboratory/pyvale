# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class CamIntrinsics:
    """Camera intrinsic parameters."""

    fx: float
    """Focal length in x direction [pixels]"""

    fy: float
    """Focal length in y direction [pixels]"""

    fs: float
    """Skew coefficient [pixels]"""

    cx: float
    """Principal point x-coordinate [pixels]"""

    cy: float
    """Principal point y-coordinate [pixels]"""

    distortion: np.ndarray
    """Distortion coefficients [kappa1, kappa2, p1, p2, kappa3]"""
    
    @property
    def camera_matrix(self) -> np.ndarray:
        """Returns the 3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx, self.fs, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass(slots=True)
class Calib:
    """Stereo camera calibration parameters."""

    cam0: CamIntrinsics
    """ Camera 0 intrinsic parameters"""

    cam1: CamIntrinsics
    """ Camera 1 intrinsic parameters"""

    translation: np.ndarray
    """Translation vector [x, y, z] in mm"""

    rotation: np.ndarray
    """Euler angles [theta, phi, psi] in degrees"""
