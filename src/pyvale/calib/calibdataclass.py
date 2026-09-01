# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

"""Dataclasses used to pass stereo calibration parameters through Pyvale."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class CamIntrinsics:
    """Intrinsic camera parameters and distortion coefficients.

    The parameters follow the usual pinhole camera model with optional skew and
    five OpenCV-style distortion coefficients. Focal lengths and the principal
    point are expressed in pixels.
    """

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
        """Return the 3x3 intrinsic camera matrix.

        Returns
        -------
        np.ndarray
            Matrix ``K`` with focal lengths, skew, and principal point arranged
            as ``[[fx, fs, cx], [0, fy, cy], [0, 0, 1]]``.
        """
        return np.array([
            [self.fx, self.fs, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass(slots=True)
class Calib:
    """Stereo camera calibration parameters.

    Attributes
    ----------
    cam0, cam1 : CamIntrinsics
        Intrinsic calibration for camera 0 and camera 1.
    translation : np.ndarray
        Translation vector from camera 0 to camera 1 in millimetres.
    rotation : np.ndarray
        Euler rotation angles from camera 0 to camera 1 in degrees, ordered as
        ``[theta, phi, psi]``.
    """

    cam0: CamIntrinsics
    """ Camera 0 intrinsic parameters"""

    cam1: CamIntrinsics
    """ Camera 1 intrinsic parameters"""

    translation: np.ndarray
    """Translation vector [x, y, z] in mm"""

    rotation: np.ndarray
    """Euler angles [theta, phi, psi] in degrees"""


def savetxt(calib: Calib, path: str | Path, delimiter: str = ",") -> None:
    """Save stereo calibration parameters to a two-column CSV file.

    Parameters
    ----------
    calib : Calib
        Stereo calibration parameters to save.
    path : str or pathlib.Path
        Output CSV file path.
    delimiter : str, optional
        Delimiter to use between CSV fields. Defaults to a comma.
    """

    path = Path(path)
    distortion_names = ("k1", "k2", "p1", "p2", "k3")

    rows: list[tuple[str, float]] = []
    for cam_name, cam in (("cam0", calib.cam0), ("cam1", calib.cam1)):
        rows.extend([
            (f"{cam_name}_fx_px", float(cam.fx)),
            (f"{cam_name}_fy_px", float(cam.fy)),
            (f"{cam_name}_fs_px", float(cam.fs)),
            (f"{cam_name}_cx_px", float(cam.cx)),
            (f"{cam_name}_cy_px", float(cam.cy)),
        ])
        rows.extend(
            (f"{cam_name}_{name}", float(value))
            for name, value in zip(distortion_names, cam.distortion)
        )

    rows.extend([
        ("tx_mm", float(calib.translation[0])),
        ("ty_mm", float(calib.translation[1])),
        ("tz_mm", float(calib.translation[2])),
        ("theta_deg", float(calib.rotation[0])),
        ("phi_deg", float(calib.rotation[1])),
        ("psi_deg", float(calib.rotation[2])),
    ])

    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerows(rows)

def loadtxt(path: str | Path, delimiter: str = ",") -> Calib:
    """Load stereo calibration parameters from a two-column CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Input CSV file path.
    delimiter : str, optional
        Delimiter used between CSV fields. Defaults to a comma.

    Returns
    -------
    Calib
        Stereo calibration parameters read from disk.
    """
    path = Path(path)
    with path.open("r", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        rows = [row for row in reader if row]

    values = {name: float(value) for name, value, *extra in rows}

    def get(name: str) -> float:
        try:
            return values[name]
        except KeyError as exc:
            raise ValueError(f"Missing calibration parameter: {name}") from exc

    def cam_from_values(cam_name: str) -> CamIntrinsics:
        return CamIntrinsics(
            fx=get(f"{cam_name}_fx_px"),
            fy=get(f"{cam_name}_fy_px"),
            fs=get(f"{cam_name}_fs_px"),
            cx=get(f"{cam_name}_cx_px"),
            cy=get(f"{cam_name}_cy_px"),
            distortion=np.array([
                get(f"{cam_name}_k1"),
                get(f"{cam_name}_k2"),
                get(f"{cam_name}_p1"),
                get(f"{cam_name}_p2"),
                get(f"{cam_name}_k3"),
            ], dtype=np.float64),
        )

    return Calib(
        cam0=cam_from_values("cam0"),
        cam1=cam_from_values("cam1"),
        translation=np.array([get("tx_mm"), get("ty_mm"), get("tz_mm")], dtype=np.float64),
        rotation=np.array([get("theta_deg"), get("phi_deg"), get("psi_deg")], dtype=np.float64),
    )

