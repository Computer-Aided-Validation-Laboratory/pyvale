# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Stereo camera descriptions and calibration-file helpers."""

from pathlib import Path
from typing import Self

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .camera import Camera


class CameraStereo:
    """A pair of cameras with their relative stereo calibration.

    Parameters
    ----------
    cam_data_0 : Camera
        First camera in the stereo system.
    cam_data_1 : Camera
        Second camera in the stereo system.
    """

    __slots__ = ("cam_data_0", "cam_data_1", "stereo_dist", "stereo_rotation")

    def __init__(self, cam_data_0: Camera, cam_data_1: Camera) -> None:
        self.cam_data_0 = cam_data_0
        self.cam_data_1 = cam_data_1

        cam0_rotation = Rotation.as_matrix(cam_data_0.rot_world)
        cam1_rotation = Rotation.as_matrix(cam_data_1.rot_world)
        self.stereo_rotation = Rotation.align_vectors(cam0_rotation, cam1_rotation)[0]

        distance = cam_data_0.pos_world - cam_data_1.pos_world
        distance_rotated = cam_data_0.rot_world.apply(distance)
        inverse = self.stereo_rotation.inv().as_quat()
        inverse[3] *= -1.0
        self.stereo_dist = Rotation.from_quat(inverse).apply(distance_rotated)

    @classmethod
    def from_calibration(
        cls,
        calib_path: Path,
        pos_world_0: np.ndarray,
        rot_world_0: Rotation,
        focal_length: float,
    ) -> Self:
        """Create a stereo system from a pyvale YAML calibration file.

        Parameters
        ----------
        calib_path : pathlib.Path
            Calibration YAML file created by :meth:`save_calibration`.
        pos_world_0 : numpy.ndarray
            Camera-zero position in world coordinates.
        rot_world_0 : scipy.spatial.transform.Rotation
            Camera-zero world rotation.
        focal_length : float
            Camera-zero focal length in millimetres.

        Returns
        -------
        CameraStereo
            Reconstructed stereo camera system.
        """
        parameters = yaml.safe_load(calib_path.read_text())
        pixels_num_0 = np.array((
            int(parameters["Cam0_Cx [pixels]"] * 2),
            int(parameters["Cam0_Cy [pixels]"] * 2),
        ))
        pixels_num_1 = np.array((
            int(parameters["Cam1_Cx [pixels]"] * 2),
            int(parameters["Cam1_Cy [pixels]"] * 2),
        ))
        
        pixel_size = focal_length / parameters["Cam0_Fx [pixels]"]
        stereo_rotation = Rotation.from_euler(
            "xyz",
            (parameters["Theta [deg]"], parameters["Phi [deg]"],
             parameters["Psi [deg]"]),
            degrees=True,
        )
        stereo_dist = np.array((parameters["Tx [mm]"], parameters["Ty [mm]"],
                                parameters["Tz [mm]"]))

        rot_world_1 = stereo_rotation * rot_world_0
        inverse = stereo_rotation.inv().as_quat()
        inverse[3] *= -1.0
        distance_rotated = Rotation.from_quat(inverse).inv().apply(stereo_dist)
        distance = rot_world_0.inv().apply(distance_rotated)
        pos_world_1 = pos_world_0 - distance

        camera_0 = Camera(pixels_num_0, np.full(2, pixel_size), pos_world_0,
                          rot_world_0, np.zeros(3), focal_length)
        camera_1 = Camera(pixels_num_1, np.full(2, pixel_size), pos_world_1,
                          rot_world_1, np.zeros(3), focal_length)

        return cls(camera_0, camera_1)

    def save_calibration(self, base_dir: Path) -> None:
        """Save the system in pyvale's YAML stereo-calibration format.

        Parameters
        ----------
        base_dir : pathlib.Path
            Existing output directory. The calibration is saved beneath its
            ``calibration`` subdirectory.

        Raises
        ------
        ValueError
            If ``base_dir`` is not an existing directory.
        """
        if not base_dir.is_dir():
            raise ValueError("The specified save directory does not exist")

        rotation = self.stereo_rotation.as_euler("xyz", degrees=True)
        parameters = {
            "Cam0_Fx [pixels]": float(self.cam_data_0.focal_length /
                                       self.cam_data_0.pixels_size[0]),
            "Cam0_Fy [pixels]": float(self.cam_data_0.focal_length /
                                       self.cam_data_0.pixels_size[1]),
            "Cam0_Fs [pixels]": 0,
            "Cam0_Kappa 1": self.cam_data_0.distortion_k1,
            "Cam0_Kappa 2": self.cam_data_0.distortion_k2,
            "Cam0_Kappa 3": self.cam_data_0.distortion_k3,
            "Cam0_P1": self.cam_data_0.distortion_p1,
            "Cam0_P2": self.cam_data_0.distortion_p2,
            "Cam0_Cx [pixels]": float(self.cam_data_0.c0),
            "Cam0_Cy [pixels]": float(self.cam_data_0.c1),
            "Cam1_Fx [pixels]": float(self.cam_data_1.focal_length /
                                       self.cam_data_1.pixels_size[0]),
            "Cam1_Fy [pixels]": float(self.cam_data_1.focal_length /
                                       self.cam_data_1.pixels_size[1]),
            "Cam1_Fs [pixels]": 0,
            "Cam1_Kappa 1": self.cam_data_1.distortion_k1,
            "Cam1_Kappa 2": self.cam_data_1.distortion_k2,
            "Cam1_Kappa 3": self.cam_data_1.distortion_k3,
            "Cam1_P1": self.cam_data_1.distortion_p1,
            "Cam1_P2": self.cam_data_1.distortion_p2,
            "Cam1_Cx [pixels]": float(self.cam_data_1.c0),
            "Cam1_Cy [pixels]": float(self.cam_data_1.c1),
            "Tx [mm]": float(self.stereo_dist[0]),
            "Ty [mm]": float(self.stereo_dist[1]),
            "Tz [mm]": float(self.stereo_dist[2]),
            "Theta [deg]": float(rotation[0]),
            "Phi [deg]": float(rotation[1]),
            "Psi [deg]": float(rotation[2]),
        }
        output_dir = base_dir / "calibration"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "calibration.yaml").write_text(yaml.safe_dump(parameters))

    def save_calibration_mid(self, base_dir: Path) -> None:
        """Save the system in the MatchID ``.caldat`` calibration format.

        Parameters
        ----------
        base_dir : pathlib.Path
            Existing output directory. The calibration is written beneath its
            ``calibration`` subdirectory.

        Raises
        ------
        ValueError
            If ``base_dir`` is not an existing directory.
        """
        if not base_dir.is_dir():
            raise ValueError("The specified save directory does not exist")

        rotation = self.stereo_rotation.as_euler("xyz", degrees=True)
        camera_0 = self.cam_data_0
        camera_1 = self.cam_data_1
        lines = (
            f"Cam0_Fx [pixels]; {camera_0.focal_length / camera_0.pixels_size[0]}",
            f"Cam0_Fy [pixels]; {camera_0.focal_length / camera_0.pixels_size[1]}",
            "Cam0_Fs [pixels];0",
            f"Cam0_Kappa 1;{camera_0.distortion_k1}",
            f"Cam0_Kappa 2;{camera_0.distortion_k2}",
            f"Cam0_Kappa 3;{camera_0.distortion_k3}",
            f"Cam0_P1;{camera_0.distortion_p1}",
            f"Cam0_P2;{camera_0.distortion_p2}",
            f"Cam0_Cx [pixels];{camera_0.c0}",
            f"Cam0_Cy [pixels];{camera_0.c1}",
            f"Cam1_Fx [pixels]; {camera_1.focal_length / camera_1.pixels_size[0]}",
            f"Cam1_Fy [pixels]; {camera_1.focal_length / camera_1.pixels_size[1]}",
            "Cam1_Fs [pixels];0",
            f"Cam1_Kappa 1;{camera_1.distortion_k1}",
            f"Cam1_Kappa 2;{camera_1.distortion_k2}",
            f"Cam1_Kappa 3;{camera_1.distortion_k3}",
            f"Cam1_P1;{camera_1.distortion_p1}",
            f"Cam1_P2;{camera_1.distortion_p2}",
            f"Cam1_Cx [pixels];{camera_1.c0}",
            f"Cam1_Cy [pixels];{camera_1.c1}",
            f"Tx [mm];{self.stereo_dist[0]}",
            f"Ty [mm];{self.stereo_dist[1]}",
            f"Tz [mm];{self.stereo_dist[2]}",
            f"Theta [deg];{rotation[0]}",
            f"Phi [deg];{rotation[1]}",
            f"Psi [deg];{rotation[2]}",
        )
        output_dir = base_dir / "calibration"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "calibration.caldat").write_text("\n".join(lines))


__all__ = ["CameraStereo"]
