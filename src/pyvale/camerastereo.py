# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

from typing import Self
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from pyvale.cameradata import CameraData
from pyvale.pyvaleexceptions import BlenderError


class CameraStereo:
    __slots__ = ("cam_data_0","cam_data_1","stereo_dist","stereo_rotation")

    def __init__(self, cam_data_0: CameraData, cam_data_1: CameraData) -> None:
        self.cam_data_0 = cam_data_0
        self.cam_data_1 = cam_data_1

        cam0_rot_matrix = Rotation.as_matrix(self.cam_data_0.rot_world)
        cam1_rot_matrix = Rotation.as_matrix(self.cam_data_1.rot_world)
        (self.stereo_rotation, _) = Rotation.align_vectors(cam0_rot_matrix,
                                                           cam1_rot_matrix)
        dist = self.cam_data_0.pos_world - self.cam_data_1.pos_world
        dist_rot = self.cam_data_0.rot_world.apply(dist)
        inverse = self.stereo_rotation.inv().as_quat()
        inverse[3] *= -1
        inverse = Rotation.from_quat(inverse)
        self.stereo_dist = inverse.apply(dist_rot)

    @classmethod
    def from_calibration(cls,
                         calib_path: Path,
                         pos_world_0: np.ndarray,
                         rot_world_0: Rotation,
                         focal_length: float) -> Self:
        """A method to initialise the CameraStereo using a calibration file and
        some additional parameters. This creates an instance of the CameraStereo
        class from the calibration parameters.

        Parameters
        ----------
        calib_path : Path
            The path to the calibration file (in yaml format).
        pos_world_0 : np.ndarray
            The position of camera 0 in world coordinates.
        rot_world_0 : Rotation
            The rotation of camera 0 in world coordinates.
        focal_length : float
            The focal length of camera 0.

        Returns
        -------
        Self
            An instance of the CameraStereo class, given the specified parameters.
        """
        calib_params = yaml.safe_load(calib_path.read_text())
        pixels_num_cam0 = np.array([calib_params['Cam0_Cx [pixels]']*2,
                           calib_params['Cam0_Cy [pixels]']*2])
        pixels_num_cam1 = np.array([calib_params['Cam1_Cx [pixels]']*2,
                           calib_params['Cam1_Cy [pixels]']*2])
        pixels_size = focal_length / calib_params["Cam0_Fx [pixels]"]
        stereo_rotation = Rotation.from_euler("xyz", ([calib_params['Theta [deg]'],
                                    calib_params['Phi [deg]'],
                                    calib_params['Psi [deg]']]), degrees=True)
        stereo_dist = np.array([calib_params["Tx [mm]"],
                                calib_params["Ty [mm]"],
                                calib_params["Tz [mm]"]])

        rot_world_1 = stereo_rotation * rot_world_0

        inverse = stereo_rotation.inv().as_quat()
        inverse[3] *= -1
        inverse = Rotation.from_quat(inverse)

        dist_rot = inverse.inv().apply(stereo_dist)
        dist = rot_world_0.inv().apply(dist_rot)
        pos_world_1 = pos_world_0 - dist

        cam_data_0 = CameraData(pixels_num=pixels_num_cam0,
                                pixels_size=np.array([pixels_size, pixels_size]),
                                pos_world=pos_world_0,
                                rot_world=rot_world_0,
                                roi_cent_world=np.array([0, 0, 0]),
                                focal_length=focal_length)
        cam_data_1 = CameraData(pixels_num=pixels_num_cam1,
                                pixels_size=np.array([pixels_size, pixels_size]),
                                pos_world=pos_world_1,
                                rot_world=rot_world_1,
                                roi_cent_world=np.array([0, 0, 0]),
                                focal_length=focal_length)
        camera_stereo = cls(cam_data_0, cam_data_1)

        return camera_stereo

    def save_calibration(self, base_dir: Path) -> None:
        """A method to save a calibration file of the stereo system as a yaml.
        This is so that the file can easily be read into python, but is also
        user-readable.

        Parameters
        ----------
        base_dir : Path
            The base directory to which all files should be saved. The
            calibration file will be saved in a sub-directory named "calibration"
            within this directory.

        Raises
        ------
        BlenderError
            "The specified save directory does not exist"
        """
        stereo_rotation = self.stereo_rotation.as_euler("xyz", degrees=True)
        calib_params = {
            "Cam0_Fx [pixels]": float(self.cam_data_0.focal_length /
                                 self.cam_data_0.pixels_size[0]),
            "Cam0_Fy [pixels]": float(self.cam_data_0.focal_length /
                                 self.cam_data_0.pixels_size[1]),
            "Cam0_Fs [pixels]": 0,
            "Cam0_Kappa 1": self.cam_data_0.k1,
            "Cam0_Kappa 2": self.cam_data_0.k2,
            "Cam0_Kappa 3": self.cam_data_0.k3,
            "Cam0_P1": self.cam_data_0.p1,
            "Cam0_P2": self.cam_data_0.p2,
            "Cam0_Cx [pixels]": float(self.cam_data_0.c0),
            "Cam0_Cy [pixels]": float(self.cam_data_0.c1),
            "Cam1_Fx [pixels]": float(self.cam_data_1.focal_length /
                                 self.cam_data_1.pixels_size[0]),
            "Cam1_Fy [pixels]": float(self.cam_data_1.focal_length /
                                 self.cam_data_1.pixels_size[1]),
            "Cam1_Fs [pixels]": 0,
            "Cam1_Kappa 1": self.cam_data_1.k1,
            "Cam1_Kappa 2": self.cam_data_1.k2,
            "Cam1_Kappa 3": self.cam_data_1.k3,
            "Cam1_P1": self.cam_data_1.p1,
            "Cam1_P2": self.cam_data_1.p2,
            "Cam1_Cx [pixels]": float(self.cam_data_1.c0),
            "Cam1_Cy [pixels]": float(self.cam_data_1.c1),
            "Tx [mm]": float(self.stereo_dist[0]),
            "Ty [mm]": float(self.stereo_dist[1]),
            "Tz [mm]": float(self.stereo_dist[2]),
            "Theta [deg]": float(stereo_rotation[0]),
            "Phi [deg]": float(stereo_rotation[1]),
            "Psi [deg]": float(stereo_rotation[2])
        }
        if not base_dir.is_dir():
            raise BlenderError("The specified save directory does not exist")

        save_dir = base_dir / "calibration"
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)

        filepath = str(save_dir / "calibration.yaml")
        calib_file = open(filepath, "w")
        yaml.safe_dump(calib_params, calib_file)
        calib_file.close()
        print("Calibration file saved to:", (save_dir / "calibration.yaml"))

    def save_calibration_mid(self, base_dir: Path) -> None:
        """A method to save a calibration file of the stereo system in a MatchID
        accepted format.

        Parameters
        ----------
        base_dir : Path
            The base directory to which all files should be saved. The
            calibration file will be saved in a sub-directory named "calibration"
            within this directory.

        Raises
        ------
        BlenderError
            "The specified save directory does not exist"
        """
        if not base_dir.is_dir():
            raise BlenderError("The specified save directory does not exist")

        save_dir = base_dir / "calibration"
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)

        filepath = str(save_dir / "calibration.caldat")
        with open(filepath, "w") as file:
            file.write(f'Cam0_Fx [pixels]; {self.cam_data_0.focal_length/ self.cam_data_0.pixels_size[0]}\n')
            file.write(f'Cam0_Fy [pixels]; {self.cam_data_0.focal_length/ self.cam_data_0.pixels_size[1]}\n')
            file.write("Cam0_Fs [pixels];0\n")
            file.write(f'Cam0_Kappa 1;{self.cam_data_0.k1}\n')
            file.write(f'Cam0_Kappa 2;{self.cam_data_0.k2}\n')
            file.write(f'Cam0_Kappa 3;{self.cam_data_0.k3}\n')
            file.write(f'Cam0_P1;{self.cam_data_0.p1}\n')
            file.write(f'Cam0_P2;{self.cam_data_0.p2}\n')
            file.write(f'Cam0_Cx [pixels];{self.cam_data_0.c0}\n')
            file.write(f'Cam0_Cy [pixels];{self.cam_data_0.c1}\n')
            file.write(f'Cam1_Fx [pixels]; {self.cam_data_1.focal_length/ self.cam_data_1.pixels_size[0]}\n')
            file.write(f'Cam1_Fy [pixels]; {self.cam_data_1.focal_length/ self.cam_data_1.pixels_size[1]}\n')
            file.write("Cam1_Fs [pixels];0\n")
            file.write(f'Cam1_Kappa 1;{self.cam_data_1.k1}\n')
            file.write(f'Cam1_Kappa 2;{self.cam_data_1.k2}\n')
            file.write(f'Cam1_Kappa 3;{self.cam_data_1.k3}\n')
            file.write(f'Cam1_P1;{self.cam_data_1.p1}\n')
            file.write(f'Cam1_P2;{self.cam_data_1.p2}\n')
            file.write(f'Cam1_Cx [pixels];{self.cam_data_1.c0}\n')
            file.write(f'Cam1_Cy [pixels];{self.cam_data_1.c1}\n')
            file.write(f"Tx [mm];{self.stereo_dist[0]}\n")
            file.write(f"Ty [mm];{self.stereo_dist[1]}\n")
            file.write(f"Tz [mm];{self.stereo_dist[2]}\n")
            stereo_rotation = self.stereo_rotation.as_euler("xyz", degrees=True)
            file.write(f"Theta [deg];{stereo_rotation[0]}\n")
            file.write(f"Phi [deg];{stereo_rotation[1]}\n")
            file.write(f"Psi [deg];{stereo_rotation[2]}")
