"""Developing stereo camera system
"""
from dataclasses import dataclass
import numpy as np
import math
import mathutils
from pathlib import Path
from dev_camerablender import CameraData
from scipy.spatial.transform import Rotation

@dataclass
class StereoData:
    cam_data_0: CameraData = CameraData
    cam_data_1: CameraData = CameraData
    base: float | None = 5
    angle_deg: float | None = 15
    calib_file: bool | None = False
    calib_filepath: str | None = None

class Stereo:
    def __init__(self, stereo_data: StereoData):
        self.stereo_data = stereo_data


    def add_stereo_system(self, scene):
        self.stereo_data.cam_data_0.name = 'cam_0'
        cam_0 = scene.add_camera(self.stereo_data.cam_data_0)

        cam_1_pos = list(self.stereo_data.cam_data_0.position)
        cam_1_pos[0] = cam_1_pos[0] + self.stereo_data.base
        self.stereo_data.cam_data_1.position = cam_1_pos

        self.stereo_data.cam_data_1.name = 'cam_1'
        cam_1 = scene.add_camera(self.stereo_data.cam_data_1)
        cam_1.rotation_mode = 'XYZ'
        cam_1.rotation_euler = (0, np.radians(self.stereo_data.angle_deg), 0)


        if self.stereo_data.calib_file:
            calib_filepath = self.stereo_data.calib_filepath
            self.stereo_calibration(cam_0, cam_1, calib_filepath)


    def stereo_calibration(self, cam0, cam1, calib_filepath):
        # Get rotation of cam0 to cam1
        cam0_orient = cam0.rotation_euler
        cam0_orient = Rotation.from_euler('XYZ', cam0_orient)
        cam1_orient = cam1.rotation_euler
        cam1_orient = Rotation.from_euler('XYZ', cam1_orient)

        cam0_quat = Rotation.as_quat(cam0_orient)
        cam1_quat = Rotation.as_quat(cam1_orient)

        q_x = [math.cos(np.pi / 2), math.sin(np.pi / 2), 0, 0]
        cam0_quat = self._rotate_quaternion(q_x, cam0_quat)
        cam1_quat = self._rotate_quaternion(q_x, cam1_quat)

        q_rot = self._quaternion_multiply(cam0_quat,
                                          self._quaternion_conjugate(cam1_quat))
        q_rot_conj = self._quaternion_conjugate(q_rot)
        q_rot = Rotation.from_quat(q_rot)
        ang = Rotation.as_euler(q_rot, 'XYZ', degrees=True)

        # Translation of cam0 to cam1 + rotate vector to orientation of cam1
        dist = cam0.location - cam1.location
        dist[2] *= -1
        dist[1] *= -1

        dist_rot = self._rotate_vec(self._rotate_vec(dist, Rotation.as_quat(cam0_orient)), q_rot_conj)

        if Path(calib_filepath).is_dir() is False:
            Path.mkdir(calib_filepath)
        calib_filepath = calib_filepath / 'calib.caldat'
        with open(calib_filepath, "w") as file:
            file.write("Cam0_Fx [pixels];" + f'{cam0.data.lens/cam0["px_size"][0]}\n')
            file.write("Cam0_Fy [pixels];" + f'{cam0.data.lens/cam0["px_size"][1]}\n')
            file.write("Cam0_Fs [pixels];0\n")
            file.write(f'Cam0_Kappa 1;{cam0["k1"]}\n')
            file.write(f'Cam0_Kappa 2;{cam0["k2"]}\n')
            file.write(f'Cam0_Kappa 3;{cam0["k3"]}\n')
            file.write(f'Cam0_P1;{cam0["p1"]}\n')
            file.write(f'Cam0_P2;{cam0["p2"]}\n')
            file.write(f'Cam0_Cx [pixels];{cam0["c0"]}\n')
            file.write(f'Cam0_Cy [pixels];{cam0["c1"]}\n')
            file.write("Cam1_Fx [pixels];" + f'{cam1.data.lens/cam1["px_size"][0]}\n')
            file.write("Cam1_Fy [pixels];" + f'{cam1.data.lens/cam1["px_size"][1]}\n')
            file.write("Cam1_Fs [pixels];0\n")
            file.write(f'Cam1_Kappa 1;{cam1["k1"]}\n')
            file.write(f'Cam1_Kappa 2;{cam1["k2"]}\n')
            file.write(f'Cam1_Kappa 3;{cam1["k3"]}\n')
            file.write(f'Cam1_P1;{cam1["p1"]}\n')
            file.write(f'Cam1_P2;{cam1["p2"]}\n')
            file.write(f'Cam1_Cx [pixels];{cam1["c0"]}\n')
            file.write(f'Cam1_Cy [pixels];{cam1["c1"]}\n')
            file.write(f"Tx [mm];{dist_rot[0]}\n")
            file.write(f"Ty [mm];{dist_rot[1]}\n")
            file.write(f"Tz [mm];{dist_rot[2]}\n")
            file.write(f"Theta [deg];{ang[0]}\n")
            file.write(f"Phi [deg];{ang[1]}\n")
            file.write(f"Psi [deg];{ang[2]}")

    def _rotate_quaternion(self, q0 ,q1):
        # q2 = q1.q0.conj(q0)

        q0_conj = self._quaternion_conjugate(q0)
        q2 = self._quaternion_multiply(self._quaternion_multiply(q0, q1), q0_conj)
        return q2

    def _quaternion_conjugate(self, q0):
        if type(q0) is list or type(q0) is mathutils.Quaternion:
            q0_conj = [-j if i > 0 else j for i, j in enumerate(q0)]
        elif type(q0) is np.ndarray:
            q0_conj = np.array([-j if i > 0 else j for i, j in enumerate(q0)])
        return q0_conj

    def _quaternion_multiply(self, q0, q1):
        w0, x0, y0, z0 = q0
        w1, x1, y1, z1 = q1
        return np.array(
            [
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            ],
            dtype=np.float64,
        )

    def _rotate_vec(self, v, q):
        # v = q.v.conj(q)
        v = self._vec_to_quaternion(v)
        q_conj = self._quaternion_conjugate(q)
        v = self._quaternion_multiply(self._quaternion_multiply(q, v), q_conj)
        v = v[1:]
        return v

    def _vec_to_quaternion(self, v):
        if np.linalg.norm(v) == 0:
            w = 1.0
        else:
            w = 0.0
        if type(v) is list:
            v.insert(0, w)
            np.array(v)
        elif type(v) is np.ndarray:
            v = np.insert(v, 0, w)
        elif type(v) is mathutils.Vector:
            v = list(v)
            v.insert(0, w)
            np.array(v)
        return v






