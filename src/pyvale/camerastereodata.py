"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.cameradata import CameraData

@dataclass(slots=True)
class CameraStereoData:
    cam_data_0: CameraData
    cam_data_1: CameraData

    stereo_rotation: Rotation = field(init=False)
    stereo_dist: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
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


