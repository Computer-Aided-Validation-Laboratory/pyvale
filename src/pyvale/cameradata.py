# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
NOTE: This module is a feature under developement.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(slots=True)
class CameraData:
    pixels_num: np.ndarray
    pixels_size: np.ndarray

    pos_world: np.ndarray
    rot_world: Rotation
    roi_cent_world: np.ndarray

    focal_length: float | None = 50.0
    sub_samp: int = 2

    back_face_removal: bool = True

    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    c0: float | None = None
    c1: float | None = None

    fstop: float | None = None

    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)


    def __post_init__(self) -> None:
        relative_pos = np.subtract(self.pos_world, self.roi_cent_world)
        self.image_dist = np.linalg.norm(relative_pos)
        self.sensor_size = self.pixels_num*self.pixels_size
        self.image_dims = (self.image_dist
                           *self.sensor_size/self.focal_length)

        self.cam_to_world_mat = np.zeros((4,4))
        self.cam_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1,-1] = 1.0
        self.cam_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)

        if self.c0 is None:
            self.c0 = self.pixels_num[0] / 2
        if self.c1 is None:
            self.c1 = self.pixels_num[1] / 2






