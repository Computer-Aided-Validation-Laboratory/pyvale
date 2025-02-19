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


# NOTE: This module is a feature under developement.
#
# - Camera Local Coords: Pixel positions in pixels/meters
# - Global Sim Coords: Transform from local pixel positions to sim coords in meters
# - For this transformation we need user to specify center of ROI in sim coords
# - There are going to be different ways to specify the camera properties

# For thin lens theory will need to know some combination of:
#   - The focal length of the lense
#   - The working distance

# Will need to create different ways for the user to automatically position the
# camera


@dataclass(slots=True)
class CameraData:
    pixels_num: np.ndarray
    pixels_size: np.ndarray

    pos_world: np.ndarray
    rot_world: Rotation
    roi_cent_world: np.ndarray

    focal_length: float = 50.0
    sub_samp: int = 2

    back_face_removal: bool = True

    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.image_dist = np.linalg.norm(self.pos_world - self.roi_cent_world)
        self.sensor_size = self.pixels_num*self.pixels_size
        self.image_dims = (self.image_dist
                           *self.sensor_size/self.focal_length)

        self.cam_to_world_mat = np.zeros((4,4))
        self.cam_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1,-1] = 1.0
        self.cam_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)






