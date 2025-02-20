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

    k1: float | None = 0.0
    k2: float | None = 0.0
    k3: float | None = 0.0
    p1: float | None = 0.0
    p2: float | None = 0.0
    c0: float | None = None
    c1: float | None = None

    object_distance: float | None = None
    fstop: float | None = None

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

        if self.c0 is None:
            self.c0 = self.pixels_num[0] / 2
        if self.c1 is None:
            self.c1 = self.pixels_num[1] / 2


@dataclass(slots=True)
class CameraData2D:
    #shape=(n_px_X,n_px_Y)
    num_pixels: np.ndarray

    # Center location of the region of interest in world coords
    #shape=(3,) as (x,y,z)
    roi_center_world: np.ndarray

    # Converts pixels to length units to align with global coords
    leng_per_px: float

    #shape=(n_time_steps,)
    sample_times: np.ndarray | None = None

    #TODO: this only works for flat surfaces aligned with the axis
    view_axes: tuple[int,int] = (0,1)

    bits_sensor: int = 16
    bits_file: int = 16

    angle: Rotation | None = None

    field_of_view_center_local: np.ndarray = field(init=False)
    field_of_view_local: np.ndarray = field(init=False)
    roi_shift_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.field_of_view_local = self.num_pixels*self.leng_per_px
        self.field_of_view_center_local = self.field_of_view_local/2

        self.roi_shift_world = np.zeros_like(self.roi_center_world)
        for ii,vv in enumerate(self.view_axes):
            self.roi_shift_world[vv] = self.roi_center_world[vv] - \
                self.field_of_view_center_local[ii]




