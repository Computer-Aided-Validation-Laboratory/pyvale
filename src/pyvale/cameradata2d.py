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
class CameraData2D:
    pixels_count: np.ndarray | None = None
    leng_per_px: float = 1.0e-3
    bits: int = 8
    roi_cent_world: np.ndarray | None = None

    background: float = 0.5
    sample_times: np.ndarray | None = None
    angle: Rotation | None = None

    subsample: int = 2

    field_of_view: np.ndarray = field(init=False)
    dynamic_range: int = field(init=False)

    world_to_cam: np.ndarray = field(init=False)
    cam_to_world: np.ndarray = field(init=False)

    def __post_init__(self) -> None:

        if self.pixels_count is None:
            self.pixels_count = np.array((1000,1000),dtype=np.int32)

        if self.roi_cent_world is None:
            self.roi_cent_world = np.array((0.0,0.0,0.0),dtype=np.float64)

        self.field_of_view = self.leng_per_px*(self.pixels_count.astype(np.float64))
        self.dynamic_range = 2**self.bits
        self.background = self.background*float(self.dynamic_range)

        self.world_to_cam = self.field_of_view/2 - self.roi_cent_world[:-1]
        self.cam_to_world = -self.world_to_cam


#@dataclass(slots=True)
#class CameraData2D:
#     #shape=(n_px_X,n_px_Y)
#     num_pixels: np.ndarray

#     # Center location of the region of interest in world coords
#     #shape=(3,) as (x,y,z)
#     roi_center_world: np.ndarray

#     # Converts pixels to length units to align with global coords
#     leng_per_px: float

#     #shape=(n_time_steps,)
#     sample_times: np.ndarray | None = None

#     #TODO: this only works for flat surfaces aligned with the axis
#     view_axes: tuple[int,int] = (0,1)

#     bits_sensor: int = 16
#     bits_file: int = 16

#     angle: Rotation | None = None

#     field_of_view_center_local: np.ndarray = field(init=False)
#     field_of_view_local: np.ndarray = field(init=False)
#     roi_shift_world: np.ndarray = field(init=False)

#     def __post_init__(self) -> None:
#         self.field_of_view_local = self.num_pixels*self.leng_per_px
#         self.field_of_view_center_local = self.field_of_view_local/2

#         self.roi_shift_world = np.zeros_like(self.roi_center_world)
#         for ii,vv in enumerate(self.view_axes):
#             self.roi_shift_world[vv] = self.roi_center_world[vv] - \
#                 self.field_of_view_center_local[ii]