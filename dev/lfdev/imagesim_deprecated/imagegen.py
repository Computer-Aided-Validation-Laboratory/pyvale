"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from pyvale.imagesim.cameradataimagedef import CameraImageDef
import pyvale.imagesim.imagedef as sid

def gen_grid_image(camera: CameraImageDef,
                   px_per_period: int,
                   contrast_amp: float,
                   contrast_offset: float = 0.5) -> np.ndarray:

    (px_grid_x,px_grid_y) = sid.get_pixel_grid_in_px(camera)

    grid_image = (2*contrast_amp*camera.dyn_range)/4 \
                    *(1+np.cos(2*np.pi*px_grid_x/px_per_period)) \
                    *(1+np.cos(2*np.pi*px_grid_y/px_per_period)) \
                    +camera.dyn_range*(contrast_offset-contrast_amp)

    return grid_image
