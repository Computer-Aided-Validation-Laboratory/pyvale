"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class ImageDefOpts:
    """ _summary_
    """
    #----------------------------------------------------------------------
    # USER CONTROLLED OPTIONS
    # Set these to achieve desired behaviour

    # Path to save the deformed images
    save_path: Path = Path.cwd() / 'deformed_images'
    save_tag: str = 'defimage'

    # Use this if starting with a full speckle or grid to create an
    # an artificial image with just the specimen geom
    mask_input_image: bool = True

    # Use this to crop the image to reduce computational  time, useful if
    # there are large borders around the specimen
    crop_on: bool = False
    crop_px: np.ndarray | None = None # only used to crop input image if above is true

    # Used to calculate ideal resolution using FE data and leaving a
    # a specified number of pixels around the border of the image
    calc_res_from_fe: bool =  False
    calc_res_border_px: int = 5

    # Option to append the input image to the list or to add a zero disp
    # frame at the start of the displacement data, useful to create static
    # image with masking
    add_static_ref: str = 'off'

    #----------------------------------------------------------------------
    # IMAGE AND DISPLACEMENT INTERPOLATION OPTIONS
    # Leave these as defaults unless advanced options are required. The
    # Following setup achieves a 1/1000th pixel accuracy using a rigid body
    # motion test analysed with the grid method and DIC. Computation time
    # is ~15 seconds per 1Mpx image on an AMD Ryzen 7, 4700U, 8 core CPU.

    # Interpolation type for using griddata to interpolate: scipy-griddata
    fe_interp: str = 'linear'
    fe_rescale: bool = True
    fe_extrap_outside_fov: bool = True # forces displacements outside the
    # specimen area to be padded with border values

    # Subsampling used to split each pixel in the input image
    subsample: int = 3

    # Interpolation used to upsample the input image: scipy-interp2d
    # image_upsamp_interp: str = 'cubic'

    # Order of interpolant used to deform the image: scipy.ndimage.map_coords
    image_def_order: int = 3
    image_def_extrap: str = 'nearest'
    image_def_extval: float = 0.0 # only used if above is 'constant'

    # Used to deal with holes and notches - if the specimen is just a
    # rectangle this can be set to false. Allows for an image mask which is
    # also deformed and applied to the deformed image
    def_complex_geom: bool = True

    #----------------------------------------------------------------------
    # PARALLELISATION OPTIONS


