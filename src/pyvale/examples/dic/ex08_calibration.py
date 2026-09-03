#================================================================================
# Example: stereo calibration
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================
"""
Stereo Calibration + DIC
---------------------

This example detects a calibration-dot target in synchronised stereo images
and uses the detected correspondences to calibrate the two cameras.

**The required calibration images are not distributed with the package to keep things lightweight. 
** 
`here <https://github.com/Computer-Aided-Validation-Laboratory/calibration-example-data>`_.

**Make sure to download/clone the repository and unzip the calibration images.
For this example we'll assume you've cloned the images to your current working
directory.**

Currently, the DIC module in pyvale only supports the below type of calibration targets (If there's
a specific type of calibration target you would like adding then please get in touch 
and we'll see what we can do).
"""

# %%
# .. image:: ../../../../images/cal_target_flipped.png
#    :alt: Calibration Target
#    :width: 100%
#    :align: center


# pyvale modules
import numpy as np
from pathlib import Path

import pyvale.dic as dic
import pyvale.calib as calib

# %%
# Calibration has two stages: the dot detection, then a bundle adjustment 
# to find the camera parameters that minimise
# the reprojection error across every image pair. you might need to ammend the
# below path depending on where you have saved the calibration images. The below
# path assumes you have saved the calibration images in a folder called
# "calibration-data" in your current working directory.

cam0 = "./calibration-data/cam0_*.bmp"
cam1 = "./calibration-data/cam1_*.bmp"

# %%
# Dot detection needs the image pairs, the target dimensions and spacing, and
# the three deliberately missing circles. Coordinates use the target grid with
# ``(0, 0)`` at the bottom-left dot; the missing dots can be specified in any order.

dots0, dots1, grid, filenames0, filenames1 = calib.detect_dots(
    cam0=cam0,
    cam1=cam1,
    grid_height=9,
    grid_width=12,
    missing_dots=[(9, 6), (2, 2), (2, 6)], # order of white dots does not matter
    min_dot_fraction=0.5, # minimum fraction of matching dots.
    grid_spacing=1.25,  # in mm.
)

# %%
# ``dots0`` and ``dots1`` are lists with one ``(N, 2)`` array per accepted image
# pair. Each row is a dot position in pixel coordinates. ``grid`` is the matching list of
# ``(N, 3)`` target coordinates in millimetres; it is the common object-point
# reference frame used by both cameras. The filename lists contain the accepted 
# filenames from the dot detection.
#
# For a high volume of calibration images that have large dimensions. It might be worth
# saving the detected points so the detection doesn't need to be rerun everything you need 
# to reperform a stereo calibration.

# make the output directory
output_dir = Path.cwd() / "pyvale-output" / "dic_ex08"
output_dir.mkdir(parents=True, exist_ok=True)

# save the detected points to plain text files
for points0, points1, filename0, filename1 in zip(dots0, dots1, filenames0, filenames1):
    np.savetxt(output_dir / f"dots_{Path(filename0).stem}.txt", points0)
    np.savetxt(output_dir / f"dots_{Path(filename1).stem}.txt", points1)

# %%
# The initial estimates are computed with Zhang's calibration method. More details 
# of the approach, which we use as an initial estimate, can be found in the 
# paper titled "A Flexible New Technique for Camera Calibration". The parameters from
# this method are then further refined with stereo bundle adjustment. You can toggle
# distortion parameters in the calibration using the ``optimize_distortion=`` function
# argument. The argument ``img_dims`` is ``[width, height]`` in
# pixels and must match the input images.

calib_params, reproj_err0, reproj_err1 = calib.calibrate_stereo(
    dots_cam0=dots0,
    dots_cam1=dots1,
    grid=grid,
    optimize_distortion=False, # all distortion parameters set to 0.
    img_dims=[2464, 2056],
    error_formulation="RMSE"
)

calib.savetxt(calib=calib_params, path=output_dir / "stereo_calibration.txt", delimiter=",")

# %%
# ``calib_params`` contains the intrinsics of each camera (focal lengths, skew,
# principal point and distortion), plus the translation in millimetres and
# rotation in degrees from camera 0 to camera 1. ``reproj_err0`` and
# ``reproj_err1`` contain one root-mean-square reprojection error, in pixels,
# for each accepted image pair. Large values identify target poses that should
# be checked for poor dot detection or unsuitable images.

print("")
print("Cam 0 Intrinsic Parameters:")
print(calib_params.cam0)
print("")
print("Cam 1 Intrinsic Parameters:")
print(calib_params.cam1)
print("")
print("Translation from cam 0 to cam 1 (mm):")
print(calib_params.translation)
print("")
print("Rotation from cam 0 to cam 1 (degrees):")
print(calib_params.rotation)
print("")
print("Reprojection Errors for cam 0 (pixels):")
print(reproj_err0)
print("")
print("Reprojection Errors for cam 1 (pixels):")
print(reproj_err1)









