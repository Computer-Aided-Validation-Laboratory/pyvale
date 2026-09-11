#================================================================================
# Example: stereo calibration
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================
"""
Stereo DIC rigid body motion of a plate
---------------------

This example demonstrates how to perform stereo DIC using pyvale. The example
uses synthetic images generated using `Riley <https://github.com/Computer-Aided-Validation-Laboratory/riley-raster>`_
from a known calibration and deformation field.
The calibration parameters are loaded from a text file, and the DIC calculation
is performed on the reference and deformed images.
"""

# pyvale modules
import numpy as np
from pathlib import Path
from dataclasses import fields

import pyvale.dic as dic
import pyvale.calib as calib
import pyvale.dataset as dataset

# %% 
# Load in the ground truth calibration parameters. These are the parameters
# that were used to generate the synthetic images:

calib_params = calib.loadtxt(dataset.dic_ex09_stereo_calibration(), delimiter=",")


# %%
# The dataset module provides access to the synthetic images used in this
# example. These functions return Path objects pointing to TIFF files bundled
# with pyvale. The reference helpers return a concrete image path, while the
# deformed-image helpers return a wildcard path pattern. For example,
# ``def0`` resolves to a path ending in ``rigid_cam0_frame*.tiff``.
#
# During ``dic.calculate_3d`` pyvale expands each deformed-image wildcard with
# ``glob`` and sorts the matching filenames. The sorted camera 0 and camera 1
# lists are then processed as synchronized stereo frame pairs, so the image
# naming must leave both cameras with the same number of frames in the same
# order. To avoid wildcard discovery, pass an explicit ``list[Path]`` for each
# camera instead.

# reference images
ref0 = dataset.dic_plate_rigid_cam0_ref()
ref1 = dataset.dic_plate_rigid_cam1_ref()

# deformed images
def0 = dataset.dic_plate_rigid_cam0_def()
def1 = dataset.dic_plate_rigid_cam1_def()

# Deformed images can be supplied either as wildcard Path objects, as above,
# or as explicit lists of image paths, for example:
# def0 = [Path("cam0_frame00.tiff"), Path("cam0_frame01.tiff"), ...]
# def1 = [Path("cam1_frame00.tiff"), Path("cam1_frame01.tiff"), ...]


# Build ROI using cam 0 reference image
roi = dic.RegionOfInterest(ref0)
roi.rect_boundary(50,50,50,50)

# create an output directory 
output_path = Path.cwd() / "pyvale-output" / "dic_ex09"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

# %% 
# To perform the stereo DIC calculation, pass the two camera references as
# ``[cam0_reference, cam1_reference]`` and the two deformed-image inputs as
# ``[cam0_deformed, cam1_deformed]``. In this example the deformed inputs are
# wildcard ``Path`` objects, so pyvale discovers all matching frames for each
# camera before starting the calculation. The calibration parameters describe
# the relationship between the two camera views. The remaining arguments are
# the same as ``dic.calculate_2d``.

dic.calculate_3d(reference=[ref0, ref1],
                 deformed=[def0, def1],
                 calibration=calib_params,
                 roi_mask=roi.mask,
                 seed=[500,500],
                 subset_size=31,
                 subset_step=10,
                 max_displacement=100,
                 output_basepath=output_path)

# %%
# can now import the results using dic.import_3d
dic_files = output_path / "dic_results_*.csv"
dic_results = dic.import_3d(data=dic_files, delimiter=",", binary=False)

# %%
# The stereo DIC results are stored in a nested dataclass. The fields can be
# seen using the below commands:

print("")
print("DIC results fields:")
for field in fields(dic_results):
    print(f"{field.name}")

print("")
print("Stereo DIC results fields:")
for field in fields(dic_results.stereo):
    print(f"{field.name}")

# %%
# plot a 3d reconstruction using pyvista

import pyvista as pv

stereo = dic_results.stereo
frame = 12


points = np.column_stack((
    np.ravel(stereo.x_mm[frame]),
    np.ravel(stereo.y_mm[frame]),
    np.ravel(stereo.z_mm[frame]),
))

cloud = pv.PolyData(points)
cloud['elevation'] = points[:, 1]
cloud.plot(eye_dome_lighting=True)







