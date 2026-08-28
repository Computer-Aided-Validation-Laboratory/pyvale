#================================================================================
#Example: thermocouples on a 2d plate
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================
"""
2D Incremental DIC
---------------------

This example walks through setting up an incremental DIC caluculation for the
simple case of rigid body motion of a plate. Incremental DIC works by updating
the reference image at an interval to <do something>. it is good in cases where
there is large deformation.""" 

# %%
# While incremental can usually withstand a greater level of subset warping, it is important to remember that
# by updating the reference images you are compounding errors from previous
# correlations. The incremental DIC process as implemented in pyvale is best
# highlighted in the image below
#
# .. image:: ../../../../_static/incremental_light.png
#    :alt: Incremental DIC
#    :width: 100%
#    :class: only-light
#
# .. image:: ../../../../_static/incremental_dark.png
#    :alt: Incremental DIC
#    :width: 100%
#    :class: only-dark
# 
# We start in the usual way by selecting the images and building the ROI:

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# pyvale modules
import pyvale.dataset as dataset
import pyvale.dic as dic

subset_size = 31
ref_img = dataset.dic_plate_rigid_cam0_ref()
def_img = dataset.dic_plate_rigid_cam0_def()

# create a directory for the the different outputs
output_path = Path.cwd() / "pyvale-output" / "ex07"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

roi = dic.RegionOfInterest(ref_img)
roi.rect_boundary(left=50,right=50,top=50,bottom=50)


# %%
# We can now proceed with the incremental DIC calculation. There are three key
# arguments to be aware of when enabling incremental DIC:
#
# - ``incremental`` (bool): Enables incremental DIC when set to True. Default is False.
# - ``incremental_update_condition`` (``str``): Specifies the condition under which the reference 
#     image is updated. Valid options are:
#
#     - ``"IMAGE"``: Update the reference image every N images, where N is given by
#         ``incremental_update_value``.
#     - ``"COST"``: Update the reference image when the mean ZNCC cost across all subsets
#         falls below the threshold specified by ``incremental_update_value``.
#     - ``"ITER"``: Update the reference image when the mean number of iterations exceeds
#         the value specified by ``incremental_update_value``.
# - ``incremental_update_value`` (``int`` or ``float``): The threshold or interval used alongside 
#   ``incremental_update_condition``.
#
# In this example we will proceed with the simple case of updating the reference
# image after every image correlation procedure. Note: While the displacements
# are reported as a cumulative value, the reported ZNCC value in the results is relative to the
# subset in the current updated reference image, NOT the original reference image.

dic.calculate_2d(reference=ref_img,
                 deformed=def_img,
                 roi_mask=roi.mask,
                 seed=[500,500],
                 subset_size=subset_size,
                 subset_step=10,
                 incremental=True,
                 incremental_update_condition="IMAGE", # can also be "COST" or "ITER"
                 incremental_update_value=1, # update the reference every 1 image(s)
                 output_basepath=output_path,
                 output_delimiter=",",
                 output_prefix="results_inc_")


