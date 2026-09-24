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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyvale import dic
from pyvale import render


output_path = Path.cwd() / "pyvale-output" / "dic_ex06"
output_path.mkdir(parents=True, exist_ok=True)

# %%
# Riley generates the example image sequence here. For a custom render, see the
# Riley examples in the Render 3D section (:ref:`examples_render3d`) and the
# implementation of :func:`pyvale.render.create_example_images_rigid`.

images = render.create_example_images_rigid()
ref_img = images.cam0_reference
def_img = images.cam0_deformed

# image files can be passed directly instead:
# ref_img = Path("/path/to/reference.tiff")
# def_img = Path("/path/to/deformed*.tiff")

# %%
# Create the ROI and exclude a rectangular border of 50 pixels on all sides.

roi = dic.RegionOfInterest(ref_img)
roi.rect_boundary(left=50, right=50, top=50, bottom=50)


# %%
# We can now proceed with the incremental DIC calculation. There are two key
# arguments to be aware of when enabling incremental DIC:
#
# - ``incremental_update`` (``str``): Specifies the condition under which the reference
#     image is updated. Use ``"OFF"`` to disable incremental DIC. Valid update options are:
#
#     - ``"IMAGE"``: Update the reference image every N images, where N is given by
#         ``incremental_update_value``.
#     - ``"COST"``: Update the reference image when the mean ZNCC cost across all subsets
#         falls below the threshold specified by ``incremental_update_value``.
#     - ``"ITER"``: Update the reference image when the mean number of iterations exceeds
#         the value specified by ``incremental_update_value``.
# - ``incremental_update_value`` (``int`` or ``float``): The threshold or interval used alongside 
#   ``incremental_update``.
#
# In this example we will proceed with the simple case of updating the reference
# image after every image correlation procedure. Note: While the displacements
# are reported as a cumulative value, the reported ZNCC value in the results is relative to the
# subset in the current updated reference image, NOT the original reference image.

dic.calculate_2d(reference=ref_img,
                 deformed=def_img,
                 roi_mask=roi.mask,
                 seed=[100,100],
                 subset_size=31,
                 subset_step=10,
                 incremental_update="IMAGE", # use "OFF" to disable; can also be "COST" or "ITER"
                 incremental_update_value=1, # update the reference every 1 image(s)
                 output_basepath=output_path,
                 output_delimiter=",",
                 output_prefix="results_inc_",
                 print_level=1)
