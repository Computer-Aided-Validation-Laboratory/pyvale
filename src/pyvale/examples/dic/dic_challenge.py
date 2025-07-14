# ================================================================================
# Example: DIC Challenge 2.0 Comparison
# 
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ================================================================================

"""
Comparison to the 2.0 2D DIC Challenge
---------------------------------------------

This example takes you through setting up A DIC and strain calculation for 1
"""

import pyvale
import matplotlib.pyplot as plt

ref_pattern = "../../data/DIC_Challenge_Star_Noise_Ref.tif"
def_pattern = "../../data/DIC_Challenge_Star_Noise_Def.tif"
subset_size = 17
subset_radius = subset_size // 2


# %%
# we need to select our region of interest. For this example, we are only
# interested in the subsets along mid point. 
# We can use :func:`roi.rect_boundary` to exclude a large border region so we
# only correlate along the horizontal at the midpoint
roi = pyvale.DICRegionOfInterest(ref_pattern)
roi.rect_boundary(50,50,250-subset_radius,250-subset_radius) # left, right, top, bottom
roi.show_image()


# %%
# .. image:: ../../../../_static/dic_challenge_roi.png
#    :alt: Selected roi
#    :width: 800px
#    :align: center 


# %% 
# To perform the correlation we need to select a seed point. Ideally, this is
# somewhere in a region of a small displacement. Here we'll select it to be
# [3500,250] which is close to the right hand boundary of the image along the
# midpoint where the spatial frequency is lower. The results will be saved in
# the current working directory with a filename prefix of subset_size_19_*.txt
# If you are feeling adventorous you could investigate the effect of varying the
# subset size by placing this script within some kind of loop.
pyvale.dic_2d(reference=ref_pattern,
              deformed=def_pattern,
              roi_mask=roi.mask,
              subset_size=subset_size,
              subset_step=1,
              seed=[3500,250],
              max_displacement=10)


# %% 
# We can import the results in the standard way
dicdata = pyvale.dic_data_import(data="dic_results_DIC_Challenge*",
                                 layout='column', binary=False, delimiter=" ")


plt.figure()
plt.xlabel("subset x location [px]")
plt.ylabel("Displacement [px]")
plt.grid(True)
plt.axhline(y=0.5, color='red', linestyle='--', linewidth=4)
plt.plot(dicdata.ss_x, dicdata.v[0,:])
plt.tight_layout()
plt.show()

# %%
# .. image:: ../../../../_static/dic_challenge_plot.png
#    :alt: DIC Comparison with ground truth
#    :width: 600px
#    :align: center
