#================================================================================
#Example: thermocouples on a 2d plate
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================

"""
Simple Example of a 2d plate with a hole
---------------------------------------------

This example takes you through setting up A DIC and strain calculation for 1
"""

import matplotlib.pyplot as plt
import pyvale

# %%
# there'll be a couple of places where we'll be referring back to the reference
# image, deformed image and subset size, so we'll define them here to start. For
# series of deformed images, I'd recommend either having them in a seperate
# folder to the reference image. Or, if they follow a naming convention then you
# can use a the wildcard operator '*' to select multiple files.
subset_size = 31
ref_img = "../../data/ref0000.tiff"
def_img = "../../data/def000*.tiff"

# %% 
# We'll want to select our Region of Interset (ROI) using the interactive tool.
# Firstly we create an instance of the  ROI class using
# :class:`pyvale.dicregionofinterest.DICRegionOfInterest`. The reason we pass the reference image
# here is that this image will be used as an underlay for the ROI selection
# process.
roi = pyvale.DICRegionOfInterest(ref_img)
roi.interactive_selection(subset_size)

# %%
# Once you've closed the ROI interactive selection, this will generate a mask
# and seed location coordinates that can then be passed to the DIC engine. It
# might be the case at this stage you'd want to save the mask for any future DIC
# calculations with this set of images. For exceptionally large images, 
# it's recommended to save with binary=True to reduce file size and the time 
# it takes to save the array to disk.This can be done with:
roi.save(filename="roi.dat",binary=False)

# For any future DIC calculations, you can read the ROI mask back in using the
# :func:`roi.roiread` command. Remember to update the filename and the whether the ROI
# mask has been saved in human readable or binary format.
roi.read(filename="roi.dat")


# %%
# Now for the main event, the 2D DIC engine can be run using the command
# :func:`pyvale.dic2d.DIC2D`. There's a large number of arguments that can be passed to the
# DIC engine so please consult the in-depth documentation for further details.
# In all cases you'll need to specify your reference & deformed images, your ROI
# mask, and subset information. By default, the engine will use an affine shape
# function using a Zero Normalised Sum of Squared Differences (ZNSSD)
# correlation criterion. The results will be saved to disk, you
# can specify the filename, location, delimiter and format of the output data
# using the appropriate flags. Again, please see the in-depth documentation for
# further details.
pyvale.dic_2d(reference=ref_img,
              deformed=def_img,
              roi_mask=roi.mask,
              seed=roi.seed,
              subset_size=subset_size,
              subset_step=10,
              shape_function="AFFINE",
              correlation_criteria="ZNSSD")

# %%
# If you've saved the results in human readable format, then feel free to use
# whatever tool you'd like to perform and visualisation and further analysis. If
# you'd like to use python, we've written a handly tool that can import the data
# for further inspection. This can be done with the DICdata_import command. The
# results will be placed in a DICResults object. You can find more information
# about the structure of the dataclass here
# :class:`pyvale.DICResults`. If the results
# have been saved in binary format, or have a user specified delimiter, then
# they'll also need to be specified. There's also the option to specify the
# layout of the imported data. See :class:`pyvale.DICResults` for more details.
# You can also look at :func:`pyvale.DICdata_import` for more info
dicdata = pyvale.dic_data_import(data="dic_results_*.dat", delimiter=" ", binary=False)

# %%
# As an example of some very simple visualisation, you could loop over the
# number of deformed images and plot the displacement and cost values using the
# below. You'll need to make sure you have matplotlib.pyplot installed and imported.
fig, axes = plt.subplots(2, 3, figsize=(15, 5))
axes = axes.flatten()

# first deformation image
im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.u[0])
im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.v[0])
im3 = axes[2].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.cost[0])

# second deformation image
im4 = axes[3].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.u[1])
im5 = axes[4].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.v[1])
im6 = axes[5].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.cost[1])

# Titles
axes[0].set_title('u component (def0000.tiff)')
axes[1].set_title('v component (def0000.tiff)')
axes[2].set_title('cost (def0000.tiff)')
axes[3].set_title('u component (def0001.tiff)')
axes[4].set_title('v component (def0001.tiff)')
axes[5].set_title('cost (def0001.tiff)')


fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])
fig.colorbar(im4, ax=axes[3])
fig.colorbar(im5, ax=axes[4])
fig.colorbar(im6, ax=axes[5])

# layout
plt.show()


# %%
# .. image:: ../../../../_static/plate_with_hole.png
#    :alt: Displacement and cost values
#    :width: 600px
#    :align: center



