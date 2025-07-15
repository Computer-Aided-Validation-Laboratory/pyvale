#================================================================================
#Example: thermocouples on a 2d plate
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================

"""
Region of Interest (ROI) Selection
---------------------------------------------

this is a test

"""
import pyvale

# %% 
# We'll want to select our Region of Interset (ROI) using the interactive tool.
# Firstly we create an instance of the  ROI class. The reason we pass the reference image
# here is that this image will be used as an underlay for the ROI selection
# process.
roi = pyvale.DICRegionOfInterest(ref_image="../../data/plate_hole_ref0000.tiff")
roi.interactive_selection(subset_size=31)
exit(0)

# %%
# .. image:: ../../../../_static/roi_tool.gif
#    :alt: ROI selection GUI (animated)
#    :width: 600px
#    :align: center

# %%
# Once you've closed the ROI interactive selection, this will generate a mask
# and seed location coordinates that can then be passed to the DIC engine. It
# might be the case at this stage you'd want to save the mask for any future DIC
# calculations with this set of images. For exceptionally large images, 
# it's recommended to save with binary=True to reduce file size and the time 
# it takes to save the array to disk.This can be done with:
roi.save_array(filename="roi.dat",binary=False)

# %%
# For any future DIC calculations, you can read the ROI mask back in using the
# :func:`roi.roiread` command. Remember to update the filename and the whether the ROI
# mask has been saved in human readable or binary format.
roi.read_array(filename="roi.dat",binary=False)

# %%
# if you are reading in a previously saved ROI. Then you can view it
# overlayed over your reference image to check before proceeding with any
# correlation. This can be done with the command:
roi.show_image()

# %%
# At present, there are a couple of other ways of selecting a region of
# interest programatically. Firstly, there's the option to remove a boundary
# from the ROI, leaving a central region. This can be done with:
roi.reset_mask()
roi.rect_boundary(left=50,right=50,bottom=50,top=50)
roi.save_image("rect_boundary.tiff")

# %%
# and this will exclude the 50 pixels along all edges from the correlation. You
# can also select a specific region using the command:
roi.reset_mask()
roi.rect_region(x=200,y=200,size_x=200,size_y=200)
roi.save_image("rect_region.tiff")

# %%
# .. list-table::
#    :widths: 50 50
#    :align: center
#    :header-rows: 0
#
#    * - .. figure:: ../../../../_static/rect_boundary.png
#          :width: 300px
#          :align: center
#
#          ``roi.rect_boundary(left=50, right=50, bottom=50, top=50)``
#
#      - .. figure:: ../../../../_static/rect_region.png
#          :width: 300px
#          :align: center
#
#          ``roi.rect_region(x=200, y=200, size_x=200, size_y=200)``

# %%
# this will create a ROI starting at pixel coordinates (200,200) with dimensions
# (50,100).
#
# Of course you could manually edit the ROI however you'd like. I'd suggest
# creating a initial ROI using `roi.rect_boundary(0,0,0,0)`. You can then
# manipulate `roi.mask` as you would with any other 2D numpy array. 



