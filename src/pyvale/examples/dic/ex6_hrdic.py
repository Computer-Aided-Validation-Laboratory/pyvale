#================================================================================
#Example: thermocouples on a 2d plate
#
#pyvale: the python validation engine
#License: MIT
#Copyright (C) 2024 The Computer Aided Validation Team
#================================================================================
"""
HRDIC
---------------------

.. note:
   This example walks through how a user might setup a DIC calculation for large
   displacment images as is often found in micromechanics.

"""

# %%
#  .. note:
#    Because of the size of the images (10000x10000 pixels), 
#    you'll need to download them seperately. This example assumes the
#    files are in the same directory/folder as the example script.
#
#    Images can be downloaded `here <https://github.com/Computer-Aided-Validation-Laboratory/HRDIC-example-data>`_.

# pyvale modules
import pyvale.dic as dic
import matplotlib.pyplot as plt

# %%
# Because of the size of the images, we'll avoid using the interactive GUI for this example.
# We'll create an ROI mask that is the same size as the images using the
# rect_boundary command. If you know that pixels along a certain edge will go
# outside the image bounds post deformation, then excluding them from the ROI
# will prevent the DIC engine from trying to correlate subsets in the reference
# image are not present in the deformed image.
roi = dic.RegionOfInterest(ref_image="ref.tiff")
roi.rect_boundary(left=0,right=0,top=0,bottom=0)


# %%
# We'll also chose a seed location. We've picked this at the centre of the image
# because this approximate area will be displaced the least.
roi.seed = [5000,5000]

# %%
# We can then run the DIC engine using our ROI mask and seed location

dic.calculate_2d(reference="ref.tiff",
                 deformed="def.tiff",
                 roi_mask=roi.mask,
                 seed=roi.seed,
                 subset_size=31,
                 subset_step=15,
                 max_displacement=1000,
                 fft_mad=True,
                 fft_mad_scale=3.0,
                 correlation_criteria="ZNSSD",
                 shape_function="AFFINE",
                 precision=0.001,
                 threshold=0.8,
                 output_basepath="./")

# %%
# If you saved the results in a human-readable format, you can use any tool
# (e.g., Excel, Python, MATLAB) for post-processing.
#
# For convenience, we provide a utility function to import results back into Python
# for analysis and visualization: :func:`pyvale.dic.import_2d`.
#
# The returned object is an instance of :class:`pyvale.DICResults`. If the results
# were saved in binary format or with a custom delimiter, be sure to specify those parameters.
dic_files = "dic_results_*.csv"
dicdata = dic.import_2d(data=dic_files)

# %%
# As an example, here's a simple visualization of the displacement (u, v) and
# correlation cost for the two deformed images using matplotlib. You'll need to
# ensure you have `matplotlib.pyplot` installed and imported.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# First deformation image
im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.u[0])
im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.v[0])
im3 = axes[2].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.cost[0])

# Second deformation image
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

for aa in axes:
    aa.set_aspect('equal')

# Colorbars
fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])
fig.colorbar(im4, ax=axes[3])
fig.colorbar(im5, ax=axes[4])
fig.colorbar(im6, ax=axes[5])

plt.tight_layout()
plt.show()

# %%
# .. image:: ../../../../_static/plate_with_hole.png
#    :alt: Displacement and cost values
#    :width: 800px
#    :align: center
