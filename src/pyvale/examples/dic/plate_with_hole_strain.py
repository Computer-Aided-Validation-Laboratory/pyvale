# ================================================================================
# Example: DIC Challenge 2.0 Comparison
# 
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ================================================================================

"""
Strain and Deformation Gradient Calculations
---------------------------------------------
This example leads on from the previous one, so its assumed that the results
have been generated in the current working directory ready to be used for the
strain calculation
"""

import matplotlib.pyplot as plt
import pyvale


# %%
# you can perform a strain calculation on the DIC data. You don't have
# to import the data first, you can simply set data argument to the name and location of
# the output data (and you'll also need to set the delimiter and format if that
# has been altered in anyway).
dic_data = pyvale.dic_data_import(data="dic_results_*.dat")
 
# %%
# At a minimum you'll need to set the strain window wize and
# the element type to use over the strain window. The allowed options for the
# window_elemnt is 4 (bilinear) or 9 (biquadratic). The output will contain, at
# a minimum, the strain window locations and the deformation gradient. If the
# user has provided a strain_formulation, then the 2D strain tensor will also
# be included in the output files.
pyvale.strain_2d(data=dic_data, window_size=5, window_element=4)

# %% 
# The results can be read back into python following the completion of the
# calculation by using the func:`pyvale.strainDataImport` command.
straindata = pyvale.strain_data_import(data="strain_dic_results_*", 
                                       binary=False, delimiter=" ",
                                       layout="matrix")

# %%
# As an example of some very simple visualisation, you could loop over the
# number of deformed images and plot the displacement and cost values using the
# below. You'll need to make sure you have matplotlib.pyplot installed and imported.
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
axes = axes.flatten()

# first deformation image
im1 = axes[0].pcolor(straindata.window_x, straindata.window_y, straindata.def_grad[0,:,:,0,0])
im2 = axes[1].pcolor(straindata.window_x, straindata.window_y, straindata.def_grad[0,:,:,0,1])
im3 = axes[2].pcolor(straindata.window_x, straindata.window_y, straindata.def_grad[0,:,:,1,0])
im4 = axes[3].pcolor(straindata.window_x, straindata.window_y, straindata.def_grad[0,:,:,1,1])

# Titles
axes[0].set_title('deformation gradient xx')
axes[1].set_title('deformation gradient xy')
axes[2].set_title('deformation gradient yx')
axes[3].set_title('deformation gradient yy')


fig.colorbar(im1, ax=axes[0])
fig.colorbar(im2, ax=axes[1])
fig.colorbar(im3, ax=axes[2])
fig.colorbar(im4, ax=axes[3])


# layout
plt.show()




