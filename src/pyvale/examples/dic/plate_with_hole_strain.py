# ================================================================================
# Example: DIC Challenge 2.0 Comparison
# 
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2024 The Computer Aided Validation Team
# ================================================================================

"""
Strain and Deformation gradient Calculations
---------------------------------------------
This example leads on from the previous one, so its assumed that the results
have been generated in the current working directory ready to be used for the
strain calculation
"""

import pyvale


# %%
# you can perform a strain calculation on the DIC data. You don't have
# to import the data first, you can simply set data argument to the name and location of
# the output data (and you'll also need to set the delimiter and format if that
# has been altered in anyway).
dic_data = pyvale.dic_data_import(data="./plate_with_hole_*.dat")
 
# %%
# At a minimum you'll need to set the strain window wize and
# the element type to use over the strain window. The allowed options for the
# window_elemnt is 4 (bilinear) or 9 (biquadratic). The output will contain, at
# a minimum, the strain window locations and the deformation gradient. If the
# user has provided a strain_formulation, then the 2D strain tensor will also
# be included in the output files.
pyvale.dic_strain(data=dic_data, window_size=5, window_element=4,
                            strain_formulation="ALMANSI")

# %% 
# The results can be read back into python following the completion of the
# calculation by using the func:`pyvale.strainDataImport` command.
straindata = pyvale.strain_data_import()



