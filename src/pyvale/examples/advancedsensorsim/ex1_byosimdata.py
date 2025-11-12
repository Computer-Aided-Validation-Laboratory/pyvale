# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Bring your own simulation data 
================================================================================

In this example we demonstrate how you can load your own simulation data from
either plain text or numpy binary array files into a `SimData` object for use
with the pyvale sensor simulation engine. We do this by first converting and 
saving an exodus simulation output file in plain text so we can see the format
that is expected when we load the data.

This example only demonstrates how to load simulation data into a `SimData`
object 
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataset as dataset


#%%
# 1. Save existing SimData to csv 
# --------------------------------
#
# We are going to start by loading one the pre-packaged simulation datasets in
# exodus format that comes with `pyvale`. We are then going to take the
# `SimData` object and save it to the csv/txt format and numpy array formats
# that can be loaded into `pyvale` so we can see what structure these files need
# to be.
data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

#%%
# Let's create our standard pyvale output directory in our current working
# directory so we know where to find the csv and npy files we are going to
# create from our `SimData` object.
output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

#%%
# All the simulation IO for `pyvale` can be found in the `mooseherder` module.
# Here we will save our fields in both formats `BY_TIME` and `BY_FIELD`. For the
# `BY_TIME` case each file will be a time step or frame  were the rows
# correspond to the coordinate and the column corresponds to the individual
# field. For the plain  text version the column headers will be the keys from
# the `SimData.node_vars` dictionary. `BY_FIELD` means there will be one file
# per field where the rows are the coordinate and the columns are the time
# steps. In this case the field key will appear in the file name. We will see
# later that we can load field in either of these formats.
#
# For saving the array files we can either use plain `TXT` or numpy binary
# format `NPY`- here we save both.
#
# Finally, we use the simulation tag of "hex20" which will be a prefix on the
# files we output so we can identify the simulation.
save_opts = mh.SimDataSaveOpts(fields_save_by=mh.ESaveFieldOpt.BOTH,
                               array_format = mh.ESaveArray.BOTH,
                               sim_tag="hex20")

mh.save_sim_data_to_arrays(output_path,sim_data,save_opts)

#%%
# Now if we have a look at the files in the pyvale-output directory we can see
# what the expected format is going to be. There are two key files we need to
# make sure everything loads correctly: 1) the list of nodal coordinates for the
# simulation; and 2) the list of time steps. These can be found in the files:
# "hex20_coords" and "hex20_time.csv". The connectvity table is optional as we
# saw in our last example on mesh free virtual sensors but we will load it here
# to demonstrate mesh based. In this case each meshed object in the simulation
# has a connectivity table labelled "connectX" where X is an integer specifying
# the unique mesh in the simulation. The "hex20_connect1.csv" has the shape 20
# by number of elements in the mesh as we are using 20 node hexahedral elements.
#
# We can also see the field files which are labelled "hex20_node_field_*" with
# a suffix of "frameX" for fields save by time step or a suffix of the field key
# for the case where we have saved by field name.

#%%
# Before we load the data we will specify a common file suffix and the paths to
# the coordinate and time files.

suffix = ".csv" # can be changed to ".npy"
coord_path = output_path / ("hex20_coords" + suffix)
time_path = output_path / ("hex20_time" + suffix)

#%%
# 2. Load SimData 'by time'
# -------------------------
#
# First let's load the data 'by time'. This is a bit more complicated than 'by
# field' as we need to specify how to slice into each frame to extract each
# nodal field variable as well as specifying the wildcard pattern to search for
# in the list of output time steps. 
# 
# We create the `field_slices` dictionary keyed by the same string we want the
# field variable to have when it is loaded into `SimData.node_vars`. The value
# for each key is then a slice specifying the column for the field variable in
# the time step/frame file to read.
#
# We then create the wildcard pattern used to find the field files in the 
# `field_pattern` variable below.  

field_slices = {"disp_x": slice(0,1),
                "disp_y": slice(1,2),
                "disp_z": slice(2,3),
                "temperature": slice(9,10),}

field_pattern = f"hex20_node_field_frame*{suffix}"

connect_pattern = f"hex20_connect*" 

#%%
# We create our load options specifying that we will load the data single 
# threaded and not using the multi-processing library (the default). For loading
# large datasets the number of threads can be changed to a positive integer 
# based on the number of threads available on your machine. Parallelisation will
# only be helpful if your files are large and you have many of them. Here we 
# have a fairly small simulation with few files so single threaded will be 
# faster.
# 
# We can now create our loader and use it to load all the simulation data into
# our `SimData` object which we can now use with the rest of the `pyvale` tools.

load_opts = mh.SimTxtLoadOpts(threads_num=None)

sim_loader = mh.SimTxtLoader(fields_dir=output_path,
                             coords=coord_path,
                             time_steps=time_path,
                             node_files=field_pattern,
                             node_slices=field_slices,
                             connect_dir=output_path,
                             connect_files=connect_pattern,
                             glob_file=None,
                             glob_slices=None,
                             load_opts=load_opts)

sim_data_load = sim_loader.load_all_sim_data()

#%%
# Let's print some summary data to the terminal so we can see what our `SimData`
# object contains:
print(80*"-")
print("SIM DATA: by time")
print(80*"-")
sens.print_sim_data(sim_data_load)
print()

#%%
# 3. Load SimData 'by field'
# --------------------------
#
# Now we will load the data 'by field' which is the simplest case as it is most
# similar to how the `SimData` object stores our nodal fields. Specifically, 
# each nodal variable is a numpy array where the first dimension corresponds to
# the position in the coordinate array and the second dimension is the time 
# step. Here we don't need to slice into field arrays so can just provide a set 
# of the keys we want our node varibles to have in our `SimData` object.

field_slices = {"disp_x","disp_y","disp_z","temperature"}

prefix = "hex20_node_field"

field_patterns = {}
for ff in field_slices:
    field_patterns[ff] = f"{prefix}_{ff}{suffix}"

#%%
# When we load data 'by field' the files generally do no have header rows so we
# set this to None in the load options. Other than that loading the simulation 
# data into our `SimData` object is exactly the same as we did previously. 

load_opts = mh.SimTxtLoadOpts(node_field_header=None,
                              threads_num=None)

sim_loader = mh.SimTxtLoader(fields_dir=output_path,
                             coords=coord_path,
                             time_steps=time_path,
                             node_files=field_patterns,
                             node_slices=field_slices,
                             connect_dir=output_path,
                             connect_files=connect_pattern,
                             glob_file=None,
                             glob_slices=None,
                             load_opts=load_opts)

sim_data_load = sim_loader.load_all_sim_data()

print(80*"-")
print("SIM DATA: by field")
print(80*"-")
sens.print_sim_data(sim_data_load)

#%%
# That's it for this example! We will leave it as an exercise to load the strain
# fields from the files in the output directory.
