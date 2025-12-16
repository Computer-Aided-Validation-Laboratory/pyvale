# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Mesh-free virtual sensors
================================================================================

This example shows the ability of the pyvale sensor simulation module to apply
virtual sensors for the case where the provided simulation data is a point
cloud and does not contain a structured mesh with a connectivity table. For this
case pyvale uses a Delaunay triangulation to perform linear interpolation of
the simulation point cloud at the virtual sensor locations. The mesh free
method is computationally more expensive than the mesh based method, especially
in 3D, but it does give much more flexibility in input simulation data.

The only difference between the mesh-based and mesh-free cases from a user
interface perspective is that there is no connectivity table in the `SimData`
object created by the user. Otherwise creating a sensor array is exactly the
same as all previous examples - apart from the computational overhead of the
mesh-free interpolation.
"""

import copy
import time
import numpy as np
import pyvale as pyv

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.verif as verif


#%%
# 1. Create analytic simulation data
# ----------------------------------
# For this example we are going to use a generated analytic dataset instead of
# a finite element simulation as we want an analytic function for the measured
# field that is linear so we can see that mesh-free interpolation matches the
# mesh-based interpolation. For finite element simulations using non-tetrahedral
# and/or high order elements the mesh-free method will not match the mesh-based.
#
# The analytic case is a rectangular plate that is 10x7.5 units in length with
# a mesh of 40x30 elements. The physical field is scalar with a bi-linear
# spatial gradient of 20/10 = 2 in the X direction and 10/7.5 = 1.33 in the Y
# direction.
#
# Initially the generated `SimData` object is mesh-based so we create a copy and
# then set the connectivity table dictionary to None to indicate that our data
# is mesh-free.
(sim_data,_) = verif.analyticsimdatafactory.scalar_linear_2d()

sim_data_nomesh = copy.deepcopy(sim_data)
sim_data_nomesh.connect = None

#%%
# We will use the same `SensorData` for the mesh-free and mesh-based sensor
# arrays. First, we use a helper function to get the limiting dimensions of the
# plate and use this to create our sensor position array. Then we specify our
# sample times and use our sensor positions and sample times to create our
# `SensorData` object.

sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos = sens.gen_pos_grid_inside(num_sensors=(4,2,1),
                                        x_lims=sim_dims["x"],
                                        y_lims=sim_dims["y"],
                                        z_lims=(0.0,0.0))

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

sens_data = sens.SensorData(positions=sens_pos,
                            sample_times=sample_times)

#%%
# 2. Build virtual sensor arrays
# --------------------------------
# Now we create our virtual sensor arrays. There is no difference between the
# mesh-based and mesh-free case in how these are created as `pyvale`
# automatically detects that the connectivity table in the `SimData` object is
# None and creates the Delaunay triangulation for later interpolation to virtual
# sensor locations.
#
# The triangulation is computationally expensive, especially in 3D. For the 2D
# case shown here with 1200 elements creating the sensor array is about 6
# times slower than the mesh-based case.

start_time = time.perf_counter()
tc_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)
mesh_time = (time.perf_counter() - start_time)*1000.0


start_time = time.perf_counter()
tc_array_nomesh: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data_nomesh,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)

nomesh_time = (time.perf_counter() - start_time)*1000.0

print(80*"-")
print("Sensor Array Creation Times:")
print(f"Mesh based = {mesh_time:.3f} milliseconds")
print(f"Mesh free  = {nomesh_time:.3f} milliseconds\n")

#%%
# 3. Run simulated experiment
# ------------------------------------
# Now we can simulate some measurements for the mesh-based and mesh-free cases
# and compare the time taken for the calculation. Rough testing shows that the
# mesh-free case takes about 5 times longer than the mesh-based case.

start_time = time.perf_counter()
meas = tc_array.get_measurements()
mesh_time = (time.perf_counter() - start_time)*1000.0

start_time = time.perf_counter()
meas_nomesh = tc_array_nomesh.get_measurements()
nomesh_time = (time.perf_counter() - start_time)*1000.0

print(80*"-")
print("Measurement Simulation Times")
print(f"Mesh based = {mesh_time:.3f} milliseconds")
print(f"Mesh free  = {nomesh_time:.3f} milliseconds\n")


#%%
# 4. Analyse the results
# ----------------------------------
# Finally, we can compare the measurements for both sensor arrays. As our
# virtual sensors are measuring a linear field in 2D with linear elements the
# mesh-based and mesh-free cases should agree exactly (especially because we
# are not simulating any measurement errors).

print(80*"-")
print("MESH BASED INTERPOLATION")
sens.print_measurements(tc_array,
                        slice(0,1), # Sensor 1
                        slice(0,1), # Component 1: scalar field = 1 component
                        slice (meas.shape[2]-5,meas.shape[2]))

print("MESH FREE INTERPOLATION")
sens.print_measurements(tc_array_nomesh,
                        slice(0,1), # Sensor 1
                        slice(0,1), # Component 1: scalar field = 1 component
                        slice (meas_nomesh.shape[2]-5,meas_nomesh.shape[2]))

print(f"\n{np.allclose(meas,meas_nomesh)=}")
print(80*"-")

# %%
# Example terminal output with timings and numerical comparison between
# mesh-based and mesh-free virtual sensors:
#
# .. image:: ../../../../_static/ext_ex2_term_out.png
#    :alt: Terminal output comparing mesh-based and mesh-free sensors.
#    :width: 700px
#    :align: center

#%%
# That's it for this example! Mesh-free virtual sensors work in exactly the same
# way as the mesh-based sensors, they are just a bit more computationally heavy
# and slower.

