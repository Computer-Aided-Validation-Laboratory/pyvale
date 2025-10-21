# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Basics: Mesh free sensors
================================================================================

In this example we will compare pyvale
TODO

Test case:
"""
import copy
import time
import numpy as np
import pyvale as pyv

# Pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.verif as verif


#%%
(sim_data,_) = verif.analyticsimdatafactory.scalar_linear_2d()

sim_data_nomesh = copy.deepcopy(sim_data)
sim_data_nomesh.connect = None

#%%
#

field_key = 'temperature'
start_time = time.perf_counter()
scal_field = sens.FieldScalar(sim_data,
                              field_key=field_key,
                              elem_dims=2)
mesh_time = (time.perf_counter() - start_time)*1000.0

start_time = time.perf_counter()
scal_field_nomesh = sens.FieldScalar(sim_data_nomesh,
                                     field_key=field_key,
                                     elem_dims=2)
nomesh_time = (time.perf_counter() - start_time)*1000.0

print(80*"-")
print("Field Creation Times")
print(f"Mesh based = {mesh_time:.3f} milliseconds")
print(f"Mesh free  = {nomesh_time:.3f} milliseconds\n")

#%%
sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos = sens.create_sensor_pos_array(num_sensors=(4,1,1),
                                        x_lims=sim_dims["x"],
                                        y_lims=sim_dims["y"],
                                        z_lims=(0.0,0.0))

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

sensor_data = sens.SensorData(positions=sens_pos,
                             sample_times=sample_times)

descriptor = sens.SensorDescriptorFactory.temperature_descriptor()


#%%

tc_array = sens.SensorArrayPoint(sensor_data,
                                 scal_field,
                                 descriptor)

tc_array_nomesh = sens.SensorArrayPoint(sensor_data,
                                        scal_field_nomesh,
                                        descriptor)


#%%
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

print(f"{np.allclose(meas,meas_nomesh)=}")
print(80*"-")


#%%

