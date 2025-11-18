# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Errors: basics
================================================================================

In this example we will provide an overview of the basic error library in 
`pyvale`. The error simulation models in pyvale are grouped into  types as 
either random (`ErrRand*`) or systematic (`ErrSys*`). In this example we will
consider probability distribution based sampled errors, constant offsets and
basic systematic errors such as digitisation / saturation.

In the next examples we will consider more advanced error sources including:
field errors that perturb the sensor parameters (e.g. location, sampling time
and orientation) requiring re-interpolation of the underlying field data; and
calibration errors.

Advanced users: It is also possible to write custom errors by writing your own
class that implements the `IErrSimulator` abstract base class and then add them
to your error chain.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.dataset as dataset



#%%
# 1. Load physics simulation data
# -------------------------------

data_path = dataset.thermal_3d_path()
sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data = sens.scale_length_units(scale=1000.0,
                                   sim_data=sim_data,
                                   disp_keys=None)

#%% 
# 2. Build virtual sensor array
# -----------------------------

sim_dims = sens.simtools.get_sim_dims(sim_data)
sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                x_lims=(12.5,12.5),
                                                y_lims=sim_dims["y"],
                                                z_lims=sim_dims["z"])

                                    
sample_times = np.linspace(0.0,np.max(sim_data.time),50)

sens_data = sens.SensorData(positions=sens_pos,
                            sample_times=sample_times)

sens_array: sens.SensorArrayPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)


#%%
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we have our sensor array applied to our simulation without any
# errors we can build a custom chain of basic errors. Here we will start by
# adding a series of systematic errors that are independent:
err_chain = []

#%%
# For probability sampling systematic errors the distribution is sampled to
# provide an offset which is assumed to be constant over all sensor sampling
# times. This is different to random errors which are sampled to provide a
# different error for each sensor and time step.
#
# These systematic errors provide a constant offset to all measurements in
# simulation units or as a percentage.
err_chain.append(sens.ErrSysOffset(offset=-10.0))
err_chain.append(sens.ErrSysOffsetPercent(offset_percent=-1.0))

#%%
# These systematic errors are sampled from a uniform or normal probability
# distribution either in simulation units or as a percentage.
err_chain.append(sens.ErrSysUnif(low=-1.0,
                                high=1.0))
err_chain.append(sens.ErrSysUnifPercent(low_percent=-1.0,
                                        high_percent=1.0))
err_chain.append(sens.ErrSysNorm(std=1.0))
err_chain.append(sens.ErrSysNormPercent(std_percent=1.0))

#%%
# pyvale includes a series of random number generator objects that wrap the
# random number generators from numpy. These are named `Gen*` and can be
# used with an `ErrSysGen` or an `ErrSysGenPercent` object to create custom
# probability distribution sampling errors:
sys_gen = sens.GenTriangular(left=-1.0,
                            mode=0.0,
                            right=1.0)
err_chain.append(sens.ErrSysGen(sys_gen))

#%%
# We can also build the equivalent of `ErrSysUnifPercent` above using a
# `Gen` object inserted into an `ErrSysGenPercent` object:
unif_gen = sens.GenUniform(low=-1.0,
                            high=1.0)
err_chain.append(sens.ErrSysGenPercent(unif_gen))

#%%
# We can also add a series of random errors in a similar manner to the
# systematic errors above noting that these will generate a new error for
# each sensor and each time step whereas the systematic error sampling
# provides a constant shift over all sampling times for each sensor.
err_chain.append(sens.ErrRandNorm(std = 2.0))
err_chain.append(sens.ErrRandNormPercent(std_percent=2.0))
err_chain.append(sens.ErrRandUnif(low=-2.0,high=2.0))
err_chain.append(sens.ErrRandUnifPercent(low_percent=-2.0,
                                        high_percent=2.0))
rand_gen = sens.GenTriangular(left=-5.0,
                              mode=0.0,
                              right=5.0)
err_chain.append(sens.ErrRandGen(rand_gen))

#%%
# Finally we add some dependent systematic errors including rounding errors,
# digitisation and saturation. Note that the saturation error must be placed
# last in the error chain. Try changing some of these values to see how the
# sensor traces change - particularly the saturation error.
err_chain.append(sens.ErrSysRoundOff(sens.ERoundMethod.ROUND,0.1))
err_chain.append(sens.ErrSysDigitisation(bits_per_unit=2**16/100))
err_chain.append(sens.ErrSysSaturation(meas_min=0.0,meas_max=400.0))

sens_array.set_error_chain(err_chain)


#%% 
# 3. Create & run simulated experiment
# ------------------------------------

measurements = sens_array.sim_measurements()

print(80*"-")

sens_print = 0
comp_print = 0
time_last = 5
time_print = slice(measurements.shape[2]-time_last,measurements.shape[2])


print(f"These are the last {time_last} virtual measurements of sensor "
        + f"{sens_print}:")

sens.print_measurements(sens_array,sens_print,comp_print,time_print)

print(80*"-")

sens.plot_time_traces(sens_array,field_key)
plt.show()

#%%
# 4. Analyse & visualise the results
# ----------------------------------
