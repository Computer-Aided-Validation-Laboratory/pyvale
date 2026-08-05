
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.dataset as dataset


# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Scalar field sensor sim
================================================================================

In this example we introduce the basic features of `pyvale` for scalar field
sensor simulation. We demonstrate quick sensor array construction with defaults
using the `pyvale` sensor factory. We also introduce some key concepts for
`pyvale` sensor simulation including error chains and the functions for running
simulated sensor measurements as well as the data structures they are stored in.
Finally, we run a sensor simulation, visualise the virtual sensor locations and
plot the simulated sensor traces.

Before we begin the example, we will briefly describe the `pyvale` sensor
measurement simulation model. In `pyvale` a simulated measurement is given by:

measurement = truth + systematic errors + random errors

The truth is interpolated from the input physics simulation to the virtual
sensor positions and times. The systematic and random errors are evaluated for
each masurement simulation by sampling probability distributions in a sequence
called an error chain.

`pyvale` provides a library of common systematic (position uncertainty,
spatial/temporal averaging, digitisation, calibration, etc.) and random errors
(probability distribution in absolute units or as a percentage of the truth
etc.). These errors all implement the `IErrSimulator` interface allowing a user
to plug-and-play any combination of simulated errors in their error chain.

Ok, now let's simulate some temperature measurements!
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.dataset as dataset


#%%
# 1. Load physics simulation data
# -------------------------------
# Here we load a MOOSE finite element simulation dataset that comes packaged
# with `pyvale` in exodus (*.e) format. `pyvale` loads simulations into a
# `SimData` object  which contains the nodal coordinates, simulation time steps,
# the nodal physics variables and optionally the element connectivity tables.
#
# We also convert the length units of our simulation from meters to milli-meters
# as our visualisation tools are based on unit scaling by default.

data_path: Path = dataset.thermal_3d_path()
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()

sim_data: io.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data,
                                               disp_keys=None)
                                               
#%%
# .. note::
#   You can load your own exodus (*.e) file here by changing the path or you can
#   load your own simulation data from delimited plain text files or numpy npy
#   files. See the advanced example 'Bring your own simulation data'.

#%%
# 2. Build virtual sensor arrays
# -------------------------------
# First, we need to specify the position of our virtual sensors and the times
# that they should take simulated measurements as a numpy array. `pyvale` has
# helper functions for common sensor patterns like a regular grid inside given
# bounds but we could also have manually built the numpy array of sensor
# locations which has shape=(num_sensors,coord[x,y,z]).
#
# The `SensorData` object allows us to specify the parameters to create the
# virtual sensor array. You can also set `sample_times=None` in `SensorData`
# which will make our virtual sensors sample at the simulation time steps.

sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                x_lims=(12.5,12.5),
                                                y_lims=(0.0,33.0),
                                                z_lims=(0.0,12.0))

sample_times: np.ndarray = np.linspace(0.0,np.max(sim_data.time),50)

sens_data = sens.SensorData(positions=sens_pos,
                            sample_times=sample_times)

#%%
# We now create our virtual sensor array for a scalar field. We need to specify
# the component string key to be the same as for the nodal field variable we
# want our sensors to sample from in the `SimData` object. Our simulation is 3D
# so we specify that here and we add a descriptor (optional) that will be used
# to set the axes labels, symbols and units on our visualisations.
sens_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)

#%%
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we add some simulated errors to our sensor array with an `error_chain`
# which is a list of objects that implement the `IErrSimulator` interface.
# `pyvale` will evaluate these errors in the order they are specified in the
# list when we simulate our measurements. The error chain is the core of the
# `pyvale` sensor simulation engine.

err_chain: list[sens.IErrSimulator] = [
    sens.ErrSysGen(sens.GenUniform(low=-10.0,high=10.0)),
    sens.ErrRandGen(sens.GenNormal(std=5.0)),
]

sens_array.set_error_chain(err_chain)

#%%
# 3. Run a simulated experiment
# -----------------------------
# We have built our sensor array so now we can call `.sim_measurements()` to
# generate simulated sensor traces. When we call this function `pyvale` will
# calculate the ground truth (if not already complete from a previous sim), then
# step through the error chain sampling probability distributions for our
# errors.
#
# If we call `.sim_measurements()` again the process is repeated and the errors
# are resampled. However, if we call `.get_measurements()` then we are returned
# the previously simulated values. Throughout `pyvale` methods prefixed with
# `get` can be expected to return previous values if they exist whereas `sim`
# or `calc `methods will actually perform a simulation or calculation.
measurements: np.ndarray = sens_array.sim_measurements()

truth: np.ndarray = sens_array.get_truth()
sys_errs: np.ndarray = sens_array.get_errors_systematic()
rand_errs: np.ndarray = sens_array.get_errors_random()

print(80*"-")
print("pyvale sensor simulation model:")
print("    measurement = truth + sysematic error + random error\n")

print(f"measurements.shape = {measurements.shape} = "
        + "(n_sensors,n_field_components,n_timesteps)\n")
print(f"truth.shape     = {truth.shape}")
print(f"sys_errs.shape  = {sys_errs.shape}")
print(f"rand_errs.shape = {rand_errs.shape}")

sens_print: int = 0
comp_print: int = 0
time_last: int = 5
time_print = slice(measurements.shape[2]-time_last,measurements.shape[2])

print(f"\nThese are the last {time_last} virtual measurements of sensor "
        + f"{sens_print}:\n")

sens.print_measurements(sens_array,sens_print,comp_print,time_print)
print("\n"+80*"-")

################################################################################

disp_keys = ("disp_x","disp_y")

data_path2: Path = dataset.thermal_3d_path()
sim_data2: io.SimData = mh.ExodusLoader(data_path2).load_all_sim_data()

sim_data2: io.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data2,
                                               disp_keys=disp_keys)


sens_angles2: tuple[Rotation] = sens_pos.shape[0] * \
    (Rotation.from_euler("zyx",[90,0,0], degrees=True),)


disp_sens_data = sens.SensorData(positions=sens_pos,
                                 sample_times=sample_times,
                                 angles=sens_angles2)

disp_sens: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data2,
    disp_sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.displacement(),
)


################################################################################

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


save_render = output_path / "basics_ex1_1_sensorlocs.svg"

#%%
# This creates a pyvista visualisation of the sensor locations on the
# simulation mesh. The plot will can be shown in interactive mode by calling
# `pv_plot.show()`.

# plot a single sensor array

field_key = "temperature"

pv_plot = sens.plot_point_sensors_on_sim(sens_array, field_key)
pv_plot.show()

pv_plot = sens.plot_point_sensors_on_sim(disp_sens, "disp_x")
pv_plot.show()


# using two sensor arrays, each with a different component key
pv_plot = sens.plot_point_sensors_on_sim([sens_array, disp_sens], [field_key, "disp_x"])
pv_plot.show()


pv_plot = sens.plot_point_sensors_on_sim(disp_sens, "disp_x")
pv_plot.show()

#%%
# We determined manually by moving camera in interative mode and then
# printing camera position to console after window close, as below.
# pyvista applies changes to the last accessed subplot - (0,1) here
# to apply to all subplots, we must manually change subplot and apply
# changes individually
pv_plot.subplot(0,0)
pv_plot.camera_position = [(-7.547, 59.753, 134.52),
                            (41.916, 25.303, 9.297),
                            (0.0810, 0.969, -0.234)]


pv_plot.screenshot(save_render.with_suffix(".png"))

#%%
# We can also show the simulation and sensor locations in interative mode
# by calling `.show()`

print(80*"-")
print("Camera position after interactive view:")
print(pv_plot.camera_position)
print(80*"-"+"\n")

