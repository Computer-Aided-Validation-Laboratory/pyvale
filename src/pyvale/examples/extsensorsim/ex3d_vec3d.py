# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Vector field sensors in 3D
================================================================================

This example demonstrates the application of the `pyvale` sensor simulation
module to vector fields in 3 spatial dimensions. An example of a vector field
sensor would be a displacement transducer, point tracking or velocity sensor.

Note that this example has minimal explanation and assumes you have reviewed the
basic sensor simulation examples to understand how the underlying engine works
as well as the sensor simulation workflow.
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

data_path: Path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
sim_data: mh.SimData = mh.ExodusLoader(data_path).load_all_sim_data()

disp_keys = ("disp_x","disp_y","disp_z")
sim_data: mh.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data,
                                               disp_keys=disp_keys)

#%%
# 2. Build virtual sensor arrays
# --------------------------------

sens.simtools.print_dimensions(sim_data)

# Simulations is a 10mm cube
sensor_positions = np.array(((5.0,0.0,5.0),     # cube x-z face
                             (5.0,10.0,5.0),    # cube x-z face
                             (5.0,5.0,0.0),     # cube x-y face
                             (5.0,5.0,10.0),    # cube x-y face
                             (0.0,5.0,5.0),     # cube y-z face
                             (10.0,5.0,5.0),))  # cube y-z face

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

sens_data = sens.SensorData(positions=sensor_positions,
                            sample_times=sample_times)


sens_array: sens.SensorsPoint = sens.SensorFactory.vector_point(
    sim_data,
    sens_data,
    comp_keys=disp_keys,
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.displacement(),
)

#%%
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

pos_uncert = 1.0 # units = mm
pos_rand = (sens.GenUniform(low=-pos_uncert,high=pos_uncert),
            sens.GenUniform(low=-pos_uncert,high=pos_uncert),
            sens.GenUniform(low=-pos_uncert,high=pos_uncert))

pos_lock = np.full(sensor_positions.shape,False,dtype=bool)
pos_lock[0,1] = True # cube x-z face, lock y
pos_lock[1,1] = True # cube x-z face, lock y
pos_lock[2,2] = True # cube x-y face, lock z
pos_lock[3,2] = True # cube x-y face, lock z
pos_lock[4,0] = True # cube y-z face, lock x
pos_lock[5,0] = True # cube y-z face, lock x

field_err_data = sens.ErrFieldData(pos_rand_xyz=pos_rand,
                                   pos_lock_xyz=pos_lock)

error_chain: list[sens.IErrSimulator] = [
    sens.ErrSysGenPercent(sens.GenUniform(low=-1.0,high=1.0)),
    sens.ErrRandGenPercent(sens.GenNormal(std=1.0)),
    sens.ErrSysField(sens_array.get_field(),field_err_data),
]

sens_array.set_error_chain(error_chain)


#%%
# 3. Run a simulated experiment
# -----------------------------

measurements: np.ndarray = sens_array.sim_measurements()

truth: np.ndarray = sens_array.get_truth()
sys_errs: np.ndarray = sens_array.get_errors_systematic()
rand_errs: np.ndarray = sens_array.get_errors_random()

print(80*"-")
print("measurement = truth + sysematic error + random error")

print(f"measurements.shape = {measurements.shape} = "
        + "(n_sensors,n_field_components,n_timesteps)")
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

#%%
# 4. Analyse & visualise the results
# ----------------------------------

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

for kk in disp_keys:
    pv_plot = sens.plot_point_sensors_on_sim(sens_array,kk)
    pv_plot.camera_position = "yz"
    pv_plot.camera.azimuth = 45
    pv_plot.camera.elevation = 45

    # Set to False to show an interactive plot instead of saving the figure
    pv_plot.off_screen = True
    if pv_plot.off_screen:
        pv_plot.screenshot(output_path/f"ext_ex3d_locs_{kk}.png")
    else:
        pv_plot.show()

# %%
# .. image:: ../../../../_static/ext_ex3d_locs_disp_y.png
#    :alt: Virtual sensor location visualisation.
#    :width: 800px
#    :align: center


for kk in disp_keys:
    (fig,ax) = sens.plot_time_traces(sens_array,comp_key=kk)
    fig.savefig(output_path/f"ext_ex3d_traces_{kk}.png",
                dpi=300,
                bbox_inches="tight")

# Uncomment this to display the sensor trace plot
# plt.show()

# %%
# .. image:: ../../../../_static/ext_ex3d_traces_disp_y.png
#    :alt: Simulated sensor traces.
#    :width: 600px
#    :align: center
