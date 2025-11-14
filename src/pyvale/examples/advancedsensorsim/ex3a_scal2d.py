# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Scalar field sensors in 2D
================================================================================

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
# 2. Build virtual sensor arrays
# --------------------------------

sens_pos = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                    x_lims=(12.5,12.5),
                                    y_lims=(0.0,33.0),
                                    z_lims=(0.0,12.0))

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

sensor_data = sens.SensorData(positions=sens_pos,
                             sample_times=sample_times)

descriptor = sens.SensorDescriptor(name="Temperature",
                                   symbol="T",
                                   units = r"^{\circ}C",
                                   tag = "TC")

sens_array: sens.SensorArrayPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=descriptor,
)

#%%
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#%% 
# 3. Create & run simulated experiment
# ------------------------------------


output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

pv_plot = sens.plot_point_sensors_on_sim(tc_array,field_key)


save_render = output_path / "customsensors_ex1_3_sensorlocs.svg"
pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
pv_plot.screenshot(save_render.with_suffix(".png"))

pv_plot.show()

errors_on = {"sys": True,
             "rand": True}

error_chain = []
if errors_on["sys"]:
    error_chain.append(sens.ErrSysOffset(offset=-10.0))
    error_chain.append(sens.ErrSysUnif(low=-10.0,
                                            high=10.0))

if errors_on["rand"]:
    error_chain.append(sens.ErrRandNorm(std=5.0))
    error_chain.append(sens.ErrRandUnifPercent(low_percent=-5.0,
                                                high_percent=5.0))

if len(error_chain) > 0:
    err_int_opts = sens.ErrIntOpts()
    error_integrator = sens.ErrIntegrator(error_chain,
                                         sensor_data,
                                         tc_array.get_measurement_shape(),
                                         err_int_opts=err_int_opts)
    tc_array.set_error_integrator(error_integrator)


measurements = tc_array.sim_measurements()

#%%
# 4. Analyse & visualise the results
# ----------------------------------


#%%
# We display the simulation results by printing to the console and by
# plotting the sensor times traces. Try experimenting with the errors above
# to see how the results change.
print("\n"+80*"-")
print("For a virtual sensor: measurement = truth + sysematic error +"
      + " random error")
print(f"measurements.shape = {measurements.shape} ="
      + " (n_sensors,n_field_components,n_timesteps)\n")
print("The truth, systematic error and random error arrays have the same "+
        "shape.")

print(80*"-")

sens_print = 0
comp_print = 0
time_last = 5
time_print = slice(measurements.shape[2]-time_last,measurements.shape[2])

print(f"These are the last {time_last} virtual measurements of sensor "
        + f"{sens_print}:")

sens.print_measurements(tc_array,sens_print,comp_print,time_print)

print(80*"-")


output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)


pv_plot = sens.plot_point_sensors_on_sim(sens_array,
                                         comp_key="temperature")
save_render: Path = output_path / "advanced_exX_locs.png"
pv_plot.off_screen = True
pv_plot.screenshot(save_render)

# Uncomment to show interactive figure and set off_screen = False above
# pv_plot.show()


(fig,ax) = sens.plot_time_traces(sens_array,comp_key="temperature")
save_traces = output_path/"advanced_exX_traces.png"
fig.savefig(save_traces, dpi=300, bbox_inches="tight")

# Uncomment this to display the sensor trace plot 
# plt.show()



