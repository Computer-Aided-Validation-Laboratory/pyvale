# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Quickstart sensor sim
================================================================================

This is a quick example with minimal explantion to get users familiar with the 
overall workflow for the `pyvale` sensor simulation engine - to see if `pyvale` 
is the right virtual laboratory for them. 
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset

#%%
# 1. Load physics simulation
# --------------------------
data_path: Path = dataset.thermomechanical_3d_path()
sim_data: mh.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data: mh.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data,
                                               disp_comps=None)

#%% 
# 2. Create a virtual sensor array
# --------------------------------
sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,4,1),
                                                    x_lims=(12.5,12.5),
                                                    y_lims=(0.0,33.0),
                                                    z_lims=(0.0,12.0))
sens_data = sens.SensorData(positions=sens_pos)

sens_array: sens.SensorArrayPoint = sens.SensorFactory.scalar_no_errs(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)


#%%
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
err_chain: list[sens.IErrSimulator] = [sens.ErrSysUnif(low=-1.0,high=1.0),]
err_chain.append(sens.ErrSysUnifPercent(low_percent=-1.0,high_percent=1.0))
err_chain.append(sens.ErrRandNorm(std=1.0))
err_chain.append(sens.ErrRandNormPercent(std_percent=1.0))
err_chain.append(sens.ErrSysDigitisation(bits_per_unit=2**16/100))
err_chain.append(sens.ErrSysSaturation(meas_min=0.0,meas_max=450.0))

sens_array.set_error_chain(err_chain)

#%% 
# 3. Create virtual experiment
# ----------------------------
sim_list: list[mh.SimData] = [sim_data,]
sensor_arrays: list[sens.ISensorArray] = [sens_array,]
exp_sim = sens.ExperimentSimulator(sim_list,
                                   sensor_arrays)

#%%
# 4. Run virtual experiments
# --------------------------
exp_data = exp_sim.run_experiments(num_exp_per_sim=100)
exp_stats = exp_sim.calc_stats()


#%%
# 5. Visualise the setup and results
# ----------------------------------
pv_plot = sens.plot_point_sensors_on_sim(sens_array,"temperature")
pv_plot.camera_position = [(59.354, 43.428, 69.946),
                            (-2.858, 13.189, 4.523),
                            (-0.215, 0.948, -0.233)]
pv_plot.show()


trace_opts = sens.TraceOptsExperiment(plot_all_exp_points=True)
(fig,ax) = sens.plot_exp_traces(exp_sim,
                                component="temperature",
                                sens_array_num=0,
                                sim_num=0,
                                trace_opts=trace_opts)
plt.show()
