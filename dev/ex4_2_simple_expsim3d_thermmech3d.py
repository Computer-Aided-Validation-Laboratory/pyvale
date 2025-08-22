# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pyvale.mooseherder as mh
import pyvale as pyv


#-------------------------------------------------------------------------------
sim_path = pyv.DataSet.thermomechanical_3d_path()
sim_data = mh.ExodusReader(sim_path).read_all_sim_data()
sim_data = pyv.scale_length_units(scale=1000.0,
                                  sim_data=sim_data)


#-------------------------------------------------------------------------------
x_lims = (12.5,12.5)
y_lims = (0.0,33.0)
z_lims = (0.0,12.0)
n_sens = (1,4,1)
tc_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

sample_times = np.linspace(0.0,np.max(sim_data.time),50)

tc_sens_data = pyv.SensorData(positions=tc_sens_pos,
                              sample_times=sample_times)


tc_array = pyv.SensorArrayFactory \
    .thermocouples_no_errs(sim_data,
                            tc_sens_data,
                            elem_dims=3,
                            field_name="temperature")

tc_err_chain = []
tc_err_chain.append(pyv.ErrSysUnifPercent(low_percent=1.0,high_percent=1.0))
tc_err_chain.append(pyv.ErrRandNormPercent(std=1.0))
tc_error_int = pyv.ErrIntegrator(tc_err_chain,
                                 tc_sens_data,
                                 tc_array.get_measurement_shape())
tc_array.set_error_integrator(tc_error_int)


#-------------------------------------------------------------------------------
exp_sim = pyv.ExperimentSimulator([sim_data,],
                                  [tc_array,],
                                  num_exp_per_sim=100)
exp_data = exp_sim.run_experiments()
exp_stats = exp_sim.calc_stats()



#-------------------------------------------------------------------------------
pv_plot = pyv.plot_point_sensors_on_sim(tc_array,"temperature")
pv_plot.camera_position = [(59.354, 43.428, 69.946),
                            (-2.858, 13.189, 4.523),
                            (-0.215, 0.948, -0.233)]

pv_plot.show()


(fig,ax) = pyv.plot_exp_traces(exp_sim,
                                component="temperature",
                                sens_array_num=0,
                                sim_num=0)

plt.show()
