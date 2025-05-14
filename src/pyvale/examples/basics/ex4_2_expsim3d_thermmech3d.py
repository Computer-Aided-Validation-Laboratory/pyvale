# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Pyvale example: Multi-physics experiment simulation in 3D
--------------------------------------------------------------------------------
In the previous example we performed a series of simulated experiments on a set
of 2D multi-physics simulations. Here we use a 3D thermo-mechanical analysis of
a divertor armour heatsink to show how we can run simulated experiments in 3D.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar and tensor fields as described in the previous examples.

Test case: thermo-mechanical analysis of a divertor heatsink in 3D
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    # First we get the path to simulation output file and then read the
    # simulation into a `SimData`  object. In this case our simulation is a
    # thermomechanical model of a divertor heatsink.
    sim_path = pyv.DataSet.thermomechanical_3d_path()
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()
    elem_dims: int = 3
    # We scale our length and displacement units to mm to help with
    # visualisation.
    disp_comps = ("disp_x","disp_y","disp_z")
    sim_data = pyv.scale_length_units(scale=1000.0,
                                      sim_data=sim_data,
                                      disp_comps=disp_comps)

    # If we are going to save figures showing where our sensors are and their
    # simulated traces we need to create a directory. Set the flag below to
    # save the figures when you run the script
    save_figs = False
    save_tag = "thermomech3d"
    fig_save_path = Path.cwd()/"images"
    if not fig_save_path.is_dir():
        fig_save_path.mkdir(parents=True, exist_ok=True)

    # We specify manual sensor sampling times but we could also set this to None
    # for the sensors to sample at the simulation time steps.
    sample_times = np.linspace(0.0,np.max(sim_data.time),50)


    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    n_sens = (1,4,1)
    tc_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_sens_data = pyv.SensorData(positions=tc_sens_pos,
                                  sample_times=sample_times)

    # We use the sensor array factory to create our thermocouple array with no
    # errors.
    tc_field_name = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_no_errs(sim_data,
                                tc_sens_data,
                                elem_dims=elem_dims,
                                field_name=tc_field_name)

    # Now we build our error chain starting with some basic errors on the order
    # of 1 degree.
    tc_err_chain = []
    tc_err_chain.append(pyv.ErrSysUnif(low=1.0,high=1.0))
    tc_err_chain.append(pyv.ErrRandNorm(std=1.0))

    # Now we add positioning error for our thermocouples.
    tc_pos_uncert = 0.1 # units = mm
    tc_pos_rand = (pyv.GenNormal(std=tc_pos_uncert),
                    pyv.GenNormal(std=tc_pos_uncert),
                    pyv.GenNormal(std=tc_pos_uncert))

    # We block translation in x so the thermocouples stay attached.
    tc_pos_lock = np.full(tc_sens_pos.shape,False,dtype=bool)
    tc_pos_lock[:,0] = True

    tc_field_err_data = pyv.ErrFieldData(pos_rand_xyz=tc_pos_rand,
                                         pos_lock_xyz=tc_pos_lock)
    tc_err_chain.append(pyv.ErrSysField(tc_array.get_field(),
                                        tc_field_err_data))
    # We have finished our error chain so we can build our error integrator and
    # attach it to our thermocouple array.
    tc_error_int = pyv.ErrIntegrator(tc_err_chain,
                                     tc_sens_data,
                                     tc_array.get_measurement_shape())
    tc_array.set_error_integrator(tc_error_int)

    # We visualise our thermcouple locations on our mesh to make sure they are
    # in the correct positions.
    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,"temperature")
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]
    if save_figs:
        pv_plot.save_graphic(fig_save_path/(save_tag+"_tc_vis.svg"))
        pv_plot.screenshot(fig_save_path/(save_tag+"_tc_vis.png"))

    pv_plot.show()

    # Now we have finished with our thermocouple array we can move on to our
    # strain gauge array.

    # We use the same sampling time but we are going to place the strain gauges
    # down the side of the monoblock where the pipe passes through.
    x_lims = (9.4,9.4)
    y_lims = (0.0,33.0)
    z_lims = (12.0,12.0)
    n_sens = (1,4,1)
    sg_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sg_sens_data = pyv.SensorData(positions=sg_sens_pos,
                                  sample_times=sample_times)

    # We use the sensor array factory to give us a strain gauge array with no
    # errors.
    sg_field_name = "strain"
    sg_norm_comps = ("strain_xx","strain_yy","strain_zz")
    sg_dev_comps = ("strain_xy","strain_yz","strain_xz")
    sg_array = pyv.SensorArrayFactory \
        .strain_gauges_no_errs(sim_data,
                                sg_sens_data,
                                elem_dims=elem_dims,
                                field_name=sg_field_name,
                                norm_comps=sg_norm_comps,
                                dev_comps=sg_dev_comps)

    # Now we build our error chain starting with some basic errors on the order
    # of 1 percent.
    sg_err_chain = []
    sg_err_chain.append(pyv.ErrSysUnifPercent(low_percent=1.0,high_percent=1.0))
    sg_err_chain.append(pyv.ErrRandNormPercent(std_percent=1.0))

    # We are going to add +/-2 degree rotation uncertainty to our strain gauges.
    angle_uncert = 2.0
    angle_rand_zyx = (pyv.GenUniform(low=-angle_uncert,high=angle_uncert), # units = deg
                      pyv.GenUniform(low=-angle_uncert,high=angle_uncert),
                      pyv.GenUniform(low=-angle_uncert,high=angle_uncert))

    # We only allow rotation on the face the strain gauges are on
    angle_lock = np.full(sg_sens_pos.shape,True,dtype=bool)
    angle_lock[:,0] = False   # Allow rotation about z

    sg_field_err_data = pyv.ErrFieldData(ang_rand_zyx=angle_rand_zyx,
                                         ang_lock_zyx=angle_lock)
    sg_err_chain.append(pyv.ErrSysField(sg_array.get_field(),
                                        sg_field_err_data))

    # We have finished our error chain so we can build our error integrator and
    # attach it to our thermocouple array.
    sg_error_int = pyv.ErrIntegrator(sg_err_chain,
                                     sg_sens_data,
                                     sg_array.get_measurement_shape())
    sg_array.set_error_integrator(sg_error_int)

    # Now we visualise the strain gauge locations to make sure they are where
    # we expect them to be.
    pv_plot = pyv.plot_point_sensors_on_sim(sg_array,"strain_yy")
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]
    if save_figs:
        pv_plot.save_graphic(fig_save_path/(save_tag+"_sg_vis.svg"))
        pv_plot.screenshot(fig_save_path/(save_tag+"_sg_vis.png"))

    pv_plot.show()

    # We have both our sensor arrays so we will create and run our experiment.
    # Here we only have a single input simulation in our list and we only run
    # 100 simulated experiments as we are going to plot all simulated data
    # points on our traces. Note that if you are running more than 100
    # experiments here you will need to set the trace plots below to not show
    # all points on the graph.
    sim_list = [sim_data,]
    sensor_arrays = [tc_array,sg_array]
    exp_sim = pyv.ExperimentSimulator(sim_list,
                                      sensor_arrays,
                                      num_exp_per_sim=100)

    # We run our experiments and calculate summary statistics as in the previous
    # example
    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()

    # We print the lengths of our exp_data and exp_stats lists along with the
    # shape of the numpy arrays they contain so we can index into them easily.
    print(80*"=")
    print("exp_data and exp_stats are lists where the index is the sensor array")
    print("position in the list as field components are not consistent dims:")
    print(f"{len(exp_data)=}")
    print(f"{len(exp_stats)=}")
    print()
    print(80*"-")
    print("Thermal sensor array @ exp_data[0]")
    print(80*"-")
    print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_data[0].shape=}")
    print()
    print("Stats are calculated over all experiments (axis=1)")
    print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_stats[0].max.shape=}")
    print()
    print(80*"-")
    print("Mechanical sensor array @ exp_data[1]")
    print(80*"-")
    print("shape=(n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_data[1].shape=}")
    print()
    print("shape=(n_sims,n_sensors,n_field_comps,n_time_steps)")
    print(f"{exp_stats[1].max.shape=}")
    print(80*"=")

    # Finally, we are going to plot the simulated sensor traces but we are going
    # to control some of the plotting options using the options data class here.
    # We set the plot to show all simulated experiment data points and to plot
    # the median as the centre line and to fill between the min and max values.
    # Note that the default here is to plot the mean and fill between 3 times
    # the standard deviation.
    trace_opts = pyv.TraceOptsExperiment(plot_all_exp_points=True,
                                         centre=pyv.EExpVisCentre.MEDIAN,
                                         fill_between=pyv.EExpVisBounds.MINMAX)

    (fig,ax) = pyv.plot_exp_traces(exp_sim,
                                   component="temperature",
                                   sens_array_num=0,
                                   sim_num=0,
                                   trace_opts=trace_opts)
    if save_figs:
        fig.savefig(fig_save_path/(save_tag+"_tc_traces.png"),
                dpi=300, format='png', bbox_inches='tight')

    (fig,ax) = pyv.plot_exp_traces(exp_sim,
                                   component="strain_yy",
                                   sens_array_num=1,
                                   sim_num=0,
                                   trace_opts=trace_opts)
    if save_figs:
        fig.savefig(fig_save_path/(save_tag+"_sg_traces.png"),
                dpi=300, format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()