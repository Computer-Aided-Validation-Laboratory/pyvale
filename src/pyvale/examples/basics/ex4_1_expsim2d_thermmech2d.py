# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Pyvale example: Multi-physics experiment simulation in 2D
--------------------------------------------------------------------------------
In previous examples we have built our virtual sensor array and used this to
run a single simulated experiment. However, we will generally want to run many
simulated experiments and perform statistical analysis on the results. In this
example we demonstrate how pyvale can be used to run a set of simulated
experiments with a series of sensor arrays one measuring temperature and the
other measuring strain. We also show how this analysis can be performed over a
set of input physics simulations.

Note that this tutorial assumes you are familiar with the use of pyvale for
scalar and tensor fields as described in the previous examples.

Test case: thermo-mechanical analysis of a 2D plate with a temperature gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    # Here we get a list of paths to a set of 3 simulations in this case the
    # simulation is a 2D plate with a heat flux on one edge and a heat transfer
    # coefficient on the other. The mechanical deformation is a result of
    # thermal expansion. The 3 simulation cases cover a nominal thermal
    # and a perturbation of +/-10%.
    data_paths = pyv.DataSet.thermomechanical_2d_experiment_paths()
    elem_dims: int = 2

    # We now loop over the paths and load each into a `SimData` object. We then
    # scale our length units to mm and append the simulation to a list which we
    # will use to perform our simulated experiments.
    disp_comps = ("disp_x","disp_y")
    sim_list = []
    for pp in data_paths:
        sim_data = mh.ExodusReader(pp).read_all_sim_data()
        sim_data = pyv.scale_length_units(scale=1000.0,
                                          sim_data=sim_data,
                                          disp_comps=disp_comps)
        sim_list.append(sim_data)


    # We will use the same sampling times for both the thermal and strain
    # sensor arrays as well as the same positions.
    sample_times = np.linspace(0.0,np.max(sim_data.time),50)


    # We place 4 thermal sensors along the mid line of the plate in the
    # direction of the temperature gradient.
    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    tc_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_sens_data = pyv.SensorData(positions=tc_sens_pos,
                                  sample_times=sample_times)

    # We use the sensor array factory to give use thermocouples with basic 2%
    # errors with uniform systematic error and normal random error. Note that
    # we need to provide a `SimData` object to create our sensor array but when
    # we run our experiment the field object that relies on this will switch the
    # sim data for the required simulation in our list.
    tc_field_name = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_list[0],
                                  tc_sens_data,
                                  elem_dims=elem_dims,
                                  field_name=tc_field_name,
                                  errs_pc=2.0)

    # We place 3 strain gauges along the direction of the temperature gradient.
    n_sens = (3,1,1)
    sg_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
    sg_sens_data = pyv.SensorData(positions=sg_sens_pos,
                                  sample_times=sample_times)

    # We use the factory to give us a basic strain gauge array as well.
    sg_field_name = "strain"
    sg_norm_comps = ("strain_xx","strain_yy")
    sg_dev_comps = ("strain_xy",)
    sg_array = pyv.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_list[0],
                                  sg_sens_data,
                                  elem_dims=elem_dims,
                                  field_name=sg_field_name,
                                  norm_comps=sg_norm_comps,
                                  dev_comps=sg_dev_comps,
                                  errs_pc=2.0)


    # Now we have our list of simulation and the two sensor arrays we want to
    # apply to the simulations. We create a list of our two sensor arrays and
    # use this to create an experiment simulator while specifying how many
    # simulate experiments we want to run per simulation and sensor array.
    sensor_arrays = [tc_array,sg_array]
    exp_sim = pyv.ExperimentSimulator(sim_list,
                                      sensor_arrays,
                                      num_exp_per_sim=1000)

    # We can now run our experiments for all our sensor arrays. We are returned
    # a list of numpy arrays. The index in the list corresponds to the position
    # of the sensor array in the list. So if we want our thermocouple results we
    # want exp_data[0] and for our strain gauges exp_data[1]. The numpy array
    # has the following shape:
    # (n_sims,n_exps,n_sensors,n_field_comps,n_time_steps)
    exp_data = exp_sim.run_experiments()

    # We can also calculate summary statistics for each sensor array which is
    # returned as a list where the position corresponds to the sensor array as
    # in our experimental data. The experiment stats object contains numpy
    # arrays for each statistic that is collapsed over the number of
    # experiments. The statistics we can acces include: mean, standard deviation
    # minimum, maximum, median, median absolute deviation and the 25% and 75%
    # quartiles. See the `ExperimentStats` data class for details.
    exp_stats = exp_sim.calc_stats()

    # We will index into and print the shape of our exp_data and exp_stats
    # lists to demonstrate how this works in practice:
    print(80*"=")
    print("exp_data and exp_stats are lists where the index is the sensor array")
    print("position in the list as field components are not consistent dims.\n")

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

    # We also have specific plotting tools which allow us to visualise the
    # uncertainty bounds for our sensor traces. The defaults plot options show
    # the mean sensor trace and uncertainty bounds of 3 times the stanard
    # deviation. In the next example we will see how to control these plots.
    # For now we will plot the temperature traces for the first simulation and
    # the strain traces for the third simulation in our list of SimData objects.
    (fig,ax) = pyv.plot_exp_traces(exp_sim,
                                   component="temperature",
                                   sens_array_num=0,
                                   sim_num=0)

    (fig,ax) = pyv.plot_exp_traces(exp_sim,
                                   component="strain_yy",
                                   sens_array_num=1,
                                   sim_num=2)
    plt.show()


if __name__ == "__main__":
    main()