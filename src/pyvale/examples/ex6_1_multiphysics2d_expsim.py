# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: thermo-mechanical multi-physics sensors on a 2D plate
    ----------------------------------------------------------------------------
    - Demonstrates the experiment module for running many monte-carlo style
      experiments and statistically analysing the results.
    """
    # Load Simulations as mooseherder.SimData objects
    #base_path = Path("src/pyvale/data")
    data_paths = pyv.DataSet.thermomechanical_2d_experiment_paths()

    sim_list = []
    for pp in data_paths:
        sim_data = mh.ExodusReader(pp).read_all_sim_data()
        # Scale to mm to make 3D visualisation scaling easier
        sim_data.coords = sim_data.coords*1000.0 # type: ignore
        sim_list.append(sim_data)

    #===========================================================================
    # Create pyvale sensor arrays for thermal and mechanical data
    sim_data = sim_list[0]

    n_sens = (4,1,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,50.0)
    z_lims = (0.0,0.0)
    sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    sens_data = pyv.SensorData(positions=sens_pos,
                                  sample_times=sample_times)

    tc_field = 'temperature'
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  tc_field,
                                  spat_dims=2,
                                  errs_pc=1.0)

    sg_field = 'strain'
    sg_array = pyv.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_data,
                                  sens_data,
                                  sg_field,
                                  spat_dims=2,
                                  errs_pc=1.0)

    sensor_arrays = [tc_array,sg_array]

    #===========================================================================
    # Create and run the simulated experiment
    exp_sim = pyv.ExperimentSimulator(sim_list,
                                         sensor_arrays,
                                         num_exp_per_sim=1000)

    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()

    #===========================================================================
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

    #===========================================================================
    # VISUALISE RESULTS
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