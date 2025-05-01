# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

def main() -> None:
    """pyvale example: 3D thermo-mechanical with thermocouples & strain gauges
    ----------------------------------------------------------------------------
    """
    # Load Simulations as mooseherder.SimData objects
    sim_path = pyv.DataSet.thermomechanical_3d_path()
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()

    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0
    pyv.print_dimensions(sim_data)

    sim_list = [sim_data]

    use_sim_time = True
    if use_sim_time:
        sample_times = None
    else:
        sample_times = np.linspace(0.0,np.max(sim_data.time),50)

    save_figs = False
    save_tag = "thermomech3d"
    fig_save_path = Path.cwd()/"images"
    if not fig_save_path.is_dir():
        fig_save_path.mkdir(parents=True, exist_ok=True)

    #---------------------------------------------------------------------------
    # Create the thermocouple array
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    n_sens = (1,4,1)
    tc_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_sens_data = pyv.SensorData(positions=tc_sens_pos,
                                  sample_times=sample_times)

    tc_field = "temperature"
    tc_array = pyv.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  tc_sens_data,
                                  tc_field,
                                  spat_dims=3,
                                  errs_pc=1.0)

    # Visualise the thermocouple locations:
    pv_plot = pyv.plot_point_sensors_on_sim(tc_array,"temperature")
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]
    if save_figs:
        pv_plot.save_graphic(fig_save_path/(save_tag+"_tc_vis.svg"))
    pv_plot.show()

    #---------------------------------------------------------------------------
    # Create the strain gauge array
    x_lims = (9.4,9.4)
    y_lims = (0.0,33.0)
    z_lims = (12.0,12.0)
    n_sens = (1,4,1)
    sg_sens_pos = pyv.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sg_sens_data = pyv.SensorData(positions=sg_sens_pos,
                                  sample_times=sample_times)


    sg_field = "strain"
    sg_array = pyv.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_data,
                                  sg_sens_data,
                                  sg_field,
                                  spat_dims=3,
                                  errs_pc=1.0)

    # Visualise the strain gauge locations:
    pv_plot = pyv.plot_point_sensors_on_sim(sg_array,"strain_yy")
    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                               (-2.858, 13.189, 4.523),
                               (-0.215, 0.948, -0.233)]
    if save_figs:
        pv_plot.save_graphic(fig_save_path/(save_tag+"_sg_vis.svg"))
    pv_plot.show()

    #---------------------------------------------------------------------------
    # Create and run the simulated experiment
    sensor_arrays = [tc_array,sg_array]

    exp_sim = pyv.ExperimentSimulator(sim_list,
                                      sensor_arrays,
                                      num_exp_per_sim=100)

    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()

    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    # Visualise all sensor traces over all experiments
    trace_opts = pyv.TraceOptsExperiment(plot_all_exp_points=True)

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