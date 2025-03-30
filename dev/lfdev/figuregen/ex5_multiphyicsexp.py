'''
================================================================================
example: strain gauges on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale

def main() -> None:
    #===========================================================================
    # Load Simulations as mooseherder.SimData objects
    #base_path = Path("src/data")
    base_path = Path('src/data/')
    data_paths = [base_path / 'case16_out.e',]
    spat_dims = 3

    sim_list = []
    for pp in data_paths:
        sim_data = mh.ExodusReader(pp).read_all_sim_data()
        # Scale to mm to make 3D visualisation scaling easier
        sim_data.coords = sim_data.coords*1000.0 # type: ignore
        sim_list.append(sim_data)

    #===========================================================================
    # Creaet pyvale sensor arrays for thermal and mechanical data
    err_pc = 2.0
    sim_data = sim_list[0]

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_field = 'temperature'
    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_pos,
                                  tc_field,
                                  spat_dims=spat_dims,
                                  sample_times=None,
                                  errs_pc=err_pc)

    sg_field = 'strain'
    sg_array = pyvale.SensorArrayFactory \
        .strain_gauges_basic_errs(sim_data,
                                  sens_pos,
                                  sg_field,
                                  spat_dims=spat_dims,
                                  sample_times=None,
                                  errs_pc=err_pc)

    sensor_arrays = [tc_array,sg_array]

    #===========================================================================
    # Create and run the simulated experiment
    exp_sim = pyvale.ExperimentSimulator(sim_list,
                                        sensor_arrays,
                                        num_exp_per_sim=100)

    exp_data = exp_sim.run_experiments()
    exp_stats = exp_sim.calc_stats()

    #===========================================================================
    # VISUALISE RESULTS
    save_traces = True
    save_path = Path('dev/lfdev/figuregen')

    trace_opts = pyvale.TraceOptsExperiment()

    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                      component="temperature",
                                      sens_array_num=0,
                                      sim_num=0,
                                      trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_temp_fill.png',
                    dpi=300, format='png', bbox_inches='tight')


    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="strain_xx",
                                    sens_array_num=1,
                                    sim_num=0,
                                    trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_strain_fill.png',
                    dpi=300, format='png', bbox_inches='tight')

    #---------------------------------------------------------------------------
    trace_opts.plot_all_exp_points = True
    trace_opts.fill_between = None

    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="temperature",
                                    sens_array_num=0,
                                    sim_num=0,
                                    trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_temp_dots.png',
                    dpi=300, format='png', bbox_inches='tight')


    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="strain_xx",
                                    sens_array_num=1,
                                    sim_num=0,
                                    trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_strain_dots.png',
                    dpi=300, format='png', bbox_inches='tight')

    #---------------------------------------------------------------------------
    trace_opts.plot_all_exp_points = True
    trace_opts.fill_between = "3std"

    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="temperature",
                                    sens_array_num=0,
                                    sim_num=0,
                                    trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_temp_both.png',
                    dpi=300, format='png', bbox_inches='tight')


    (fig,ax) = pyvale.plot_exp_traces(exp_sim,
                                    component="strain_xx",
                                    sens_array_num=1,
                                    sim_num=0,
                                    trace_opts=trace_opts)
    if save_traces:
        fig.savefig(save_path / 'ex5_trace_strain_both.png',
                    dpi=300, format='png', bbox_inches='tight')


    #---------------------------------------------------------------------------
    if not save_traces:
        plt.show()


if __name__ == "__main__":
    main()