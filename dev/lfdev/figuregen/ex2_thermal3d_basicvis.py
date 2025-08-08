'''
================================================================================
Example: 3d thermocouples on a monoblock

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import matplotlib.pyplot as plt
import pyvale.mooseherder as mh
import pyvale


def main() -> None:
    """Basics
    """
    data_path = Path('data/examplesims/monoblock_3d_thermal_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_name = list(sim_data.node_vars.keys())[0] # type: ignore
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    n_sens = (1,4,1)
    x_lims = (11.5,11.5)
    y_lims = (-11.5,19.5)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_pos,
                                  field_name,
                                  elem_dims=3)

    measurements = tc_array.get_measurements()
    print(f'\nMeasurements for sensor at top of block:\n{measurements[-1,0,:]}\n')

    pv_plot = pyvale.plot_point_sensors_on_sim(tc_array,field_name)

    # Set this to 'interactive' to get an interactive 3D plot of the simulation
    # and labelled sensor locations, set to 'save_fig' to create a vector
    # graphic using a specified camera position.
    pv_plot_mode = 'save_fig'

    if pv_plot_mode == 'interactive':
        pv_plot.camera_position = [(52.198, 26.042, 60.099),
                                    (0.0, 4.0, 5.5),
                                    (-0.190, 0.960, -0.206)]
        pv_plot.show(cpos="xy")

        print(80*"=")
        print('Camera positions = ')
        print(pv_plot.camera_position)
        print(80*"="+"\n")

    if pv_plot_mode == 'save_fig':
        # Determined manually by moving camera and then dumping camera position
        # to console after window close - see 'interactive above'
        pv_plot.camera_position = [(52.198, 26.042, 60.099),
                                    (0.0, 4.0, 5.5),
                                    (-0.190, 0.960, -0.206)]
        save_render = Path('src/examples/monoblock_thermal_sim_view.svg')
        pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
        pv_plot.screenshot(save_render.with_suffix('.png'))

    # Set this to 'interactive' to get a matplotlib.pyplot with the sensor
    # traces plotted over time. Set to 'save_fig' to save an image of the plot
    # to file.
    trace_plot_mode = 'save_fig'

    (fig,_) = pyvale.plot_time_traces(tc_array,field_name)

    if trace_plot_mode == 'interactive':
        plt.show()
    if trace_plot_mode == 'save_fig':
        save_traces = Path('src/examples/monoblock_thermal_traces.png')
        fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')


if __name__ == '__main__':
    main()
