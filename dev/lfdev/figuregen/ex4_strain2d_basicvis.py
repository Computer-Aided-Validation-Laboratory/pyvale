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
    data_path = Path('simcases/case17/case17_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0 # type: ignore

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    straingauge_array = pyvale.SensorArrayFactory \
                            .strain_gauges_basic_errs(sim_data,
                                                     sens_pos,
                                                     "strain",
                                                     elem_dims=2)

    plot_field = 'strain_yy'
    pv_plot = pyvale.plot_point_sensors_on_sim(straingauge_array,plot_field)
    pv_plot.camera_position = [(214.08261967353556, 46.15582361499647, 308.687529820126),
                            (49.5, 74.5, 0.0),
                            (-0.04768267074047773, -0.996673492281819, -0.06609321216144791)]

    save_render = Path('dev/lfdev/figuregen/strain2d_sensvis.svg')
    pv_plot.save_graphic(save_render) # only for .svg .eps .ps .pdf .tex
    pv_plot.screenshot(save_render.with_suffix('.png'))

    #pv_plot.show(cpos="xy")

    (fig,_) = pyvale.plot_time_traces(straingauge_array,'strain_xx')
    save_traces = Path('dev/lfdev/figuregen/strain2d_traces_exx.png')
    fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')

    (fig,_) =pyvale.plot_time_traces(straingauge_array,'strain_yy')
    save_traces = Path('dev/lfdev/figuregen/strain2d_traces_eyy.png')
    fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')

    (fig,_) =pyvale.plot_time_traces(straingauge_array,'strain_xy')
    save_traces = Path('dev/lfdev/figuregen/strain2d_traces_exy.png')
    fig.savefig(save_traces, dpi=300, format='png', bbox_inches='tight')

    #plt.show()


if __name__ == "__main__":
    main()