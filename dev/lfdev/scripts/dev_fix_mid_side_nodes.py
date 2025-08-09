'''
================================================================================
example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import pyvale.mooseherder as mh
import pyvale

def main() -> None:
    """Basics displacement sensors on a 2D plate with a hole
    ----------------------------------------------------------------------------
    """
    data_path = Path("dev/lfdev/meshtest/meshtest_higherorder_2d_out.e")
    sim_data = mh.ExodusLoader(data_path).read_all_sim_data()
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    #---------------------------------------------------------------------------
    print()
    print(80*"=")
    print(sim_data.coords)
    print()
    print(80*"=")
    print(sim_data.connect["connect1"])
    print()
    print(80*"=")
    print()

    for rr in range(sim_data.coords.shape[0]):
        print(f"Node {rr+1} = {sim_data.coords[rr,:]}")

    #---------------------------------------------------------------------------

    n_sens = (2,3,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    sens_data = pyvale.SensorData(positions=sens_pos)

    disp_sens_array = pyvale.SensorArrayFactory \
                            .disp_sensors_basic_errs(sim_data,
                                                     sens_data,
                                                     "displacement",
                                                     elem_dims=2)

    plot_field = 'disp_x'
    pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_array,plot_field)
    pv_plot.show(cpos="xy")


if __name__ == "__main__":
    main()