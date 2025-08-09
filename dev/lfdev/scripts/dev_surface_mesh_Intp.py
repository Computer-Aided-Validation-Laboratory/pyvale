
from pathlib import Path
import numpy as np
import pyvista as pv
import pyvale.mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path("src/pyvale/simcases/case21_out.e")
    sim_data = mh.ExodusLoader(data_path).read_all_sim_data()
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0

    components = ("disp_x","disp_y","disp_z")
    (pv_grid,pv_grid_vis) = pyvale.simdata_to_pyvista(sim_data,
                                                           components,
                                                           elem_dims=3)

    pv_surf = pv_grid.extract_surface()
    #pv_surf.plot()

    rad: float = 25/2
    height = 25
    points = np.array([[0,height/2,0],
                       [0,0,0],
                       [rad,height,0]])
    pv_points = pv.PolyData(points)
    sample_data = pv_points.sample(pv_surf)

    print()
    print(pv_grid.array_names)
    print(pv_surf.array_names)
    print()
    print(80*"=")
    print(sample_data["disp_y"])
    print(80*"=")
    print()
    print(sample_data)
    print()

    save_path = Path().cwd() / "test_output"
    if not save_path.is_dir():
        save_path.mkdir()
    save_file = save_path / "test_mesh.stl"
    pv_surf.save(save_file,binary=False)



    # n_sens = (1,3,1)
    # x_lims = (25.0,25.0)
    # y_lims = (0.0,25.0)
    # z_lims = (0.0,0.0)
    # sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    # sens_data = pyvale.SensorData(positions=sens_pos)

    # disp_sens_array = pyvale.SensorArrayFactory \
    #                         .disp_sensors_basic_errs(sim_data,
    #                                                  sens_data,
    #                                                  "displacement",
    #                                                  spat_dims=3)

    # plot_field = 'disp_y'
    # pv_plot = pyvale.plot_point_sensors_on_sim(disp_sens_array,plot_field)
    # pv_plot.show(cpos="xy")





if __name__ == "__main__":
    main()