from pathlib import Path
from types import CellType
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale
import pyvista as pv


def main() -> None:
    data_path = Path("case21_m1_out.e") #pyvale.DataSet.thermal_3d_output_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_key = "disp_x"
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0

    n_sens = (1,4,1)
    x_lims = (12.5,12.5)
    y_lims = (0.0,33.0)
    z_lims = (0.0,12.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
    sens_data = pyvale.SensorData(positions=sens_pos)

    tc_array = pyvale.SensorArrayFactory \
        .thermocouples_basic_errs(sim_data,
                                  sens_data,
                                  field_key,
                                  spat_dims=3)

    vis_opts = pyvale.VisOptsSimSensors()
    pv_plot = pyvale.create_pv_plotter(vis_opts)

    sim_vis = tc_array.field.get_visualiser()
    sim_vis[field_key] = sim_data.node_vars[field_key][:,-1]
    # sim_vis is the mesh of interest.
    points = sim_vis.points

    surf = sim_vis.extract_surface()
    # tris_con = []
    # for c in surf.cell:
    #     assert c.type == pv.CellType.TRIANGLE





    pv_plot.add_mesh(sim_vis,
                     show_edges=vis_opts.show_edges,
                     lighting=False,
                     )

    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                                (-2.858, 13.189, 4.523),
                                (-0.215, 0.948, -0.233)]
    pv_plot.show()

if __name__ == "__main__":
    main()
