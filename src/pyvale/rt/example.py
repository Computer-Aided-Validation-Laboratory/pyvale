from pathlib import Path
from types import CellType
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale
import pyvista as pv
import numpy as np

from quad import Tri
from hittable_list import HittableList
from material import Lambertian
from camera import Camera
from bvh import BVH_Node


def extract_tris() -> None:
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
    tris_con = []
    for c in surf.cell:
        assert c.type == pv.CellType.TRIANGLE
        o = c.points[0]
        u = c.points[1] - o
        v = c.points[2] - o
        tris_con.append([o,u,v])

    return tris_con



    pv_plot.add_mesh(sim_vis,
                     show_edges=vis_opts.show_edges,
                     lighting=False,
                     )

    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                                (-2.858, 13.189, 4.523),
                                (-0.215, 0.948, -0.233)]
    pv_plot.show()

def main():
    world: HittableList = HittableList()

    # Materials
    left_red     = Lambertian(np.array([1.0, 0.2, 0.2]))

    tris = extract_tris()
    for t in tris:
        world.add(Tri(t[0], t[1], t[2], left_red))
    
    bv = HittableList(BVH_Node(world._objects))
    bv.bbox

    cam: Camera = Camera()
    cam.image_width = 100
    cam.image_height = 80
    cam.samples_per_pixel = 5
    cam.max_depth = 5
    cam.vfov = 80
    cam.look_from = np.array([0,3,9])
    cam.look_at = np.array([0,0,0])

    cam.render(world)


if __name__ == "__main__":
    main()
