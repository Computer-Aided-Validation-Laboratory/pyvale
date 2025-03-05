from pathlib import Path
from types import CellType
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale
import pyvista as pv
import numpy as np

from quad import Quad, Tri
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



    pv_plot.add_mesh(surf,
                     show_edges=vis_opts.show_edges,
                     lighting=False,
                     )

    pv_plot.camera_position = [(59.354, 43.428, 69.946),
                                (-2.858, 13.189, 4.523),
                                (-0.215, 0.948, -0.233)]
    pv_plot.show()

def quadfoo():
    lin_pts = np.array(
        [
            [-1, -1, -1],  # point 0
            [1, -1, -1],  # point 1
            [1, 1, -1],  # point 2
            [-1, 1, -1],  # point 3
            [-1, -1, 1],  # point 4
            [1, -1, 1],  # point 5
            [1, 1, 1],  # point 6
            [-1, 1, 1],  # point 7
        ],
        np.double,
    )

    # these are the "midside" points of a quad cell.  See the definition of a
    # vtkQuadraticHexahedron at:
    # https://vtk.org/doc/nightly/html/classvtkQuadraticHexahedron.html
    quad_pts = np.array(
        [
            (lin_pts[1] + lin_pts[0]) / 2,  # between point 0 and 1
            (lin_pts[1] + lin_pts[2]) / 2,  # between point 1 and 2
            (lin_pts[2] + lin_pts[3]) / 2,  # and so on...
            (lin_pts[3] + lin_pts[0]) / 2,
            (lin_pts[4] + lin_pts[5]) / 2,
            (lin_pts[5] + lin_pts[6]) / 2,
            (lin_pts[6] + lin_pts[7]) / 2,
            (lin_pts[7] + lin_pts[4]) / 2,
            (lin_pts[0] + lin_pts[4]) / 2,
            (lin_pts[1] + lin_pts[5]) / 2,
            (lin_pts[2] + lin_pts[6]) / 2,
            (lin_pts[3] + lin_pts[7]) / 2,
        ],
    )

    # introduce a minor variation to the location of the mid-side points
    # seed the random numbers for reproducibility
    rng = np.random.default_rng(seed=0)
    quad_pts += rng.random(quad_pts.shape) * 0.3
    pts = np.vstack((lin_pts, quad_pts))

    # create the grid
    cells = np.hstack((20, np.arange(20))).astype(np.int64, copy=False)
    celltypes = np.array([pv.CellType.QUADRATIC_HEXAHEDRON])
    grid = pv.UnstructuredGrid(cells, celltypes, pts)

    # finally, extract the surface and plot it
    surf = grid.extract_surface()
    tris_con = []
    for c in surf.cell:
        assert c.type == pv.CellType.TRIANGLE
        o = c.points[0]
        u = c.points[1] - o
        v = c.points[2] - o
        tris_con.append([o,u,v])

    return tris_con
 
    surf.plot(show_scalar_bar=False)

def quadish():
    world: HittableList = HittableList()

    # Materials
    left_red = Lambertian.from_colour(np.array([1.0, 0.2, 0.2]))

    tris = extract_tris()
    for t in tris:
        world.add(Tri(t[0], t[1], t[2], left_red))
    
    bv = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 150
    cam.image_height = 200
    cam.samples_per_pixel = 1
    cam.max_depth = 3
    cam.vfov = 90
    cam.look_from = np.array([0,10,45])
    cam.look_at = np.array([0,0,0])

    cam.render(bv)

def squa():
    # camera = pv.Camera()
    # camera.position = (0.0, 0.0, 10)
    # camera.focal_point = (0, 0, 0)
    # # axes = pv.Axes(show_actor=True, actor_scale=2.0, line_width=5)
    # # axes.origin = (0.5, 0.5, 0.5)
    # grid = pv.Cube()
    # grid = grid.rotate_y(65, inplace=False)
    # p = pv.Plotter()
    # p.camera = camera
    # p.add_mesh(grid)
    # p.show()

    grid = pv.Cube()
    # grid = grid.rotate_y(60, inplace=False)
    grid = grid.triangulate()

    surf = grid.extract_surface()
    tri_con = []
    for c in surf.cell:
        assert c.type == pv.CellType.TRIANGLE
        o = c.points[0]
        u = c.points[1] - o
        v = c.points[2] - o
        tri_con.append([o,u,v])

    # grid = pv.Cube((1, 1,-1))
    # grid = grid.triangulate()

    # surf = grid.extract_surface()
    # for c in surf.cell:
    #     assert c.type == pv.CellType.TRIANGLE
    #     o = c.points[0]
    #     u = c.points[1] - o
    #     v = c.points[2] - o
    #     tri_con.append([o,u,v])
    
    # print(tris_con)
    world: HittableList = HittableList()

    i = 0
    for t in tri_con:
        world.add(Tri(t[0], t[1], t[2], Lambertian(np.array([(i % 5) / 5, (i % 10) / 10, (i % 20) / 20])) ))
        i += 1    

    world = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 600
    cam.image_height = 600
    cam.samples_per_pixel = 1
    cam.max_depth = 3
    cam.vfov = 40
    cam.look_from = np.array([-2,2,4])
    cam.look_at = np.array([0,0,0])

    cam.render(world)

def quad():
    grid = pv.Cube((-1,-1,-1))

    surf = grid.extract_surface()
    quad_con = []
    for c in surf.cell:
        assert c.type == pv.CellType.QUAD
        o = c.points[0]
        u = c.points[1] - o
        v = c.points[3] - o
        quad_con.append([o,u,v])
    
    grid = pv.Cube((1,1,-1))
    surf = grid.extract_surface()
    for c in surf.cell:
        assert c.type == pv.CellType.QUAD
        o = c.points[0]
        u = c.points[1] - o
        v = c.points[3] - o
        quad_con.append([o,u,v])

    # print(tris_con)
    world: HittableList = HittableList()

    # Materials
    left_red = Lambertian(np.array([1.0, 0.2, 0.2]))

    i = 0
    for t in quad_con:
        world.add(Quad(t[0], t[1], t[2], Lambertian(np.array([(i % 5) / 5, (i % 10) / 10, (i % 20) / 20])) ))
        i += 1
    
    bv = HittableList(BVH_Node(world._objects))

    cam: Camera = Camera()
    cam.image_width = 600
    cam.image_height = 600
    cam.samples_per_pixel = 2
    cam.max_depth = 3
    cam.vfov = 80
    cam.look_from = np.array([0,0,3])
    cam.look_at = np.array([0,0,0])

    cam.render(bv)

if __name__ == "__main__":
    
    quadish()
