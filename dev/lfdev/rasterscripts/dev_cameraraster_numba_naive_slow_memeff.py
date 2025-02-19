"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from dataclasses import dataclass, field
import time
from pathlib import Path
from multiprocessing.pool import ThreadPool, Pool

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import convolve2d
import numba

import mooseherder as mh
import pyvale


@numba.jit(nopython=True)
def meshgrid2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    x_grid = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    y_grid = np.empty(shape=(y.size, x.size), dtype=y.dtype)

    for ii in range(y.size):
        for jj in range(x.size):
            x_grid[ii,jj] = x[jj]
            y_grid[ii,jj] = y[ii]

    return (x_grid,y_grid)


@dataclass(slots=True)
class CameraRasterData:
    num_pixels: np.ndarray
    pixel_size: np.ndarray

    pos_world: np.ndarray
    rot_world: Rotation

    roi_center_world: np.ndarray

    focal_length: float = 50.0
    sub_samp: int = 2

    back_face_removal: bool = True

    sensor_size: np.ndarray = field(init=False)
    image_dims: np.ndarray = field(init=False)
    image_dist: float = field(init=False)
    cam_to_world_mat: np.ndarray = field(init=False)
    world_to_cam_mat: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.image_dist = np.linalg.norm(self.pos_world - self.roi_center_world)
        self.sensor_size = self.num_pixels*self.pixel_size
        self.image_dims = (self.image_dist
                           *self.sensor_size/self.focal_length)

        self.cam_to_world_mat = np.zeros((4,4))
        self.cam_to_world_mat[0:3,0:3] = self.rot_world.as_matrix()
        self.cam_to_world_mat[-1,-1] = 1.0
        self.cam_to_world_mat[0:3,-1] = self.pos_world
        self.world_to_cam_mat = np.linalg.inv(self.cam_to_world_mat)


@numba.jit(nopython=True)
def world_to_raster_coords(coords_world: np.ndarray,
                           world_to_cam_mat: np.ndarray,
                           image_dims: np.ndarray,
                           image_dist: float,
                           num_pixels: np.ndarray) -> np.ndarray:
    # Index notation for numpy array index interpretation
    xx: int = 0
    yy: int = 1
    zz: int = 2
    ww: int = 3

    # Project onto camera coords using matrix multiplication
    coords_raster = world_to_cam_mat @ coords_world

    # NOTE: w is not 1 when the matrix is a perspective projection! It is only 1
    # here when we have an affine transformation
    coords_raster[xx] = coords_raster[xx] / coords_raster[ww]
    coords_raster[yy] = coords_raster[yy] / coords_raster[ww]
    coords_raster[zz] = coords_raster[zz] / coords_raster[ww]

    # Coords Image: Perspective divide
    coords_raster[xx] = (image_dist * coords_raster[xx]
                        / -coords_raster[zz])
    coords_raster[yy] = (image_dist * coords_raster[yy]
                        / -coords_raster[zz])

    # Coords NDC: Convert to normalised device coords in the range [-1,1]
    coords_raster[xx] = 2*coords_raster[xx] / image_dims[xx]
    coords_raster[yy] = 2*coords_raster[yy] / image_dims[yy]

    # Coords Raster: Covert to pixel (raster) coords
    # Shape = ([X,Y,Z],num_nodes)
    coords_raster[xx] = (coords_raster[xx] + 1)/2 * num_pixels[xx]
    coords_raster[yy] = (1-coords_raster[yy])/2 * num_pixels[yy]
    coords_raster[zz] = -coords_raster[zz]

    return coords_raster


@numba.jit(nopython=True)
def edge_function(vert_a: np.ndarray,
                  vert_b: np.ndarray,
                  vert_c: np.ndarray) -> np.ndarray:
    xx: int = 0
    yy: int = 1
    edge_fun = ((vert_c[xx] - vert_a[xx]) * (vert_b[yy] - vert_a[yy])
              - (vert_c[yy] - vert_a[yy]) * (vert_b[xx] - vert_a[xx]))
    return edge_fun


def average_subpixel_image(subpx_image: np.ndarray,
                           subsample: int) -> np.ndarray:
    conv_mask = np.ones((subsample,subsample))/(subsample**2)
    if subsample > 1:
        subpx_image_conv = convolve2d(subpx_image,conv_mask,mode='same')
        avg_image = subpx_image_conv[round(subsample/2)-1::subsample,
                                     round(subsample/2)-1::subsample]
    else:
        subpx_image_conv = subpx_image
        avg_image = subpx_image

    return avg_image


#===============================================================================
# MAIN
def main() -> None:
    # 3D cylinder, mechanical, tets
    data_path = Path("dev/lfdev/rastermeshbenchmarks")
    data_path = data_path / "case21_m1_out.e"

    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    field_keys = tuple(sim_data.node_vars.keys())
    # Scale to mm to make 3D visualisation scaling easier
    sim_data.coords = sim_data.coords*1000.0


    field_key = "disp_y"
    components = ("disp_x","disp_y","disp_z")
    (pv_grid,_) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                components,
                                                spat_dim=3)
    pyvale.print_dimensions(sim_data)

    time_start_setup = time.perf_counter()

    pv_surf = pv_grid.extract_surface()
    faces = np.array(pv_surf.faces)

    first_elem_nodes_per_face = faces[0]
    nodes_per_face_vec = faces[0::(first_elem_nodes_per_face+1)]
    assert np.all(nodes_per_face_vec == first_elem_nodes_per_face), \
    "Not all elements in the simdata object have the same number of nodes per element"

    nodes_per_face = first_elem_nodes_per_face
    num_faces = int(faces.shape[0] / (nodes_per_face+1))

    # Reshape the faces table and slice off the first column which is just the
    # number of nodes per element and should be the same for all elements
    pv_connect = np.reshape(faces,(num_faces,nodes_per_face+1))
    pv_connect = pv_connect[:,1:].T
    pv_coords = np.array(pv_surf.points).T

    print()
    print(80*"-")
    print("EXTRACTED SURFACE MESH DATA")
    print(pv_surf)
    print()
    print("Attached array names:")
    print(pv_surf.array_names)
    print()
    print(f"{nodes_per_face=}")
    print(f"{num_faces=}")
    print()
    print("NOTE: shape needs to be coord/node_per_elem first.")
    print(f"{pv_coords.shape=}")
    print(f"{pv_connect.shape=}")
    print()
    print(f"{pv_surf[components[0]].shape=}")
    print()
    print(80*"-")

    #===========================================================================
    # Create Camera and World Parameters
    (xx,yy,zz,ww) = (0,1,2,3)

    #shape=(3,num_coords)
    coords_world = pv_coords
    coords_count = coords_world.shape[1]
    # shape=(4,num_nodes)
    coords_world= np.vstack((coords_world,np.ones((1,coords_count))))
    # shape=(nodes_per_elem,num_elems)
    connectivity = pv_connect
    elem_count = connectivity.shape[1]
    # shape=(num_nodes,num_time_steps)
    field_array = np.array(pv_surf[field_key])
    frame_to_render: int = -1

    rot_axis: str = "x"
    phi_y_degs: float = -45
    theta_x_degs: float = -45
    psi_z_degs: float = 0.0

    phi_y_rads: float = phi_y_degs * np.pi/180.0
    theta_x_rads: float = theta_x_degs * np.pi/180.0

    # Set this to 0.0 to get some of the plate outside the FOV
    roi_pos_world = np.mean(sim_data.coords,axis=0)

    # Number of divisions (subsamples) for each pixel for anti-aliasing
    sub_samp: int = 2

    cam_type = "Test"
    if cam_type == "AV507":
        cam_num_px = np.array([2464,2056],dtype=np.int32)
        pixel_size = np.array([3.45e-3,3.45e-3]) # in millimeters!
        focal_leng: float = 25.0

        imaging_rad: float = 150.0 # Not needed for camera data, just for cam pos below
    else:
        cam_num_px = np.array([510,260],dtype=np.int32)
        pixel_size = np.array([10.0e-3,10.0e-3])
        focal_leng: float = 25.0

        imaging_rad: float = 300.0 # Not needed for camera data, just for cam pos below

    if rot_axis == "y":
        cam_pos_world = np.array([roi_pos_world[xx] + imaging_rad*np.sin(phi_y_rads),
                                  roi_pos_world[yy],
                                  imaging_rad*np.cos(phi_y_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, phi_y_degs, 0], degrees=True)
    elif rot_axis == "x":
        cam_pos_world = np.array([roi_pos_world[xx] ,
                                  roi_pos_world[yy] - imaging_rad*np.sin(theta_x_rads),
                                  imaging_rad*np.cos(theta_x_rads)])

        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, theta_x_degs], degrees=True)

    else:
        cam_pos_world = np.array([roi_pos_world[xx],
                                  roi_pos_world[yy],
                                  imaging_rad])
        cam_rot = Rotation.from_euler("zyx", [psi_z_degs, 0, 0], degrees=True)

    #---------------------------------------------------------------------------
    # RASTERISATION START
    cam_data = CameraRasterData(num_pixels=cam_num_px,
                                pixel_size=pixel_size,
                                pos_world=cam_pos_world,
                                rot_world=cam_rot,
                                roi_center_world=roi_pos_world,
                                focal_length=focal_leng,
                                sub_samp=sub_samp)

    #---------------------------------------------------------------------------
    # SPLIT ELEMENT COORDS

    # shape=(coord[X,Y,Z,W],node_per_elem,elem_num)
    elem_world_coords = coords_world[:,connectivity]
    # shape=(nodes_per_elem,coord[X,Y,Z,W],elem_num)
    elem_world_coords = np.swapaxes(elem_world_coords,0,1)

    # shape(nodes_per_elem,elem_num)
    field_to_render = np.ascontiguousarray(field_array[connectivity,frame_to_render])

    # LOOP IMPLEMENTATION
    # elem_count = connectivity.shape[1]
    # nodes_per_elem = connectivity.shape[0]
    # elem_world_coords = np.empty((nodes_per_elem,4,elem_count),dtype=coords_world.dtype)
    # for ee in range(elem_count):
    #     connect = connectivity[:,ee]
    #     for nn in range(nodes_per_elem):
    #         elem_world_coords[nn,:,ee] = coords_world[:,connect[nn]]

    #---------------------------------------------------------------------------
    # RASTER LOOP START
    loop_start = time.perf_counter()
    xx: int = 0
    yy: int = 1
    zz: int = 2
    ww: int = 3

    elems_in_image: int = 0
    depth_buffer = 1e6*np.ones(cam_data.sub_samp*cam_data.num_pixels).T
    image_buffer = np.full(cam_data.sub_samp*cam_data.num_pixels,0.0).T

    print()
    print(80*"=")
    print("RASTER ELEMENT LOOP START")
    print(80*"=")
    #///////////////////////////////////////////////////////////////////////////
    for ee in range(elem_count):
        print(f"Processing element {ee} of {elem_count}")
        # shape=(coords[X,Y,Z,W],nodes_per_elem)
        nodes_world = np.ascontiguousarray(elem_world_coords[:,:,ee]).T

        # shape=(coords[X,Y,Z,W],nodes_per_elem)
        nodes_raster = world_to_raster_coords(nodes_world,
                                              cam_data.world_to_cam_mat,
                                              cam_data.image_dims,
                                              cam_data.image_dist,
                                              cam_data.num_pixels)

        # Bounding box for the element verticies
        x_min = np.min(nodes_raster[xx,:])
        x_max = np.max(nodes_raster[xx,:])
        y_min = np.min(nodes_raster[yy,:])
        y_max = np.max(nodes_raster[yy,:])

        # Cropping: if the element is outside the image we don't process it
        if ((x_min > cam_data.num_pixels[xx]-1) or (x_max < 0)
            or (y_min > cam_data.num_pixels[yy]-1) or (y_max < 0)):
            continue

        elems_in_image += 1

        xi_min = np.max((0,np.floor(x_min))).astype(np.int64)
        xi_max = np.min((cam_data.num_pixels[xx]-1, np.ceil(x_max))).astype(np.int64)
        yi_min = np.max((0,np.floor(y_min))).astype(np.int64)
        yi_max = np.min((cam_data.num_pixels[yy]-1, np.ceil(y_max))).astype(np.int64)

        # Pre-divide the z coord for interpolation
        nodes_raster[zz,:] = 1 / nodes_raster[zz,:]
        # shape=(nodes_per_elem,)
        nodes_field = np.ascontiguousarray(field_to_render[:,ee])
        # We have already pre-divided the z coord so multiplication does the divide
        nodes_field = nodes_field * nodes_raster[zz,:]

        elem_area = edge_function(nodes_raster[:,0],
                                  nodes_raster[:,1],
                                  nodes_raster[:,2])

        bound_coords_x = (np.arange(xi_min,xi_max,1/cam_data.sub_samp)
                        + 1/(2*cam_data.sub_samp))
        bound_coords_y = (np.arange(yi_min,yi_max,1/cam_data.sub_samp)
                        + 1/(2*cam_data.sub_samp))

        bound_inds_x = np.arange(cam_data.sub_samp*xi_min,
                                 cam_data.sub_samp*xi_max)
        bound_inds_y = np.arange(cam_data.sub_samp*yi_min,
                                 cam_data.sub_samp*yi_max)

        for jj in range(bound_coords_y.size):
            for ii in range(bound_coords_x.size):
                px_coord = np.array((bound_coords_x[ii],bound_coords_y[jj],0))

                weights = np.zeros(3)
                weights[0] = edge_function(nodes_raster[:,1],
                                           nodes_raster[:,2],
                                           px_coord)
                weights[1] = edge_function(nodes_raster[:,2],
                                           nodes_raster[:,0],
                                           px_coord)
                weights[2] = edge_function(nodes_raster[:,0],
                                           nodes_raster[:,1],
                                           px_coord)

                if np.all(weights > 0.0):
                    weights = weights / elem_area
                    px_coord_z = 1/(np.sum(weights[:]*nodes_raster[zz,:]))
                    px_field = np.dot(nodes_field,weights) * px_coord_z

                    if px_coord_z < depth_buffer[bound_inds_y[jj],bound_inds_x[ii]]:
                        depth_buffer[bound_inds_y[jj],bound_inds_x[ii]] = px_coord_z
                        image_buffer[bound_inds_y[jj],bound_inds_x[ii]] = px_field

    depth_buffer = average_subpixel_image(depth_buffer,cam_data.sub_samp)
    image_buffer = average_subpixel_image(image_buffer,cam_data.sub_samp)
    #///////////////////////////////////////////////////////////////////////////
    loop_time = time.perf_counter() - loop_start
    print()
    print(80*"=")
    print("RASTER LOOP END")
    print(80*"=")
    print()
    print(80*"=")
    print("PERFORMANCE TIMERS")
    print(f"Total elements:    {connectivity.shape[1]}")
    print(f"Elements in image: {elems_in_image}")
    print()
    print(f"Loop time = {loop_time} seconds")

    print(80*"=")

    #===========================================================================
    # REGRESSION TESTING FOR REFACTOR
    save_regression_test_arrays = False
    check_regression_test_arrays = False

    test_path = Path.cwd() / "tests" / "regression"
    test_image = test_path / "image_buffer_naive.npy"
    test_depth = test_path / "depth_buffer_naive.npy"

    if save_regression_test_arrays:
        np.save(test_image,image_buffer)
        np.save(test_depth,depth_buffer)

    if check_regression_test_arrays:
        check_image = np.load(test_image)
        check_depth = np.load(test_depth)

        print(80*"/")
        print("REGRESSION TEST:")
        print(f"Image buffer check: {np.allclose(image_buffer,check_image)}")
        print(f"Depth buffer check: {np.allclose(depth_buffer,check_depth)}")
        print(80*"/")

        assert np.allclose(image_buffer,check_image), "Image buffer regression test FAILURE."
        assert np.allclose(depth_buffer,check_depth), "Depth buffer regression test FAILURE."

    #===========================================================================
    # PLOTTING
    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    #depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    image_to_plot = np.copy(image_buffer)
    #image_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    if plot_on:
        plot_opts = pyvale.PlotOptsGeneral()

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(depth_to_plot,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title("Depth buffer",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        (fig, ax) = plt.subplots(figsize=plot_opts.single_fig_size_square,
                                layout='constrained')
        fig.set_dpi(plot_opts.resolution)
        cset = plt.imshow(image_to_plot,
                        cmap=plt.get_cmap(plot_opts.cmap_seq))
                        #origin='lower')
        ax.set_aspect('equal','box')
        fig.colorbar(cset)
        ax.set_title("Field Image",fontsize=plot_opts.font_head_size)
        ax.set_xlabel(r"x ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
        ax.set_ylabel(r"y ($px$)",
                    fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)

        plt.show()








if __name__ == "__main__":
    main()