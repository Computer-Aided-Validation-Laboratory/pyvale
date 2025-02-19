
"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import pyvale
import mooseherder as mh

# CYTHON MODULE
import camerac


def main() -> None:
    print()
    print(80*"-")
    print("CYTHON FILE:")
    print(camerac.__file__)
    print(80*"-")

    # 3D cylinder, mechanical, tets
    data_path = Path.home() / "pyvale" / "dev" / "lfdev" / "rastermeshbenchmarks"
    data_path = data_path / "case21_m5_out.e"

    field_key = "disp_y"
    components = ("disp_x","disp_y","disp_z")
    mesh_world: pyvale.CameraMeshData = pyvale.create_camera_mesh(data_path,
                                                                  field_key,
                                                                  components,
                                                                  spat_dim=3)

    print()
    print(80*"-")
    print("EXTRACTED SURFACE MESH DATA")
    print(f"{mesh_world.name=}")
    print()
    print(f"node_count =     {mesh_world.node_count}")
    print(f"elem_count =     {mesh_world.elem_count}")
    print(f"nodes_per_elem = {mesh_world.nodes_per_elem}")
    print()
    print(f"{mesh_world.coords.shape=}")
    print(f"{mesh_world.connectivity.shape=}")
    print()
    print(f"{mesh_world.elem_coords.shape=}")
    print()
    print(f"{mesh_world.field_by_node.shape=}")
    print(f"{mesh_world.field_by_elem.shape=}")
    print()
    print(f"{mesh_world.coord_bound_min=}")
    print(f"{mesh_world.coord_bound_max=}")
    print(f"{mesh_world.coord_cent=}")
    print(80*"-")

    #---------------------------------------------------------------------------
    (xx,yy,zz,ww) = (0,1,2,3)
    frame_to_render: int = -1

    pixels_num = np.array([2464,2056],dtype=np.int32)
    pixels_size = np.array([3.45e-3,3.45e-3],dtype=np.float64)
    focal_leng: float = 50.0
    sub_samp: int = 2
    border_factor: float = 1.05
    cam_rot = Rotation.from_euler("zyx",
                                  [0.0, 0.0, -60.0],
                                  degrees=True)


    cam_z_world = cam_rot.as_matrix()[:,-1]
    fov_leng = pyvale.fov_from_cam_rot_3d(cam_rot,mesh_world.coords)*border_factor
    image_dist = pyvale.image_dist_from_fov_3d(np.array([2464,2056]),
                                            np.array([3.45e-3,3.45e-3]),
                                            50.0,
                                            fov_len)

    roi_pos_world = mesh_world.coord_cent[:-1]
    cam_pos_world = roi_pos_world + np.max(image_dist)*cam_z_world

    cam_data = pyvale.CameraData(pixels_num=pixels_num,
                                 pixels_size=pixels_size,
                                 pos_world=cam_pos_world,
                                 rot_world=cam_rot,
                                 roi_cent_world=roi_pos_world,
                                 focal_length=focal_leng,
                                 sub_samp=sub_samp)

    field_to_render = np.ascontiguousarray(mesh_world.field_by_elem[:,:,frame_to_render])

    print()
    print(f"{cam_pos_world=}")
    print(f"{roi_pos_world=}")
    print()

    #---------------------------------------------------------------------------
    print()
    print(80*"=")
    print("RASTER ELEMENT LOOP START")
    print(80*"=")

    num_loops = 1
    loop_times = np.zeros((num_loops,),dtype=np.float64)

    print()
    print("Running raster loop.")
    for nn in range(num_loops):
        print(f"Running loop {nn}")
        loop_start = time.perf_counter()
        (image_subpx_buffer,
         depth_subpx_buffer) = camerac.raster_loop(field_to_render,
                                                   mesh_world.elem_coords,
                                                   cam_data.world_to_cam_mat,
                                                   cam_data.pixels_num,
                                                   cam_data.image_dims,
                                                   cam_data.image_dist,
                                                   cam_data.sub_samp)
        loop_times[nn] = time.perf_counter() - loop_start


    avg_start = time.perf_counter()
    image_avg_buffer = np.empty(cam_data.pixels_num,dtype=np.float64).T
    depth_avg_buffer = np.empty(cam_data.pixels_num,dtype=np.float64).T
    image_buffer = camerac.average_image(image_subpx_buffer,
                                         cam_data.sub_samp,
                                         image_avg_buffer)
    depth_buffer = camerac.average_image(depth_subpx_buffer,
                                         cam_data.sub_samp,
                                         depth_avg_buffer)
    avg_time = time.perf_counter() - avg_start

    print()
    print(80*"=")
    print("PERFORMANCE TIMERS")
    print(f"Avg. loop time = {np.mean(loop_times):.4f} seconds")
    print(f"Subpx avg. time = {avg_time:.6f} seconds")
    print(80*"=")

    #===========================================================================
    # PLOTTING
    plot_on = True
    depth_to_plot = np.copy(np.array(depth_buffer))
    depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    image_to_plot = np.copy(np.array(image_buffer))
    image_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
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
