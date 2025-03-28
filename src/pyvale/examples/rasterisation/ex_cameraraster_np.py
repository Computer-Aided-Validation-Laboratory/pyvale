"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

# TODO
# - Fix the image averaging function to use cython
# - Need to have deformable meshes in 2D and 3D
# - Need to allow rendering of a set of fields
# - Implement parallel rendering for image stacks or multiple fields
# - Saving of the rendered images for post processing or analysis
# - Collapse image display functions into visual to simplify code
#
# CAMERA:
# - Need option to work camera rotation based on a given position
#   - The z axis is easy as we can just do roi-cam_pos but what about x and y
#
# RENDER OPTIONS
# - Parallelisation on/off, number of threads
#   - Need to split work over: fields to render, cameras, time steps
# - Deformable mesh: on/off
#
# SCENE OBJECT:
# - Allow multiple objects in the scene with their own transformations
# - Allow multiple cameras in the scene


def main() -> None:
    """pyvale example: rasterisation field renderer
    ----------------------------------------------------------------------------
    - TODO
    """
    # This a path to an exodus *.e output file from MOOSE, this can be
    # replaced with a path to your own simulation file
    #sim_path = pyv.DataSet.render_mechanical_3d_path()
    sim_path = Path.cwd()/"src"/"pyvale"/"simcases"/"case25_out.e"
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()
    sim_data.coords = sim_data.coords*1000.0 # scale to mm

    # TODO: scale displacements to mm

    # Extracts the surface mesh from a full 3d simulation for rendering
    # render_mesh = pyv.create_render_mesh(sim_data,
    #                                     ("disp_y","disp_x"),
    #                                     sim_spat_dim=3,
    #                                     field_disp_keys=("disp_x","disp_y","disp_z"))

    render_mesh = pyv.create_render_mesh(sim_data,
                                        ("disp_y",),
                                        sim_spat_dim=3,
                                        field_disp_keys=None)


    print()
    print(80*"-")
    print("MESH DATA:")
    print(80*"-")
    print("connectivity.shape=(num_elems,num_nodes_per_elem)")
    print(f"{render_mesh.connectivity.shape=}")
    print()
    print("coords.shape=(num_nodes,coord[x,y,z])")
    print(f"{render_mesh.coords.shape=}")
    print()
    print("fields.shape=(num_coords,num_time_steps,num_components)")
    print(f"{render_mesh.fields_render.shape=}")
    if render_mesh.fields_disp is not None:
        print(f"{render_mesh.fields_disp.shape=}")
    print(80*"-")
    print()

    pixel_num = np.array((960,1280))
    pixel_size = np.array((5.3e-3,5.3e-3))
    focal_leng: float = 50
    cam_rot = Rotation.from_euler("zyx",(0.0,-30.0,0.0),degrees=True)
    fov_scale_factor: float = 1.1

    (roi_pos_world,
     cam_pos_world) = pyv.CameraTools.pos_fill_frame_from_rotation(
         coords_world=render_mesh.coords,
         pixel_num=pixel_num,
         pixel_size=pixel_size,
         focal_leng=focal_leng,
         cam_rot=cam_rot,
         frame_fill=fov_scale_factor,
     )

    cam_data = pyv.CameraData(
        pixels_num=pixel_num,
        pixels_size=pixel_size,
        pos_world=cam_pos_world,
        rot_world=cam_rot,
        roi_cent_world=roi_pos_world,
        focal_length=focal_leng,
        sub_samp=2,
        back_face_removal=True,
    )

    print(80*"-")
    print("CAMERA DATA:")
    print(80*"-")
    print(f"{roi_pos_world=}")
    print(f"{cam_pos_world=}")
    print()
    print("World to camera matrix:")
    print(cam_data.world_to_cam_mat)
    print(80*"-")

    print()
    print(80*"=")
    print("RASTER LOOP START")

    # LOOP INDICES
    frame = -1  # render the last frame
    comp = 0

    loop_times = []
    time_start_loop = time.perf_counter()

    by_connect = True

    #===========================================================================
    if by_connect:
        #---------------------------------------------------------------------------
        # LOOP: over frames, over components here = collapse to one loop
        #---------------------------------------------------------------------------

        # coords_raster.shape=(num_coords,coord[x,y,z,w])
        # connect_in_frame.shape=(num_elems_in_scene,nodes_per_elem)
        # elem_bound_box_inds.shape=(num_elems_in_scene,4[x_min,x_max,y_min,y_max])
        (coords_raster,
        connect_in_frame,
        elem_bound_box_inds,
        elem_areas) = pyv.RasteriserNP.setup_frame_by_connect(
            cam_data,
            render_mesh.coords,
            render_mesh.connectivity,
        )

        num_elems_in_image = connect_in_frame.shape[0]
        render_field_div_z = render_mesh.fields_render[:,frame,comp]*coords_raster[:,2]

        (image_buffer,
        depth_buffer) = pyv.RasteriserNP.raster_frame_by_connect(
                                                cam_data,
                                                connect_in_frame,
                                                coords_raster,
                                                elem_bound_box_inds,
                                                elem_areas,
                                                render_field_div_z)
        #===========================================================================
    else:
        (elem_raster_coords,
        elem_bound_box_inds,
        elem_areas,
        render_field_div_z) = pyv.RasteriserNP.setup_frame_by_elem(
                                                cam_data,
                                                render_mesh.coords,
                                                render_mesh.connectivity,
                                                render_mesh.fields_render[:,:,comp])

        field_frame_divide_z = np.ascontiguousarray(render_field_div_z[:,:,frame])

        (image_buffer,
        depth_buffer,
        num_elems_in_image) = pyv.RasteriserNP.raster_frame_by_elem(
                                                cam_data,
                                                elem_raster_coords,
                                                elem_bound_box_inds,
                                                elem_areas,
                                                field_frame_divide_z)

    #===========================================================================
    time_end_loop = time.perf_counter()
    loop_times.append(time_end_loop - time_start_loop)

    print("RASTER LOOP END")
    print(80*"=")
    print("PERFORMANCE")
    print(f"Elements in image: {num_elems_in_image}")
    print(f"Render time = {np.mean(loop_times):.6f} seconds")
    print(80*"=")

    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.nan
    image_to_plot = np.copy(image_buffer)
    image_to_plot[depth_buffer > 10*cam_data.image_dist] = np.nan
    if plot_on:
        plot_opts = pyv.PlotOptsGeneral()

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