"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import pyvale

def main() -> None:
    """pyvale example: rasterisation field renderer
    ----------------------------------------------------------------------------
    - TODO
    """

    # This is just a path to an exodus output file from MOOSE, this can be
    # replaced with your own simulation file
    data_path = pyvale.DataSet.render_mechanical_3d_path()

    # Load the mesh to render and extract the surface mesh from the full 3d sim
    mesh_data = pyvale.create_camera_mesh(data_path,
                                        "disp_y",
                                        ("disp_x","disp_y","disp_z"),
                                        spat_dim=3)

    print()
    print(80*"-")
    print("MESH DATA:")
    print(80*"-")
    print("connectivity.shape=(num_elems,num_nodes_per_elem)")
    print(f"{mesh_data.connectivity.shape=}")
    print("coords.shape=(num_nodes,coord[x,y,z])")
    print(f"{mesh_data.coords.shape=}")
    print("")
    print(f"{mesh_data.fields_by_node.shape=}")
    print(80*"-")
    print()

    #
    pixels_num = np.array((960,1280))
    pixels_size = np.array((5.3e-3,5.3e-3))
    focal_leng: float = 50

    cam_rot = Rotation.from_euler("zyx",(0.0,-30.0,0.0),degrees=True)

    fov_leng = pyvale.CameraTools.fov_from_cam_rot_3d(
        cam_rot=cam_rot,
        coords_world=mesh_data.coords,
    )

    # Scale the field of view to make sure that the mesh is fully in frame
    fov_leng = 1.1*fov_leng

    image_dist = pyvale.CameraTools.image_dist_from_fov_3d(
        num_pixels=pixels_num,
        pixel_size=pixels_size,
        focal_leng=focal_leng,
        fov_leng=fov_leng,
    )

    roi_pos_world = mesh_data.coord_cent[:-1]
    cam_z_dir_world = cam_rot.as_matrix()[:,-1]
    cam_pos_world = (roi_pos_world + np.max(image_dist)*cam_z_dir_world)

    cam_data = pyvale.CameraData(
        pixels_num=pixels_num,
        pixels_size=pixels_size,
        pos_world=cam_pos_world,
        rot_world=cam_rot,
        roi_cent_world=roi_pos_world,
        focal_length=focal_leng,
        sub_samp=2,
        back_face_removal=False,
    )

    print(80*"-")
    print("CAMERA DATA:")
    print(80*"-")
    print(f"{fov_leng=}")
    print(f"{image_dist=}\n")
    print("World to camera matrix:")
    print(cam_data.world_to_cam_mat)
    print(80*"-")

    print()
    print(80*"=")
    print("RASTER LOOP START")


    frame = -1  # render the last frame
    loop_times = []
    time_start_loop = time.perf_counter()

    (elem_raster_coords,
    elem_bound_box_inds,
    elem_areas,
    field_divide_z) = pyvale.RasteriserNP.raster_setup(
                                            cam_data,
                                            mesh_data.coords,
                                            mesh_data.connectivity,
                                            mesh_data.fields_by_node[:,:,1])

    field_frame_divide_z = np.ascontiguousarray(field_divide_z[:,:,frame])

    (image_buffer,
    depth_buffer,
    num_elems_in_image) = pyvale.RasteriserNP.raster_loop(
                                            cam_data,
                                            elem_raster_coords,
                                            elem_bound_box_inds,
                                            elem_areas,
                                            field_frame_divide_z)
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