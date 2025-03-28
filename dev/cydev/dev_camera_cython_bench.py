
"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import imagebenchmarks as ib
import pyvale

# CYTHON MODULE
import camerac


def main() -> None:
    print()
    print(80*"-")
    print("CYTHON FILE:")
    print(camerac.__file__)
    print(80*"-")

    return

    case_list = ib.load_case_list()
    case_tag = case_list[17]
    (case_ident,case_mesh,cam_data) = ib.load_benchmark_by_tag(case_tag)

    print()
    print(80*"-")
    print("BENCHMARK CASE:")
    print(f"{case_ident=}")
    print(80*"-")
    print(f"{case_mesh.connectivity.shape=}")
    print(f"{case_mesh.coords.shape=}")
    print(f"{case_mesh.fields_by_node.shape=}")
    print(80*"-")

    (elem_world_coords,
     field_to_render) = pyvale.slice_mesh_data_by_elem(case_mesh.coords,
                                                case_mesh.connectivity,
                                                case_mesh.fields_by_node[:,:,1])
    frame_to_render = np.ascontiguousarray(field_to_render[:,:,-1])

    print()
    print(f"{elem_world_coords.shape=}")
    print(f"{field_to_render.shape=}")
    print(f"{frame_to_render.shape=}")


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
         depth_subpx_buffer,
         elems_in_image) = camerac.raster_loop(frame_to_render,
                                                   elem_world_coords,
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
    print(f"Elements in image = {elems_in_image}")
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
