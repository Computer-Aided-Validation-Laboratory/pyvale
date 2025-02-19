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

def main() -> None:
    case_list = ib.load_case_list()
    case_tag = case_list[0]
    (case_ident,case_mesh,cam_data) = ib.load_benchmark_by_tag(case_tag)

    print()
    print(80*"-")
    print("BENCHMARK CASE:")
    print(f"{case_ident=}")
    print(80*"-")
    print(f"{case_mesh.connectivity.shape=}")
    print(f"{case_mesh.coords.shape=}")
    print(f"{case_mesh.field_by_node.shape=}")
    print(80*"-")


    print()
    print(80*"=")
    print("RENDER LOOP START")
    print(80*"=")

    num_raster_loops: int = 1
    frame = -1  # render the last frame
    loop_times = []

    for ll in range(num_raster_loops):
        print(f"Running render loop {ll+1}")
        time_start_loop = time.perf_counter()

        (elem_raster_coords,
        elem_bound_box_inds,
        elem_areas,
        field_divide_z) = pyvale.RasteriserNP.raster_setup(
                                                cam_data,
                                                case_mesh.coords,
                                                case_mesh.connectivity,
                                                case_mesh.field_by_node)

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


    print()
    print("RASTER LOOP END")
    print(80*"=")
    print("PERFORMANCE TIMERS")
    print(f"Elements in image: {num_elems_in_image}")
    print()
    for ll in range(num_raster_loops):
        print(f"Render loop {ll+1} time = {loop_times[ll]:.6f} seconds")
    print()
    print(f"Avg. render time = {np.mean(loop_times):.6f} seconds")
    print(80*"=")


    plot_on = True
    depth_to_plot = np.copy(depth_buffer)
    depth_to_plot[depth_buffer > 10*cam_data.image_dist] = np.NaN
    image_to_plot = np.copy(image_buffer)
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