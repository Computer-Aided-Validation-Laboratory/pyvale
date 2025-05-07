# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import mooseherder as mh
import pyvale as pyv

# TODO
# - Fix the image averaging function to use cython
# - Saving of the rendered images for post processing or analysis
# - Collapse image display functions into visual to simplify code
#
# CAMERA:
# - Need option to work camera rotation based on a given position
#   - The z axis is easy as we can just do roi-cam_pos but what about x and y
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
    #sim_path = Path.home()/"pyvale"/"src"/"pyvale"/"simcases"/"case21_out.e"
    sim_path = pyv.DataSet.render_mechanical_3d_path()

    disp_comps = ("disp_x","disp_y","disp_z")

    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()

    # Scale m -> mm
    sim_data = pyv.scale_length_units(sim_data,disp_comps,1000.0)

    # Extracts the surface mesh from a full 3d simulation for rendering
    render_mesh = pyv.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

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
    cam_rot = Rotation.from_euler("zyx",(0.0,-30.0,-10.0),degrees=True)
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

    print(80*"-")
    total_frames = render_mesh.fields_render.shape[1]*render_mesh.fields_render.shape[2]
    print(f"Time steps to render: {render_mesh.fields_render.shape[1]}")
    print(f"Fields to render: {render_mesh.fields_render.shape[2]}")
    print(f"Total frames to render: {total_frames}")
    print(80*"-")

    print()
    print(80*"=")
    print("RASTER LOOP START")

    #save_path = Path.cwd()/"example_output"
    save_path = None
    static_mesh = False

    time_start_loop = time.perf_counter()
    if static_mesh:
        images = pyv.RasterNP.raster_static_mesh(
            cam_data,render_mesh,save_path,threads_num=8
        )
    else:
        time_start_loop = time.perf_counter()
        images = pyv.RasterNP.raster_deformed_mesh(
            cam_data,render_mesh,save_path,parallel=8
        )

    time_end_loop = time.perf_counter()
    render_time = time_end_loop - time_start_loop


    print("RASTER LOOP END")
    print(80*"=")
    print("PERFORMANCE")
    print(f"Total frames = {total_frames}")
    print(f"Total render time = {render_time:.4f} seconds")
    print(f"Time per frame = {(render_time/total_frames):.4f} seconds")
    print(80*"=")

    plot_on = True
    if plot_on:
        (fig,ax) = pyv.plot_field_image(images[:,:,-1,0],
                                        title_str="Disp. y, [mm]")
        plt.show()

if __name__ == "__main__":
    main()