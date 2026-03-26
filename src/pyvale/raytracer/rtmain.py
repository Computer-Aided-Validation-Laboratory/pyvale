# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
import numpy as np
from pyvale.raytracer.rtscene import Scene, RenderType, find_max_displacements

from pyvale.raytracer.rtmaincpp import cpp_render_scene # Import C++ backend

def render_scene(image_height: int, image_width: int, 
                 scene: Scene, antialiasing_samples: int, 
                 out_directory_path: Path, 
                 render_type = RenderType.DYNAMIC, frames_to_render: int = None,
                 texture_img:  np.ndarray = np.ones((2, 2))):
    '''Sets appropriate settings and passes the data to the C++ renderer.
        frames_to_render - For dynamic renders, this is the number of frames to render. Defaults to all timesteps we have data for. For static renders,
        this is the number of frame to render; defaults to the first one otherwise. Nb4 this could maybe be a tuple to specify the range instead?'''

    # Assign default values depending on the render type if target frame count was not specified
    if frames_to_render is None:
        if render_type == RenderType.STATIC:
            frames_to_render = 1
        elif render_type == RenderType.DYNAMIC:
            frames_to_render = scene.timestep_count

    # Sanity check for the values
    if frames_to_render <= scene.timestep_count:
        scene.clip_scene(frames_to_render, render_type)
        #max_displacement_per_step_array = find_max_displacements(scene, render_type) # Data for deciding if to update/rebuild TLAS/BLAS. Currently WIP and doesn't get passed
    else:
        print("Number of requested frames exceeds the number of timesteps with availabile data.")
        return

    if render_type == RenderType.DYNAMIC:
        scene.fill_empty_timesteps() # VERY important to avoid segfaults if there is missing timestep data for some meshes in the scene

    print(f"Materials: {scene.materials}")
    cpp_render_scene(image_height, image_width, antialiasing_samples, out_directory_path, 
                     scene.timestep_count, scene.coords_expanded, scene.face_colors, 
                     scene.materials, 
                     scene.camera_center, 
                     scene.pixel_00_center, scene.matrix_pixel_spacing,
                     texture_img)