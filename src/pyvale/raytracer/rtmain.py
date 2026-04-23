# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
from pyvale.raytracer.rtscene import Scene, RenderType, find_max_displacements
from pyvale.raytracer.rtmesh import SurfType, ElementNodeCount, RTMesh

from pyvale.raytracer.rtmaincpp import cpp_render_scene # Import C++ backend

def check_uniform_surfaces(scene: Scene):
    surfaces_detected = set(scene.surface_types) # Remove duplicates
    # RTMesh can't be added without surface type, so omit checking if all of them have it
    if len(surfaces_detected) == 1:
        return True # All surfaces are the same
    else:
        return False # Mixture of solid colors and textures

def check_uniform_elements(scene: Scene):
    elements_detected = set(scene.nodes_per_element) # Remove duplicates
    if len(elements_detected) == 1:
        return True # All element types are the same
    else:
        return False

def render_scene(image_height: int,
                 image_width: int,
                 scene: Scene,
                 antialiasing_samples: int,
                 out_directory_path: Path,
                 render_type = RenderType.DYNAMIC,
                 frames_to_render: int = None):
    '''Sets appropriate settings and passes the data to the C++ renderer.
        frames_to_render - For dynamic renders, this is the number of frames to render. Defaults to all timesteps we have data for. For static renders,
        this is the number of frame to render; defaults to the first one otherwise. Nb4 this could maybe be a tuple to specify the range instead?'''

    # Assign default values depending on the render type if target frame count was not specified
    if frames_to_render is None:
        if render_type == RenderType.STATIC:
            frames_to_render = 1
        elif render_type == RenderType.DYNAMIC:
            frames_to_render = scene.timestep_count

     # Check if there are meshes in the scene and if there are, check their surfaces and element types
    if len(scene.scene_connectivity) == 0:
        print("Scene contains no meshes.")
    else:
        uniform_surfaces = check_uniform_surfaces(scene)
        uniform_elements = check_uniform_elements(scene)

    # Sanity check for the values
    if frames_to_render <= scene.timestep_count:
        scene.clip_scene(frames_to_render, render_type)
        #max_displacement_per_step_array = find_max_displacements(scene, render_type) # Data for deciding if to update/rebuild TLAS/BLAS. Currently WIP and doesn't get passed
    else:
        raise ValueError("Number of requested frames exceeds the number of timesteps with availabile data.")

    if render_type == RenderType.DYNAMIC:
        scene.fill_empty_timesteps() # VERY important to avoid segfaults if there is missing timestep data for some meshes in the scene

    # Select appropriate rendering function based on these booleans to minimize branching in backend rendered if possible
    # For now don't and just use suboptimal implementation with branching
    if uniform_surfaces and uniform_elements:
       # Best case - everything is the same
        pass
    elif uniform_surfaces and not uniform_elements:
        # One surface type, but different element types
        pass
    elif not uniform_surfaces and uniform_elements:
        # One element type, but different surface types
        pass
    else:
        # Worst case: mixture of different surface and element types
        pass

    # For now use the general function with branching in it
    #cpp_render_scene(image_height, image_width, antialiasing_samples, out_directory_path, scene.timestep_count, scene.coords_expanded, scene.face_colors, scene.camera_center, scene.pixel_00_center, scene.matrix_pixel_spacing)

    cpp_render_scene(image_height, image_width, antialiasing_samples, out_directory_path, scene.timestep_count, scene.camera_center, scene.pixel_00_center, scene.matrix_pixel_spacing, scene.coords_expanded, scene.face_colors, scene.uvs, scene.textures, scene.surface_types)