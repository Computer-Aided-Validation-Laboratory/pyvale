# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from pathlib import Path
from pyvale.raytracer.rtscene import Scene, RenderType, TextureSampler, ShadingType, find_max_displacements
from pyvale.raytracer.rtmesh import SurfType, ElementNodeCount, RTMesh

from pyvale.raytracer.rtmaincpp import cpp_render_scene # Import C++ backend

def check_uniform_surfaces(scene: Scene):
    """
    Checks if all meshes in the passed scene have the same surface type.

    Parameters:
    -----------
    scene: Scene
        The scene to check.

    Returns:
    --------
    bool
        True if all meshes have the same surface type, False otherwise.
    """
    surfaces_detected = set(scene.surface_types) # Remove duplicates
    # RTMesh can't be added without surface type, so omit checking if all of them have it
    return len(surfaces_detected) == 1 # True if all surface types are the same

def check_uniform_elements(scene: Scene):
    """
    Checks if all meshes in the passed scene have the same element type.

    Parameters:
    -----------
    scene: Scene
        The scene to check.

    Returns:
    --------
    bool
        True if all meshes have the same element type, False otherwise.
    """
    elements_detected = set(scene.nodes_per_element) # Remove duplicates
    return len(elements_detected) == 1 # True if all element types are the same


def render_scene(image_height: int,
                 image_width: int,
                 scene: Scene,
                 antialiasing_samples: int,
                 out_directory_path: Path,
                 render_type: RenderType = RenderType.DYNAMIC,
                 frames_to_render: int = None,
                 texture_sampler: TextureSampler | None = None,
                 shading_type: ShadingType = ShadingType.FLAT,
                 grayscale: bool = True):
    """
    Performs checks and dispatches the scene to the C++ rendering backend.

    Parameters:
    -----------
    image_height: int
        The height of the output image in pixels.
    image_width: int
        The width of the output image in pixels.
    scene: Scene
        The scene to render.
    antialiasing_samples: int
        The number of antialiasing samples to use.
    out_directory_path: Path
        The path to the output directory.
    render_type: RenderType
        The type of rendering to perform. Can be either DYNAMIC or STATIC.
    frames_to_render: int
        Dynamic renders: The number of frames to render. Defaults to the maximum timesteps there is available data for; if some meshes lack data for all timeframes, it is pre-filled with last known values.
        Static renders: The number of the single frame to render; defaults to the first one otherwise. Nb4 this could maybe be a tuple to specify the range instead?
    texture_sampler: TextureSampler | None
        The algorithm used to sample the textures onto the mesh surfaces. Defaults to None and gets set to nearest neighbour.
    shading_type: ShadingType
        The type of shading to use. Can be either FLAT (geometric normals used for shading) or BLENDED (node normals used for shading). Defaults to FLAT.
    grayscale: bool
        Flag to determine whether the image is to be rendered using grayscale or in colour. Defaults to True.

    Raises:
    -------
    ValueError:
        If no meshes are in the scene or the requested number of timesteps exceeds the available data.

    Returns:
    --------
    None, but rendered images will be saved to the specified output directory.
    """
    # Assign default values depending on the render type if target frame count was not specified
    if frames_to_render is None:
        if render_type == RenderType.STATIC:
            frames_to_render = 1
        elif render_type == RenderType.DYNAMIC:
            frames_to_render = scene.timestep_count

     # Check if there are meshes in the scene and if there are, check their surfaces and element types
    if scene.mesh_count == 0:
        raise ValueError("No meshes in scene.")
    if len(scene.camera_center) == 0:
        raise ValueError("No cameras in scene.")
    #else: # For potential dispatch to different versions of the renderer - it might or might not be implemented
    #    uniform_surfaces = check_uniform_surfaces(scene)
    #    uniform_elements = check_uniform_elements(scene)

    # Sanity check for the values
    if frames_to_render <= scene.timestep_count:
        scene._clip_scene(frames_to_render, render_type)
        #max_displacement_per_step_array = find_max_displacements(scene, render_type) # Data for deciding if to update/rebuild TLAS/BLAS. Currently WIP and doesn't get passed
    else:
        raise ValueError("Number of requested frames exceeds the number of timesteps with availabile data.")

    if render_type == RenderType.DYNAMIC:
        scene._fill_empty_timesteps() # VERY important to avoid segfaults if there is missing timestep data for some meshes in the scene

    # If texture sampling method is not selected, set to nearest neighbour by default (both to be able to render and because C++ expects an int, so we want to pass a number even for solid surfaces)
    if texture_sampler is None:
        texture_sampler = TextureSampler.NEAREST_NEIGHBOUR
        # Display information about setting the algorithm type if there are textured meshes in the scene 
        if SurfType.TEXTURE in scene.surface_types:
            print("Texture sampler not selected. Using nearest neighbour.")

    scene.refractive_indices.append(scene.scene_ri) # Append the scene RI at the end of the refractive indices list to pass fewer arguments to the renderer, while keeping indexing for the meshes consistent

    if shading_type == ShadingType.FLAT:
        print("Flat shading selected. Geometric normals will be used for all elements.")
    elif shading_type == ShadingType.BLENDED:
        print("Blended shading selected. Angle-averaged node normals will be used for TRI3, and Jacobians for QUAD4, QUAD8, QUAD9, and TRI6.")
    elif shading_type == ShadingType.ANGLE_AVG_BLENDED:
        print("Angle-averaged blended shading selected. Angle-averaged node normals will be used for all elements.")
        
    # Select appropriate rendering function based on these booleans to minimize branching in backend rendered if possible
    # Not sure if we will need to implement this yet - BVH builder is still fast with conditional checks (and we run it once per frame), and branching based on element/surface type was moved out of the hot loops
    #if uniform_surfaces and uniform_elements:
       # Best case - everything is the same
    #    pass
    #elif uniform_surfaces and not uniform_elements:
        # One surface type, but different element types
    #    pass
    #elif not uniform_surfaces and uniform_elements:
        # One element type, but different surface types
    #    pass
    #else:
        # Worst case: mixture of different surface and element types
    #    pass

    # For now use the general function with branching in it
    #cpp_render_scene(image_height, image_width, antialiasing_samples, out_directory_path, scene.timestep_count, scene.coords_expanded, scene.face_colors, scene.camera_center, scene.pixel_00_center, scene.matrix_pixel_spacing)
    #print(f"Materials: {scene.materials}")
    cpp_render_scene(image_height,
                     image_width,
                     antialiasing_samples,
                     out_directory_path,
                     scene.timestep_count,
                     scene.camera_center,
                     scene.pixel_00_center,
                     scene.matrix_pixel_spacing,
                     scene.matrix_defocus_disc,
                     scene.coords_expanded,
                     scene.normals_expanded,
                     scene.face_colors,
                     scene.uvs,
                     scene.textures,
                     scene.surface_types,
                     scene.materials,
                     scene.refractive_indices,
                     scene.mesh_priorities,
                     scene.mesh_object_types,
                     scene.mesh_thickness,
                     texture_sampler,
                     shading_type,
                     grayscale)