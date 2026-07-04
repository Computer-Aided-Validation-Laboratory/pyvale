"""
Application test 2: Mechanical plate with FEA data
Target: 2 images so that the second one is within 0-1 px displacement
Images: 8-bit BMP, as small as possible (not targeting a particular camera, so what renders fastest)
Cases: AIR_DIFFUSE, PIPE, WATER
Do DIC on these
"""
from enum import StrEnum
import numpy as np
from pathlib import Path
from global_utils import *
from convergence_common import *
from copy import deepcopy

from pyvale.sensorsim.imagetools import ImageTools
import os
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *
# No Blender imports => Should work on Linux


def plate_test(test_case: TestCaseApp):
    # 1. Paths and access to all data used in the scene
    # Object = main mesh that moves
    object_access = "thesis-data/plate_hole/platehole3d_2mr_63f"
    simdata_path = full_path(object_access) # full path to e.g., tank_surface_TRI3.vtk
    ref_texture = full_path("thesis-data/texture/speckle.tiff")
    object_texture = ImageTools.load_image_greyscale(ref_texture) 
    # Pipe - we borrow data from application_1_rbm because it is shorter
    pipe_access = "thesis-data/pipe_plate"
    pipe_path = get_tank_path(pipe_access, Elements.TRI6) # TRI3 or TRI6 only for pipe
    water_path = get_fill_path(pipe_access, Elements.TRI6)

    # 2. Set up the meshes
    scene = Scene()
    object = simdata_csv_to_rtmesh(simdata_path, sens.EDim.THREED) # Plate is (25, 35, 1) mm in x,y,z spans
    # Position the object so it touches the bottom of the pipe
    pipe_bottom_inner_y = - 23 + 4 # Position of the inner edge of the pipe tank, based on the geometry
    object.place_at(np.array([0.0, pipe_bottom_inner_y, 0.0]), anchor=Anchor.BASE)
    pipe = any_mesh_to_rtmesh(pipe_path)
    water = any_mesh_to_rtmesh(water_path)

    # 3. Camera and output settings
    # Output settings and directory
    output_format = ImageFormat(OutputFormat.IMG_BMP_8BIT, BitDepth.BIT_8, ChannelCount.MONO, True)
    base_data_dir = f"app2_fea/renders/{test_case.value}"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    # Anti-aliasing
    anti_alias = 1; # for anti-aliasingr
    image_width = image_width_phs6 # px
    image_height = image_width_phs6 # px; height = width for Novas
    pixel_pitch = pixel_pitch_ph6
    focal_length = 50 # mm
    sensor_height_mm = sensor_height_phs6
    # Derived camera parameters
    fov_height = object.get_size()[1] + 5 # See the entire height of the plate + some extra to get the edges
    camera_distance = camera_working_distance(focal_length, fov_height, sensor_height_mm)
    camera_y_position = pipe_bottom_inner_y + object.get_size()[1]/2 # Lower the y-position of the camera to match that of the plate
    target_distance = camera_distance - focal_length
    camera_target = np.array([0, camera_y_position, target_distance])
    camera_center = np.array([0, camera_y_position, camera_distance])
    angle_vertical_view = vertical_fov_from_sensor(sensor_height=sensor_height_mm, focal_length=focal_length)
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view)

    #SceneVisualiser([object, pipe], cam) # Check positioning
    
    if test_case == TestCaseApp.AIR_DIFFUSE:
        print(f"--------------------------------\nTESTED CASE: AIR DIFFUSE\n--------------------------------")
    elif test_case == TestCaseApp.PIPE:
        print(f"--------------------------------\nTESTED CASE: EMPTY PIPE\n--------------------------------")
        pipe.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.PLASTIC_ACRYLIC)
        scene.add_rtmesh(pipe)
    elif test_case == TestCaseApp.WATER:
        print(f"--------------------------------\nTESTED CASE: PIPE WITH WATER\n----------------------------")
        pipe.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.PLASTIC_ACRYLIC, priority=1)
        scene.add_rtmesh(pipe)
        water.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.WATER, priority=0)
        scene.add_rtmesh(pipe)

    # 4.Texture and speckle pattern information for the plate
    # The loaded texture is 2464 x 2056 px (5MPx), 8-bit .tiff; speckles sampled by 5 pixels
    # Rescale the texture since we're using 1024 x 1024 px res, so speckle size will be too small
    object.set_surface(SurfType.TEXTURE, surface_fill=object_texture, material_type=MaterialType.DIFFUSE)
    uv_scale = speckle_scaling(image_width, image_height, 2464, 2056, 5, 5) # Aim to have 5 px speckles again
    object.uvs *= uv_scale
    # But if scaling is needed refer to application_1_rbm
    scene.add_rtmesh(object)

    # 5. Pick frames to render
    # We only want 2:
    # 1. Undeformed, so at t=0
    # 2. Deformed, somewhere s.t., displacement is <= 1 px
    scale = spatial_scale(fov_height, image_height) # mm/px, so 1 px = this in mm; 0.0390625 in this case
    temp_frame_idx = object.timestep_count - 1 # Start checking displacements from the last frame
    displaced_frame_idx = 0
    max_displacement = scale * 2 # Starting value to make sure this is always bigger 
    while (max_displacement > scale):
        total_displacements = object.node_coords_over_time[temp_frame_idx] - object.node_coords_over_time[0] # Displacement between t=frame_idx and t=0
        max_displacement = np.max(total_displacements)
        displaced_frame_idx = temp_frame_idx
        temp_frame_idx -= 1
    print(f"Displaced frame idx: {displaced_frame_idx}, with maximum displacement of {max_displacement} mm, which is less than the scale 1 px = {scale} mm")
    # 5. Render
    scene.add_camera(cam)
    scene_deformed = deepcopy(scene)

    fresh_filename = "rtimage_0_cam0.bmp"
    # Render undeformed image
    #render_scene(image_height, image_width, scene, anti_alias, target_path, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
    #new_filename = "rtimage_frame0.bmp"
    #os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(new_filename))
    # Render deformed image
    #render_scene(image_height, image_width, scene_deformed, anti_alias, target_path, RenderType.STATIC, frames_to_render=displaced_frame_idx, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
    #new_filename = f"rtimage_frame{displaced_frame_idx}.bmp"
    #os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(new_filename))


#plate_test(TestCaseApp.AIR_DIFFUSE)
   
