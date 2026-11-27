# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Output image formats, stereo cameras, and thin-lens depth of field
------------------------------------------------------------------

This example demonstrates three advanced camera and export capabilities of
``pyvale.raytracer``:

1. **Multi-camera (stereo) rendering**: adding more than one ``Camera`` to a
   single ``Scene`` renders synchronized views (``cam0``, ``cam1``, ...) in one
   call to ``render_scene``.
2. **Thin-lens depth of field**: specifying ``focal_length`` (focus distance in
   world units) and ``angle_cone`` (aperture cone angle in degrees) on a
   ``Camera`` enables realistic defocus blur away from the focal plane.
3. **Output ``ImageFormat`` options**: exporting 8-bit effective greyscale
   data inside a 8-bit TIFF container (``IMG_TIFF_16BIT`` + ``BitDepth.BIT_16``)
   as well as inspecting supported formats (``IMG_BMP_8BIT``, ``IMG_BMP_24BIT``,
   ``IMG_TIFF_8BIT``, ``IMG_TIFF_16BIT``, ``IMG_PPM``).
"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale.data import EElemTest
from pyvale.sensorsim.imagetools import ImageTools
from pyvale.raytracer.rtblender import BlenderUnwrapper
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtmain import render_scene
from pyvale.raytracer.rtmesh import (
    Anchor,
    Axis,
    MaterialType,
    SurfType,
    simdata_to_rtmesh,
)
from pyvale.raytracer.rtoutputformat import (
    BitDepth,
    ChannelCount,
    ImageFormat,
    OutputFormat,
    RenderType,
    ShadingType,
    TextureSampler,
)
from pyvale.raytracer.rtscene import Scene

# %%
# Output directories
# ^^^^^^^^^^^^^^^^^^

base_out = Path.cwd() / "pyvale-output" / "raytracer" / "ex8_output_formats_and_cameras"
stereo_dir = base_out / "stereo_tiff8"
dof_dir = base_out / "thin_lens_dof"
stereo_dir.mkdir(parents=True, exist_ok=True)
dof_dir.mkdir(parents=True, exist_ok=True)

image_width = 512
image_height = 384
camera_target = np.array([0.0, 0.0, -1.5])
vertical_fov_deg = 22.0

# %%
# Part 1: Stereo camera pair + 8-bit TIFF (8-bit effective) export
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Adding two cameras angled symmetrically toward the specimen produces
# synchronized stereo pairs in a single ``render_scene`` call.

stereo_scene = Scene()

cam_left = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([-0.15, 0.0, -0.4]),
    point_camera_target=camera_target,
    angle_vertical_view=vertical_fov_deg,
)
cam_right = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([0.30, 0.0, -0.4]),
    point_camera_target=camera_target,
    angle_vertical_view=vertical_fov_deg,
)
stereo_scene.add_camera(cam_left)
stereo_scene.add_camera(cam_right)

speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())
plate_mesh = simdata_to_rtmesh(
    dataset.render_mechanical_3d_path(),
    target_size=0.40,
    size_axis=Axis.Y,
    world_position=camera_target,
    anchor=Anchor.CENTER,
)
unwrapper = BlenderUnwrapper()
unwrapper.add_rtmesh(plate_mesh)
unwrapper.smart_unwrap()
plate_mesh.set_surface(
    SurfType.TEXTURE,
    surface_fill=speckle_img,
    material_type=MaterialType.DIFFUSE,
)
stereo_scene.add_rtmesh(plate_mesh)

tiff_8bit_mono = ImageFormat(
    output_format=OutputFormat.IMG_TIFF_8BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.MONO,
    grayscale=True,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=stereo_scene,
    antialiasing_samples=16,
    out_directory_path=stereo_dir,
    render_type=RenderType.STATIC,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=tiff_8bit_mono,
)

# %%
# Part 2: Thin-lens camera with depth-of-field blur
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Comparing a pinhole camera (``angle_cone=0``) against a thin-lens camera
# focused at the near cube (``focus_distance = 0.41 - (-1.20) = 1.61`` world
# units, ``angle_cone = 2.0`` degrees). The background cube at ``z = -2.2``
# appears visibly defocused.

dof_scene = Scene()

cam_center = np.array([0.0, 0.0, 0.-0.1])
focus_distance = float(cam_center[2] - (-1.20))

thin_lens_cam = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=cam_center,
    point_camera_target=np.array([0.0, 0.0, -1.20]),
    angle_vertical_view=vertical_fov_deg,
    focal_length=focus_distance,
    angle_cone=2.0,
)
print(f"Thin-lens focus distance: {thin_lens_cam.focal_length:.2f}")
print(f"Defocus disc basis:\n{thin_lens_cam.matrix_defocus_disc}")
dof_scene.add_camera(thin_lens_cam)

hex8_path = dataset.element_case_output_path(EElemTest.HEX8)
rot = Rotation.from_euler("xy", [20.0, 30.0], degrees=True)

near_cube = simdata_to_rtmesh(
    hex8_path,
    target_size=0.18,
    size_axis=Axis.Y,
    world_position=np.array([-0.14, 0.0, -1.20]),
    anchor=Anchor.CENTER,
    world_rotation=rot,
)
near_cube.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.25, 0.65, 0.85]),
    material_type=MaterialType.DIFFUSE,
)

far_cube = simdata_to_rtmesh(
    hex8_path,
    target_size=0.26,
    size_axis=Axis.Y,
    world_position=np.array([0.22, 0.0, -2.20]),
    anchor=Anchor.CENTER,
    world_rotation=rot,
)
far_cube.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.85, 0.45, 0.25]),
    material_type=MaterialType.DIFFUSE,
)

dof_scene.add_rtmesh([near_cube, far_cube])

bmp_rgb = ImageFormat(
    output_format=OutputFormat.IMG_BMP_24BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.RGB,
    grayscale=False,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=dof_scene,
    antialiasing_samples=2**7,
    out_directory_path=dof_dir,
    render_type=RenderType.STATIC,
    shading_type=ShadingType.BLENDED,
    image_format=bmp_rgb,
)
