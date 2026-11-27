# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Basic ray tracing scene and camera setup
----------------------------------------

This example introduces the core ``pyvale.raytracer`` workflow: creating a
``Scene``, configuring a pinhole ``Camera``, loading a 3D finite-element mesh
from an Exodus simulation dataset with ``simdata_to_rtmesh``, assigning a
surface appearance, and rendering a static image.

Test case: 3D mechanical plate with a central hole loaded in tension.
"""

from pathlib import Path
import numpy as np

import pyvale.data as dataset
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
)
from pyvale.raytracer.rtscene import Scene

# %%
# Output directory
# ^^^^^^^^^^^^^^^^
# Rendered images are written to a subdirectory inside ``pyvale-output``.

out_dir = Path.cwd() / "pyvale-output" / "raytracer" / "ex1_basic_scene"
out_dir.mkdir(parents=True, exist_ok=True)

# %%
# Creating the scene and camera
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A ``Scene`` holds all cameras and ``RTMesh`` objects. Here we create a
# standard pinhole ``Camera`` positioned along ``+Z`` looking toward ``-Z``.
# Calling ``print_view_dims()`` prints the viewport size in world units, which
# is helpful when sizing and placing meshes in the scene.

scene = Scene()

image_width = 640
aspect_ratio = 16.0 / 9.0
image_height = int(image_width / aspect_ratio)

camera_center = np.array([0.0, 0.0, 0.41])
camera_target = np.array([0.0, 0.0, -1.0])
vertical_fov_deg = 20.0

camera = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=camera_center,
    point_camera_target=camera_target,
    angle_vertical_view=vertical_fov_deg,
)
camera.print_view_dims()
scene.add_camera(camera)

# %%
# Loading and positioning a mesh
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``simdata_to_rtmesh`` loads an Exodus simulation output, skins 3D volume
# meshes to surface elements, scales the mesh so that the chosen ``size_axis``
# matches ``target_size``, and places the mesh ``anchor`` at ``world_position``.

data_path = dataset.render_mechanical_3d_path()
plate_mesh = simdata_to_rtmesh(
    data_path,
    target_size=0.4,
    size_axis=Axis.Y,
    world_position=np.array([0.0, 0.0, -1.5]),
    anchor=Anchor.CENTER,
)

# %%
# Setting the surface and rendering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Every ``RTMesh`` must have its surface configured before it is added to the
# scene. ``SurfType.FIELD_COLOR`` with ``MaterialType.UNLIT`` renders a flat
# solid colour without lighting or shadows. This is useful for quick geometry checks.

plate_mesh.set_surface(
    surface_type=SurfType.FIELD_COLOR,
    surface_fill=np.array([0.25, 0.42, 0.62]),
    material_type=MaterialType.UNLIT,
)
scene.add_rtmesh(plate_mesh)

image_format = ImageFormat(
    output_format=OutputFormat.IMG_BMP_24BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.RGB,
    grayscale=False,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=scene,
    antialiasing_samples=4,
    out_directory_path=out_dir,
    render_type=RenderType.STATIC,
    shading_type=ShadingType.FLAT,
    image_format=image_format,
)
