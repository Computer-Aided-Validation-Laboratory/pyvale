# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Multi-object positioning and mixed textured/solid scenes
--------------------------------------------------------

This example shows how to assemble multi-object scenes using ``Anchor`` points,
``snap_to()``, ``translate()``, and ``rotate()``, while combining textured and
solid-colour meshes of different element types (``TRI3``, ``QUAD4``, ``QUAD8``)
and customising the scene background colour.
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
    snap_to,
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
# Scene, background, and camera setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

out_dir = (
    Path.cwd() / "pyvale-output" / "raytracer" / "ex4_positioning_and_mixed_scene"
)
out_dir.mkdir(parents=True, exist_ok=True)

scene = Scene()
scene.set_background(np.array([0.85, 0.88, 0.92]))

image_width = 640
image_height = 360
camera = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([0.0, 0.0, 0.0]),
    point_camera_target=np.array([0.0, 0.0, -1.0]),
    angle_vertical_view=20.0,
)
scene.add_camera(camera)

# %%
# Positioning objects with ``Anchor``, ``snap_to``, ``rotate``, and ``translate``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We place a tilted ``QUAD4`` base platform using ``Anchor.TOP`` and snap three
# objects onto its top surface:
# 1. A textured ``QUAD8`` block stacked directly on the platform;
# 2. A small solid ``TRI3`` cube stacked on top of the ``QUAD8`` block and
#    nudged laterally with ``translate()``;
# 3. A textured plate with a hole placed alongside with a 0.06 m gap.

hex8_path = dataset.element_case_output_path(EElemTest.HEX8)
hex20_path = dataset.element_case_output_path(EElemTest.HEX20)
tet4_path = dataset.element_case_output_path(EElemTest.TET4)
plate_path = dataset.render_mechanical_3d_path()

floor_mesh = simdata_to_rtmesh(
    hex8_path,
    target_size=1.1,
    size_axis=Axis.X,
    world_position=np.array([0.0, -0.14, -1.6]),
    anchor=Anchor.TOP,
)
floor_mesh.rotate(Rotation.from_euler("x", 10.0, degrees=True))

block_mesh = simdata_to_rtmesh(
    hex20_path,
    target_size=0.18,
    size_axis=Axis.X,
    world_position=np.array([-0.16, 0.0, -1.45]),
    anchor=Anchor.CENTER,
    world_rotation=Rotation.from_euler("y", -18.0, degrees=True),
)
# Snap onto the tilted floor along Y, inheriting the floor's tilt (rotate=True)
# while keeping block_mesh's own X and Z coordinates (align=()).
snap_to(block_mesh, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)

small_cube = simdata_to_rtmesh(
    tet4_path,
    target_size=0.07,
    size_axis=Axis.X,
    anchor=Anchor.CENTER,
    world_rotation=Rotation.from_euler("y", 15.0, degrees=True),
)
# Stack centered on top of block_mesh, then offset 1.5 cm along +X.
snap_to(small_cube, block_mesh, axis=Axis.Y, gap=0.0, stack_above=True, rotate=True)
small_cube.translate(np.array([0.015, 0.0, 0.0]))

plate_mesh = simdata_to_rtmesh(
    plate_path,
    target_size=0.26,
    size_axis=Axis.Y,
    world_position=np.array([0.16, 0.0, -1.45]),
    anchor=Anchor.CENTER,
)
snap_to(plate_mesh, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)

# %%
# Assigning mixed textured and solid surfaces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Optional interactive positioning preview:
#   from pyvale.raytracer.rtmeshvisuals import SceneVisualiser
#   SceneVisualiser([floor_mesh, block_mesh, small_cube, plate_mesh], camera)

speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())
unwrapper = BlenderUnwrapper()

floor_mesh.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.78, 0.78, 0.78]),
    material_type=MaterialType.DIFFUSE,
)
small_cube.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.25, 0.68, 0.25]),
    material_type=MaterialType.DIFFUSE,
)

for textured_mesh in (block_mesh, plate_mesh):
    unwrapper.add_rtmesh(textured_mesh)
    unwrapper.smart_unwrap()
    textured_mesh.set_surface(
        SurfType.TEXTURE,
        surface_fill=speckle_img,
        material_type=MaterialType.DIFFUSE,
    )

scene.add_rtmesh([floor_mesh, block_mesh, small_cube, plate_mesh])

# %%
# Rendering the assembled scene
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    antialiasing_samples=8,
    out_directory_path=out_dir,
    render_type=RenderType.STATIC,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=image_format,
)
