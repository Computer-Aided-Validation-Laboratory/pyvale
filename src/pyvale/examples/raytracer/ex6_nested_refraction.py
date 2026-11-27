# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Nested refractive volumes and priority ordering
-----------------------------------------------

When refractive media are nested inside one another (for example, a specimen
immersed in a liquid inside a glass vessel), ``pyvale.raytracer`` uses an
integer ``priority`` parameter on each ``RTMesh`` to resolve overlapping
dielectric boundaries:

* Lower priority number -> outer medium (e.g. ``priority=0`` for the glass wall)
* Higher priority number -> nested inner medium (e.g. ``priority=1`` for the
  liquid fill, ``priority=2`` for an immersed object)

This example models a glass container filled with tinted liquid
and a speckled mechanical specimen immersed at its centre.
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
    MeshType,
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
from pyvale.raytracer.rtpresets import MaterialPresets
from pyvale.raytracer.rtscene import Scene

# %%
# Scene and camera setup
# ^^^^^^^^^^^^^^^^^^^^^^

out_dir = Path.cwd() / "pyvale-output" / "raytracer" / "ex6_nested_refraction"
out_dir.mkdir(parents=True, exist_ok=True)

scene = Scene()
scene.set_refractive_index(MaterialPresets.AIR.RI)

image_width = 640
image_height = 480
camera = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([0.0, 0.0, 0.4]),
    point_camera_target=np.array([0.0, 0.0, -1.0]),
    angle_vertical_view=22.0,
)
scene.add_camera(camera)

# %%
# Creating nested refractive volumes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build two concentric ``HEX8`` cubes representing:
# 1. Outer glass block (``priority=0``)
# 2. Inner water volume (``priority=1``)
# Inside the water sits a textured mechanical plate with a central hole.

hex8_path = dataset.element_case_output_path(EElemTest.HEX8)
vessel_rot = Rotation.from_euler("y", 18.0, degrees=True)
centre_pos = np.array([0.0, 0.0, -1.5])

outer_glass = simdata_to_rtmesh(
    hex8_path,
    target_size=0.44,
    size_axis=Axis.X,
    world_position=centre_pos,
    anchor=Anchor.CENTER,
    world_rotation=vessel_rot,
)
outer_glass.set_surface(
    surface_type=SurfType.FIELD_COLOR,
    material_type=MaterialType.REFRACTIVE,
    material=MaterialPresets.GLASS_SODA_LIME,
    mesh_type=MeshType.SOLID,
    priority=0,
    reference_thickness=0.05,
)

inner_water = simdata_to_rtmesh(
    hex8_path,
    target_size=0.38,
    size_axis=Axis.X,
    world_position=centre_pos,
    anchor=Anchor.CENTER,
    world_rotation=vessel_rot,
)
inner_water.set_surface(
    surface_type=SurfType.FIELD_COLOR,
    material_type=MaterialType.REFRACTIVE,
    material=MaterialPresets.WATER,
    mesh_type=MeshType.SOLID,
    priority=1,
    reference_thickness=0.38,
)

# %%
# Immersed textured specimen
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())
plate_mesh = simdata_to_rtmesh(
    dataset.render_mechanical_3d_path(),
    target_size=0.28,
    size_axis=Axis.Y,
    world_position=centre_pos,
    anchor=Anchor.CENTER,
)

unwrapper = BlenderUnwrapper()
unwrapper.add_rtmesh(plate_mesh)
unwrapper.smart_unwrap()
plate_mesh.set_surface(
    surface_type=SurfType.TEXTURE,
    surface_fill=speckle_img,
    material_type=MaterialType.DIFFUSE,
)

scene.add_rtmesh([outer_glass, inner_water, plate_mesh])

# %%
# Rendering the nested refractive scene
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``render_scene`` automatically increases ``max_depth`` and
# ``min_refractive_depth`` based on the number of nested refractive priorities
# present in the scene.

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
    antialiasing_samples=12,
    out_directory_path=out_dir,
    render_type=RenderType.STATIC,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=image_format,
)
