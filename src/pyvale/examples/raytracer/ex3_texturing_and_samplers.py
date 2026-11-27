# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
UV unwrapping, speckle texturing, and texture samplers
------------------------------------------------------

This example shows how to apply a greyscale speckle pattern to linear and
higher-order meshes using ``BlenderUnwrapper`` and how to choose between the 
built-in texture interpolation filters (``TextureSampler``).

Available ``TextureSampler`` options:
    ``NEAREST_NEIGHBOUR``, ``LANCZOS_2``, ``LANCZOS_3``, ``CATMULL_ROM``,
    ``MITCHELL_NETRAVALI``, ``BSPLINE``, and ``QUINTIC_SPLINE``.
"""

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dataset as dataset
from pyvale.dataset import EElemTest
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
# Output directory and texture loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Load the 5 MPix DIC speckle pattern bundled with ``pyvale.dataset`` as a 2D
# greyscale NumPy array.

out_dir = Path.cwd() / "pyvale-output" / "raytracer" / "ex3_texturing_and_samplers"
out_dir.mkdir(parents=True, exist_ok=True)

speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())

# %%
# Building a scene with mixed element types (TRI3 and QUAD8)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We place a 3D plate with a hole (``TRI3``) alongside a rotated ``HEX20`` cube
# (skinned to ``QUAD8``).

scene = Scene()
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

plate_mesh = simdata_to_rtmesh(
    dataset.render_mechanical_3d_path(),
    target_size=0.38,
    size_axis=Axis.Y,
    world_position=np.array([-0.22, 0.0, -1.5]),
    anchor=Anchor.CENTER,
)

cube_mesh = simdata_to_rtmesh(
    dataset.element_case_output_path(EElemTest.HEX20),
    target_size=0.28,
    size_axis=Axis.Y,
    world_position=np.array([0.24, 0.0, -1.6]),
    anchor=Anchor.CENTER,
    world_rotation=Rotation.from_euler("xy", [25.0, -30.0], degrees=True),
)

# %%
# UV unwrapping and applying the speckle texture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``BlenderUnwrapper.smart_unwrap()`` automatically generates UV coordinates
# without requiring manual seam selection (for manual seams, see
# ``SeamSelector`` in ``pyvale.raytracer.rtmeshvisuals``).
# Manual seam selection is useful for the sphere, for example.

unwrapper = BlenderUnwrapper()

unwrapper.add_rtmesh(plate_mesh)
unwrapper.smart_unwrap()
plate_mesh.set_surface(
    surface_type=SurfType.TEXTURE,
    surface_fill=speckle_img,
    material_type=MaterialType.DIFFUSE,
)

unwrapper.add_rtmesh(cube_mesh)
unwrapper.smart_unwrap()
cube_mesh.set_surface(
    surface_type=SurfType.TEXTURE,
    surface_fill=speckle_img,
    material_type=MaterialType.DIFFUSE,
)

scene.add_rtmesh([plate_mesh, cube_mesh])

# %%
# Rendering with high-order texture interpolation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``TextureSampler.BSPLINE`` provides smooth sub-pixel interpolation suitable
# for synthetic DIC image generation.

image_format = ImageFormat(
    output_format=OutputFormat.IMG_BMP_8BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.MONO,
    grayscale=True,
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
