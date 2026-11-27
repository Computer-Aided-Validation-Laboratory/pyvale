# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Dynamic multi-frame rendering (FEA deformation and synthetic motion)
--------------------------------------------------------------------

This example demonstrates two ways to render multi-frame image sequences with
``RenderType.DYNAMIC``:

1. **FEA nodal displacements**: ``simdata_to_rtmesh`` preserves every timestep
   present in the Exodus file, allowing ``render_scene`` to render deformed
   frames directly with UV-mapped speckle textures attached to the deforming
   surface.
2. **Synthetic rigid-body motion**: ``create_rigid_linear_translation`` and
   ``RTMesh.add_temporal_displacement`` add a controlled linear translation to
   any mesh across a chosen number of frames.

Use ``frames_to_render`` to clip the sequence to the first *N* timesteps.
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
    create_rigid_linear_translation,
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
# Output directory, scene, and camera setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

out_dir = Path.cwd() / "pyvale-output" / "raytracer" / "ex7_dynamic_deformation"
out_dir.mkdir(parents=True, exist_ok=True)

scene = Scene()
image_width = 512
image_height = 384

camera = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([0.0, 0.0, 0.0]),
    point_camera_target=np.array([0.0, 0.0, -1.0]),
    angle_vertical_view=22.0,
)
scene.add_camera(camera)

# %%
# 1. Deforming FEA plate with a hole (multi-timestep Exodus SimData)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plate_mesh = simdata_to_rtmesh(
    dataset.render_mechanical_3d_path(),
    target_size=0.45,
    size_axis=Axis.Y,
    world_position=np.array([-0.15, 0.0, -1.8]),
    anchor=Anchor.CENTER,
)

# %%
# 2. Cube with synthetic rigid linear translation along +X
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``create_rigid_linear_translation`` generates a ``(N_nodes, N_frames, 3)``
# displacement array that can be attached via ``add_temporal_displacement``.

n_frames = 4

moving_cube = simdata_to_rtmesh(
    dataset.element_case_output_path(EElemTest.HEX8),
    target_size=0.16,
    size_axis=Axis.Y,
)
snap_to(moving_cube, plate_mesh, axis=Axis.X, stack_above=True, gap=0.06)
moving_cube.rotate(Rotation.from_euler("x", 20.0, degrees=True))

rigid_disp = create_rigid_linear_translation(
    node_count=moving_cube.node_count,
    target_timestep_count=n_frames,
    final_displacement=0.08,
    direction=Axis.X,
)
moving_cube.add_temporal_displacement(rigid_disp)

# %%
# Texturing both meshes and rendering ``n_frames`` dynamic frames
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())
unwrapper = BlenderUnwrapper()

for mesh in (plate_mesh, moving_cube):
    unwrapper.add_rtmesh(mesh)
    unwrapper.smart_unwrap()
    mesh.set_surface(
        surface_type=SurfType.TEXTURE,
        surface_fill=speckle_img,
        material_type=MaterialType.DIFFUSE,
    )

scene.add_rtmesh([plate_mesh, moving_cube])

image_format = ImageFormat(
    output_format=OutputFormat.IMG_TIFF_8BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.MONO,
    grayscale=True,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=scene,
    antialiasing_samples=4,
    out_directory_path=out_dir,
    render_type=RenderType.DYNAMIC,
    frames_to_render=n_frames,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=image_format,
)
