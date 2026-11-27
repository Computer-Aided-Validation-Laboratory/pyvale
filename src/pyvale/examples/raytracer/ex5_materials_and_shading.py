# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Materials, presets, and shading modes
-------------------------------------

This example showcases the four surface material models supported by
``pyvale.raytracer`` alongside ``MaterialPresets`` and custom ``Material``
definitions:

* ``MaterialType.DIFFUSE``    — Lambertian diffuse scattering (floor + backdrop)
* ``MaterialType.SPECULAR``   — mirror-like metallic reflection (``MaterialPresets.COPPER``)
* ``MaterialType.REFRACTIVE`` — dielectric refraction as a ``MeshType.SOLID``
  (``MaterialPresets.GLASS_BOROSILICATE``) and as a thin ``MeshType.SHELL``
  (custom tinted acrylic ``Material``)
* ``MaterialType.UNLIT``      — constant flat shading ignoring lighting

We also compare ``ShadingType.BLENDED`` (smooth vertex/Jacobian normals) and
``ShadingType.FLAT`` (faceted geometric normals).
"""

from pathlib import Path
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

import pyvale.data as dataset
from pyvale.data import EElemTest
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtmain import render_scene
from pyvale.raytracer.rtmesh import (
    Anchor,
    Axis,
    MaterialType,
    MeshType,
    SurfType,
    any_mesh_to_rtmesh,
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
)
from pyvale.raytracer.rtpresets import Material, MaterialPresets
from pyvale.raytracer.rtscene import Scene

# %%
# Output directory, scene, and camera setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

out_dir = Path.cwd() / "pyvale-output" / "raytracer" / "ex5_materials_and_shading"
out_dir.mkdir(parents=True, exist_ok=True)

scene = Scene()
# Set the ambient medium refractive index (air ~ 1.0003).
scene.set_refractive_index(1.0003)

image_width = 640
image_height = 360
camera = Camera(
    image_width=image_width,
    image_height=image_height,
    camera_center=np.array([0.0, 0.0, 0.2]),
    point_camera_target=np.array([0.0, 0.0, -1.0]),
    angle_vertical_view=22.0,
)
scene.add_camera(camera)

# %%
# Building meshes for each material type
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# A tilted diffuse floor supports four objects: a specular copper sphere, a
# refractive solid borosilicate glass sphere, a refractive thin-shell tinted
# acrylic cube, and an unlit reference cube behind them.

hex8_path = dataset.element_case_output_path(EElemTest.HEX8)
hex20_path = dataset.element_case_output_path(EElemTest.HEX20)

vtk_sphere_path = out_dir / "sphere_tri3.vtk"
sphere_grid = (
    pv.Sphere(radius=0.5, theta_resolution=32, phi_resolution=32)
    .triangulate()
    .cast_to_unstructured_grid()
)
sphere_grid.clear_data()
sphere_grid.save(vtk_sphere_path)

floor_mesh = simdata_to_rtmesh(
    hex8_path,
    target_size=1.2,
    size_axis=Axis.X,
    world_position=np.array([0.0, -0.14, -1.55]),
    anchor=Anchor.TOP,
)
floor_mesh.rotate(Rotation.from_euler("x", 10.0, degrees=True))
floor_mesh.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.75, 0.75, 0.75]),
    material_type=MaterialType.DIFFUSE,
)

# 1. Specular metallic sphere (MaterialPresets.COPPER)
copper_sphere = any_mesh_to_rtmesh(
    vtk_sphere_path,
    target_size=0.22,
    size_axis=Axis.X,
    world_position=np.array([-0.26, 0.0, -1.35]),
    anchor=Anchor.CENTER,
)
snap_to(copper_sphere, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)
copper_sphere.set_surface(
    SurfType.FIELD_COLOR,
    material_type=MaterialType.SPECULAR,
    material=MaterialPresets.COPPER,
)

# 2. Refractive solid borosilicate glass sphere (MeshType.SOLID)
glass_sphere = any_mesh_to_rtmesh(
    vtk_sphere_path,
    target_size=0.22,
    size_axis=Axis.X,
    world_position=np.array([0.0, 0.0, -1.30]),
    anchor=Anchor.CENTER,
)
snap_to(glass_sphere, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)
glass_sphere.set_surface(
    SurfType.FIELD_COLOR,
    material_type=MaterialType.REFRACTIVE,
    material=MaterialPresets.GLASS_BOROSILICATE,
    mesh_type=MeshType.SOLID,
    reference_thickness=0.22,
)

# 3. Refractive thin-shell tinted acrylic cube (MeshType.SHELL + custom Material)
tinted_acrylic = Material(color=np.array([0.65, 0.85, 0.98]), RI=1.490)
shell_cube = simdata_to_rtmesh(
    hex20_path,
    target_size=0.18,
    size_axis=Axis.X,
    world_position=np.array([0.26, 0.0, -1.35]),
    anchor=Anchor.CENTER,
    world_rotation=Rotation.from_euler("y", -25.0, degrees=True),
)
snap_to(shell_cube, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)
shell_cube.set_surface(
    SurfType.FIELD_COLOR,
    material_type=MaterialType.REFRACTIVE,
    material=tinted_acrylic,
    mesh_type=MeshType.SHELL,
    thickness=0.015,
    reference_thickness=0.015,
)

# 4. Unlit reference block positioned behind the glass sphere to highlight refraction
backdrop_cube = simdata_to_rtmesh(
    hex8_path,
    target_size=0.14,
    size_axis=Axis.X,
    world_position=np.array([0.0, 0.0, -1.75]),
    anchor=Anchor.CENTER,
)
snap_to(backdrop_cube, floor_mesh, axis=Axis.Y, align=(), gap=0.0, stack_above=True, rotate=True)
backdrop_cube.set_surface(
    SurfType.FIELD_COLOR,
    surface_fill=np.array([0.92, 0.52, 0.47]),
    material_type=MaterialType.UNLIT,
)

scene.add_rtmesh([floor_mesh, copper_sphere, glass_sphere, shell_cube, backdrop_cube])

# %%
# Rendering with smooth blended shading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``ShadingType.BLENDED`` interpolates vertex normals (or exact element
# Jacobians for higher-order curved elements), eliminating faceting on spheres
# and curved surfaces.

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
    shading_type=ShadingType.BLENDED,
    image_format=image_format,
)
