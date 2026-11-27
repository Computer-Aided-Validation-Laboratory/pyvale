# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Quadratic bunny: one speckled bunny and six flying bunnies showcasing 
various materials
---------------------------------------------------------------

This example uses a Gmsh 2 mesh of the simplified Stanford bunny.
Its boundary is made entirely of quadratic ``TRI6`` elements. The example
extracts that boundary to a VTK file in the output directory before rendering.

1. **Speckled bunny**: one upright bunny is UV-unwrapped with
   ``BlenderUnwrapper``, textured with the speckle pattern bundled in
   ``pyvale.dataset``, and shaded as a diffuse surface.
2. **Flying bunnies**: six upright bunnies at various depths, rotations
   and sizes, showcasing a speckled surface, specular copper and tungsten,
   refractive honey, and unlit cyan and pink bunnies.
"""

from pathlib import Path

import meshio
import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dataset as dataset
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
    any_mesh_to_rtmesh,
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
# Bundled Gmsh mesh, output directories and surface extraction
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``any_mesh_to_rtmesh`` could read the ``.msh`` file directly, but
# it would write a ``temp.vtk`` file next to the bundled mesh on every call.
# Extracting the quadratic surface explicitly keeps the data directory clean
# and makes the VTK reusable for all of the bunnies below. The bunny's "up"
# direction is +Z in the source file, whereas the Feebee uses +Y as up, hence
# the -90 degree tip about X below.

bunny_gmsh_path = Path(__file__).resolve().parent / "data" / "bunny.msh"
if not bunny_gmsh_path.is_file():
    raise FileNotFoundError(
        f"Gmsh bunny mesh not found at {bunny_gmsh_path}. "
        "It should be bundled alongside this example."
    )

base_out = Path.cwd() / "pyvale-output" / "raytracer" / "ex9_quadratic_bunny"
speckle_dir = base_out / "speckled_bunny"
flight_dir = base_out / "flying_bunnies"
base_out.mkdir(parents=True, exist_ok=True)
speckle_dir.mkdir(parents=True, exist_ok=True)
flight_dir.mkdir(parents=True, exist_ok=True)


def extract_quadratic_surface_vtk(gmsh_path: Path, vtk_path: Path) -> Path:
    """Write the ``TRI6`` boundary of the Gmsh bunny to a surface-only VTK."""
    gmsh_mesh = meshio.read(gmsh_path, file_format="gmsh")
    surface_cells = gmsh_mesh.get_cells_type("triangle6")
    if surface_cells.size == 0:
        raise ValueError(f"No TRI6 boundary found in {gmsh_path}.")

    used_nodes, local_connectivity = np.unique(surface_cells, return_inverse=True)
    surface_mesh = meshio.Mesh(
        points=gmsh_mesh.points[used_nodes],
        cells=[("triangle6", local_connectivity.reshape(surface_cells.shape))],
    )
    surface_mesh.write(vtk_path, file_format="vtk")
    return vtk_path


bunny_vtk_path = extract_quadratic_surface_vtk(
    bunny_gmsh_path, base_out / "bunny_tri6.vtk"
)

upright = Rotation.from_euler("x", -90.0, degrees=True)


def load_upright_bunny(position,
                       height,
                       yaw_deg=0.0,
                       roll_deg=0.0):
    """Load a fresh ``TRI6`` bunny, stand it up and place it in the world.

    ``height`` is measured along the source +Z axis, which becomes world +Y
    after the upright rotation. ``yaw_deg`` spins the bunny about its own
    vertical axis; ``roll_deg`` tilts the upright bunny in the camera view
    (about world +Z).
    """
    spin = Rotation.from_euler("z", yaw_deg, degrees=True)
    roll = Rotation.from_euler("z", roll_deg, degrees=True)
    rotation = roll * upright * spin

    return any_mesh_to_rtmesh(
        bunny_vtk_path,
        target_size=height,
        size_axis=Axis.Z,
        world_position=np.asarray(position, dtype=np.float64),
        world_rotation=rotation,
        anchor=Anchor.CENTER,
    )


# %%
# Part 1: one speckle-textured quadratic bunny
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The mesh has no UVs of its own, so Blender's smart projection generates
# them before the speckle pattern
# is attached.

image_width = 640
image_height = 400
speckle_img = ImageTools.load_image_greyscale(dataset.dic_pattern_5mpx_path())

speckle_bunny = load_upright_bunny(
    position=np.array([0.0, 0.0, -1.5]),
    height=0.60,
)

unwrapper = BlenderUnwrapper()
unwrapper.add_rtmesh(speckle_bunny)
unwrapper.smart_unwrap()
speckle_bunny.set_surface(
    surface_type=SurfType.TEXTURE,
    surface_fill=speckle_img,
    material_type=MaterialType.DIFFUSE,
)

speckle_scene = Scene()
speckle_scene.set_background(np.array([0.70, 0.72, 0.75]))
speckle_scene.add_camera(
    Camera(
        image_width=image_width,
        image_height=image_height,
        camera_center=np.array([0.0, 0.0, 0.0]),
        point_camera_target=np.array([0.0, 0.0, -1.0]),
        angle_vertical_view=30.0,
    )
)
speckle_scene.add_rtmesh(speckle_bunny)

mono_bmp = ImageFormat(
    output_format=OutputFormat.IMG_BMP_8BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.MONO,
    grayscale=True,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=speckle_scene,
    antialiasing_samples=12,
    out_directory_path=speckle_dir,
    render_type=RenderType.STATIC,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=mono_bmp,
)

# %%
# Part 2: six bunnies flying in space
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Six copies of the same quadratic bunny mesh are placed at various depths,
# positions, sizes, and rotations.
# Together they show all four material models: an unlit speckled surface,
# specular copper and tungsten, a refractive honey solid, and unlit cyan and
# pink bunnies with solid colours.

flight_scene = Scene()
flight_scene.set_refractive_index(1.0003)
flight_scene.set_background(np.array([0.48, 0.52, 0.60]))

camera_z = 0.2
vertical_fov = 36.0
flight_scene.add_camera(
    Camera(
        image_width=image_width,
        image_height=image_height,
        camera_center=np.array([0.0, 0.0, camera_z]),
        point_camera_target=np.array([0.0, 0.0, -1.8]),
        angle_vertical_view=vertical_fov,
    )
)

bunnies = [
    # position, height, yaw (deg), roll (deg), material
    (
        np.array([-0.711505,  -0.014922, -2.211369]),
        0.359449,  35.293303,   8.756683, "speckled",
    ),
    (
        np.array([-0.361233, -0.152324, -2.189348]),
        0.317703, 148.401906,  27.804966, "copper",
    ),
    (
        np.array([-1.056622,  0.389676, -2.459571]),
        0.470683, 168.639220,  19.297998, "honey",
    ),
    (
        np.array([0.0, -0.099413, -1.815651]),
        0.345571, -58.994751, -19.235497, "cyan",
    ),
    (
        np.array([0.36, -0.159688, -1.828653]),
        0.427178, -118.102767, -13.988360, "tungsten",
    ),
    (
        np.array([ 0.819269, -0.376213, -2.029652]),
        0.319755,  17.809696,  26.148741, "pink",
    ),
]

for position, height, yaw, roll, material_name in bunnies:
    bunny = load_upright_bunny(
        position=position,
        height=height,
        yaw_deg=yaw,
        roll_deg=roll,
    )

    if material_name == "speckled":
        texture_unwrapper = BlenderUnwrapper()
        texture_unwrapper.add_rtmesh(bunny)
        texture_unwrapper.smart_unwrap()
        bunny.set_surface(
            surface_type=SurfType.TEXTURE,
            surface_fill=speckle_img,
            material_type=MaterialType.UNLIT,
        )
    elif material_name == "copper":
        bunny.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            material_type=MaterialType.SPECULAR,
            material=MaterialPresets.COPPER,
        )
    elif material_name == "honey":
        bunny.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            material_type=MaterialType.REFRACTIVE,
            material=MaterialPresets.HONEY_LIQUID,
            mesh_type=MeshType.SOLID,
            reference_thickness=0.5,
        )
    elif material_name == "cyan":
        bunny.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            surface_fill=np.array([0.20, 0.85, 0.95]),
            material_type=MaterialType.UNLIT,
        )
    elif material_name == "tungsten":
        bunny.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            material_type=MaterialType.SPECULAR,
            material=MaterialPresets.TUNGSTEN,
        )
    elif material_name == "pink":
        bunny.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            surface_fill=np.array([0.895, 0.44, 0.755]),
            material_type=MaterialType.UNLIT,
        )
    else:
        raise ValueError(f"Unknown bunny material {material_name!r}.")

    flight_scene.add_rtmesh(bunny)

rgb_bmp = ImageFormat(
    output_format=OutputFormat.IMG_BMP_24BIT,
    bit_depth=BitDepth.BIT_8,
    channel_count=ChannelCount.RGB,
    grayscale=False,
)

render_scene(
    image_height=image_height,
    image_width=image_width,
    scene=flight_scene,
    antialiasing_samples=16,
    out_directory_path=flight_dir,
    render_type=RenderType.STATIC,
    texture_sampler=TextureSampler.BSPLINE,
    shading_type=ShadingType.BLENDED,
    image_format=rgb_bmp,
)
