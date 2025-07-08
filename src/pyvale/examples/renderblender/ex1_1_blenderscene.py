# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Blender example: Creating a scene with 2D DIC
---------------------------------------------

This example takes you through creating a scene and adding all the necessary
objects required to represent a 2D DIC setup (camera, lighting and sample).
This example will then show you how to render a single image of this scene.

Test case: mechanical analysis of a plate with a hole loaded in tension.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import pyvale
import mooseherder as mh

# %%
# Here we load in a pre-generated MOOSE finite element simulation dataset that
# comes packaged with pyvale. The simulation is purely mechanical test case in
#  3D of a plate with a hole loaded in tension. A mentioned in previous examples,
# this path can be replaced with your own MOOSE simulation output in exodus
# format (*.e). `mooseherder` is then used to convert the simulation output
# into a `SimData` object.

data_path = pyvale.DataSet.render_mechanical_3d_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()

# %%

# This is then scaled to mm, as all lengths in Blender are to be set in mm.
# The `SimData` object is then converted into a `RenderMeshData` object, as
# this skins the mesh ready to be imported into Blender.
# The `disp_comps` are the expected direction of displacement. Since this is a
# 3D deformation test case, displacement is expected in the x, y and z directions.
disp_comps = ("disp_x","disp_y", "disp_z")
sim_data = pyvale.scale_length_units(scale=1000.0,
                                     sim_data=sim_data,
                                     disp_comps=disp_comps)

render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

# %%
# Firstly, a save path must be set.
# In order to do this a base path must be set. Then all the generated files will
# be saved to a subfolder within this specified base directory
# (e.g. blenderimages).
# If no base directory is specified, it will be set as your home directory.

base_dir = Path.cwd()

# %%
# Creating the scene
# ^^^^^^^^^^^^^^^^^^
# In order to create a DIC setup in Blender, first a scene must be created.
# A scene is a holding space for all of your objects (e.g. camera(s), light(s)
# and sample(s)).
# A scene is initialised using the `BlenderScene` class. All the subsequent
# objects and actions necessary are then methods of this class.

scene = pyvale.BlenderScene()

# %%
# The next thing that can be added to the scene is a sample.
# This is done by passing in the `RenderMeshData` object.
# It should be noted that the mesh will be centred on the origin to allow for
# the cameras to be centred on the mesh.
# Once the part is added to the Blender scene, it can be both moved and rotated.

part = scene.add_part(render_mesh, sim_spat_dim=3)
# Set the part location
part_location = np.array([0, 0, 0])
pyvale.BlenderTools.move_blender_obj(part=part, pos_world=part_location)
# Set part rotation
part_rotation = Rotation.from_euler("xyz", [0, 0, 0], degrees=True)
pyvale.BlenderTools.rotate_blender_obj(part=part, rot_world=part_rotation)

# %%
# A camera can then be added to the scene.
# To initialise a camera, the camera parameters must be specified using the
# `CameraData` dataclass. Note that all lengths / distances inputted are in mm.
# This camera can then be added to the Blender scene.
# The camera can also be moved and rotated.

cam_data = pyvale.CameraData(pixels_num=np.array([1540, 1040]),
                            pixels_size=np.array([0.00345, 0.00345]),
                            pos_world=(0, 0, 400),
                            rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                            roi_cent_world=(0, 0, 0),
                            focal_length=15.0)
camera = scene.add_camera(cam_data)
camera.location = (0, 0, 410)
camera.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle

# %%
# A light can the be added to the scene.
# Blender offers different light types: Point, Sun, Spot and Area.
# The light can also be moved and rotated like the camera.

light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                     pos_world=(0, 0, 400),
                                     rot_world=Rotation.from_euler("xyz",
                                                                   [0, 0, 0]),
                                     energy=1)
light = scene.add_light(light_data)
light.location = (0, 0, 410)
light.rotation_euler = (0, 0, 0)

# %%
# A speckle pattern can then be applied to the sample.
# Firstly, the material properties of the sample must be specified, but these
# will all be defaulted if no inputs are provided.
#The speckle pattern can then be specified by providing a path to an image file
# with the pattern.
# The mm/px resolution of the camera must also be specified in order to
# correctly scale the speckle pattern.
# It should be noted that for a bigger camera or sample you may need to generate
# a larger speckle pattern.

material_data = pyvale.BlenderMaterialData()
speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()

mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
scene.add_speckle(part=part,
                  speckle_path=speckle_path,
                  mat_data=material_data,
                  mm_px_resolution=mm_px_resolution)

# %%
# Rendering an image
# ^^^^^^^^^^^^^^^^^^
# Once all the objects have been added to the scene, an image can be rendered.
# Firstly, all the rendering parameters must be set, including parameters such as
# the number of threads to use.

render_data = pyvale.RenderData(cam_data=cam_data,
                                base_dir=base_dir,
                                threads=8)

# %%
# A single image of the scene can then be rendered.
# If `stage_image` is set to True, the image will be saved to disk, converted to
# an array, deleted and the image array will be returned. This is due to the
# fact that an image cannot be saved directly as an array through Blender.

scene.render_single_image(stage_image=False,
                          render_data=render_data)

# %%
# The rendered image will be saved to this filepath:

print("Save directory of the image:", (render_data.base_dir / "blenderimages"))

# %%
# There is also the option to save the scene as a Blender project file.
# This file can be opened with the Blender GUI to view the scene.

pyvale.BlenderTools.save_blender_file(base_dir)

