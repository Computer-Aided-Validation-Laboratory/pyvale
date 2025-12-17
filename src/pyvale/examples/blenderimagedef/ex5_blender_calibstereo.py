# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Rendering calibration images
---------------------------------------------

This example takes you through how to render calibration images for a given DIC
setup using in-built tools for calibration target generation.

Note that this example produces a significant number of images and will take a 
long time to run.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.blender as blender
import pyvale.dataset as dataset

# %%
# Firstly, a save path must be set.
# In order to do this a base path must be set. Then all the generated files will
# be saved to a subfolder within this specified base directory
# (e.g. blenderimages).
# If no base directory is specified, it will be set as your home directory.

base_dir = Path.cwd() / "pyvale-output" / "blender-calib"
if not base_dir.is_dir():
    base_dir.mkdir(parents=True, exist_ok=True)



# %%
# Creating the scene
# ^^^^^^^^^^^^^^^^^^
# In order to create a DIC setup in Blender, first a scene must be created.
# A scene is initialised using the `BlenderScene` class. All the subsequent
# objects and actions necessary are then methods of this class.
scene = blender.Scene()

# %%
# The next thing to add to the scene is the calibration target.
# This is done by specifing the size of calibration target to add to the scene
# by passing in an array of (width, height, depth).
# The calibration target being simulated here is 12 x 9 with 10 mm spacing.

target = scene.add_cal_target(target_size=np.array([150, 100, 10]))

# %%
# The cameras can then be initialised. A stereo camera system is defined by a
# `CameraStereo` object, which contains the intrinsic parameters of both cameras
# as well as the extrinsic parameters between them.
# There are two ways to initialise a `CameraStereo` object.
# One way is to specify the camera parameters separately for each camera, create
# a `CameraStereo` object, and then add the stereo system using the
# `add_stereo_system` method.
# The other method is to use a convenience function, as shown below.
# This requires you to first initialise one camera. Then you can choose between
# either a face-on or symmetric stereo system. Then, either of the
# `symmetric_stereo_cameras` or `faceon_stereo_cameras` functions can be used to
# initialise a `CameraStereo` object. The only input required to these functions
# are the camera parameters for the first camera, and the desired stereo angle
# between the two. The cameras can then be added to the Blender scene using the
# `add_stereo_system` method.
cam_data_0 = sens.CameraData(pixels_num=np.array([1540, 1040]),
                               pixels_size=np.array([0.00345, 0.00345]),
                               pos_world=np.array([0, 0, 400]),
                               rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                               roi_cent_world=(0, 0, 0),
                               focal_length=15.0)
# Set this to "symmetric" to get a symmetric stereo system or set this to
# "faceon" to get a face-on stereo system
stereo_setup = "faceon"
if stereo_setup == "symmetric":
    stereo_system = sens.CameraTools.symmetric_stereo_cameras(
        cam_data_0=cam_data_0,
        stereo_angle=15.0)
elif stereo_setup == "faceon":
    stereo_system = sens.CameraTools.faceon_stereo_cameras(
        cam_data_0=cam_data_0,
        stereo_angle=15.0)
else:
    raise ValueError(f"Unknown stereo_setup: {stereo_setup}")

scene.add_stereo_system(stereo_system)

# %%
# Since this scene contains a stereo DIC system, a calibration file will be
# required to run the images through a DIC engine.
# A calibration file can be generated directly from the `CameraStereo` object.
# The calibration file will be saved in `YAML` format. However, if you wish to
# use MatchID to process the images, `save_calibration_mid` can be used instead
# to save the calibration in a format readable by MatchID.
# The calibration file will be saved to a sub-directory of the base directory
# called "calibration".
# This calibration file with "perfect" parameters can be used as a comparitive
# benchmark to the calibration gained from running the calibration files through
# a DIC engine.
stereo_system.save_calibration(base_dir)

# %%
# A light can the be added to the scene.
# Blender offers different light types: Point, Sun, Spot and Area.
# The light can also be moved and rotated like the camera.

light_data = blender.LightData(type=blender.LightType.POINT,
                                     pos_world=(0, 0, 200),
                                     rot_world=Rotation.from_euler("xyz",
                                                                  [0, 0, 0]),
                                     energy=1)
light = scene.add_light(light_data)
light.location = (0, 0, 210)
light.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle

# %%
# The calibration target pattern can then be added to the calibration target
# object.
# This is added in the same way that a speckle pattern is added to a sample.
# However, it is important to set the `cal` flag to True, as this means that the
# calibration target pattern will not be scaled in the same way as a speckle
# pattern.

material_data = blender.MaterialData()
cal_target = dataset.cal_target()
mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data_0)
scene.add_speckle(part=target,
                  speckle_path=cal_target,
                  mat_data=material_data,
                  mm_px_resolution=mm_px_resolution,
                  cal=True)

# %%
# Rendering a set of images
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Once all the objects have been added to the scene, a set of images can be
# rendered.Firstly, all the rendering parameters must be set, including
# parameters such as the number of threads to use.

render_data = blender.RenderData(cam_data=(stereo_system.cam_data_0,
                                            stereo_system.cam_data_1),
                                base_dir=base_dir,
                                dir_name="blender-stereo-cal")

# %%
# The parameters for the calibration target's movement can then be set. This is
# done by setting the minimum and maximum angle and plunge limits, as well as
# the step value that they should be increased by. The x and y limit of the
# calibration target's movement (from the origin) can also be set if you wish to
# perform a calibration for a constrained optical setup. If these limits are not
# passed in they will be initialised from the FOV to cover the whole FOV of the
# cameras.

calibration_data = blender.CalibrationData(angle_lims=(-10, 10),
                                          angle_step=20,
                                          plunge_lims=(-5, 5),
                                          plunge_step=10)

# %%
# It is then possible to check the number of calibration images that will be
# rendered before rendering them. The only input that is needed is the
# `calibration_data` specified above.

number_calibration_images = blender.Tools.number_calibration_images(calibration_data)
print("Number of calibration images to be rendered:", number_calibration_images)

# %%
# The calibration images can then be rendered. This function will move the
# calibration target according to movement limits set above, and will also move
# the target rigidly across the FOV of the camera, in order to characterise the
# entire FOV of the cameras.
blender.Tools.render_calibration_images(render_data,
                                        calibration_data,
                                        target)

# %%
# The rendered images will be saved to this filepath:

print("Save directory of the images:", (render_data.base_dir / render_data.dir_name))

# %%
# There is also the option to save the scene as a Blender project file.
# This file can be opened with the Blender GUI to view the scene.

blender.Tools.save_blender_file(base_dir=base_dir,over_write=True)
