# TESTING BATTLEFIELD

from ast import List
import numpy as np
from dataclasses import dataclass, field
import timeit

import pyvale.dataset as dataset
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtsimdataloader import add_mesh_to_scene
from pyvale.raytracer.rtscene import Scene

#################################################### INPUT #####################################################
# Output image dimensions
image_width = 400  # px
aspect_ratio = 16.0 / 9.0
image_height = int(image_width / aspect_ratio)  # px
number_of_samples = 1; # for anti-aliasing
# Assume single camera for now - but can be extended to multiple cameras later
camera_center = np.array([-0.5, 1.1, 1.1])
camera_target = np.array([0, 0, -1])
angle_vertical_view = 90  # degrees

################################################ SCENE BUILDER ###############################################
# Targets:
# Input: meshes for the objects + object coordinates in world coordinate system + cameraS(!) position (lights to be added)
# Functions: load meshes, specify materials (later), specify cameras and their params, create the scene
# Output: a scene dataclass containing all information to render the scene + an option to save the scene dataclass to file for later use

# Tests: For now just load simple sample data, assuming 1 of each element
scene = Scene()

# Create a camera
camera1 = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view) # Camera for tests
#camera0 = Camera(image_width, image_height) # Default camera (parameters i.e., at world origin, no funny angles) for tests
camera1.add_camera_to_scene(scene);

# Load sample data file with a simple rectangular block in 3D to test image rendering algorithm. Returns a file path to an exodus file

data_path = dataset.render_simple_block_path() 
add_mesh_to_scene(scene, data_path)
# add_mesh_to_scene(scene, data_path, world_position=[-2.0, -10.0, -2.0], scale=500)

# Lights - to be added later


from rtmaincpp import cpp_render_scene
print(timeit.timeit("cpp_render_scene(image_height, image_width, number_of_samples, scene.scene_connectivity, scene.scene_coords, scene.scene_face_colors, scene.scene_camera_center, scene.scene_pixel_00_center, scene.scene_matrix_pixel_spacing)", globals=globals(), number=1))