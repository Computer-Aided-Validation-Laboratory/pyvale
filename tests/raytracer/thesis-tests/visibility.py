import numpy as np
import os
import cv2

from global_utils import *
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *

# thesis-data -> visibility -> rabbit_elem
def get_rabbit_path(rabbit_access: str, element: Element):
    return full_path(rabbit_access + element.label.lower())

class ElementPairs:
    TRIS = (Elements.TRI3, Elements.TRI6)
    QUADS1 = (Elements.QUAD4, Elements.QUAD8)
    QUADS2 = (Elements.QUAD4, Elements.QUAD9)
    QUADS3 = (Elements.QUAD8, Elements.QUAD9)

# Add this to iterate over pairs more easily
pairs = [value
    for name, value in ElementPairs.__dict__.items()
    if not name.startswith('_') and isinstance(value, tuple)]

def difference_image(data_path_single: Path, data_path_both: Path, pair_label: str, bit_depth: BitDepth = BitDepth.BIT_12):
    max_value = 4095 # Max integer value for 12-bit uint; assign by default
    if bit_depth == BitDepth.BIT_8:
        max_value = 255 # Max integer value for 8-bit uint
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535 # Max integer value for 16-bit uint

    # cv2.IMREAD_ANYDEPTH forces OpenCV to keep the 16-bit depth instead of downsampling it to 8-bit
    pixel_array_single = cv2.imread(str(data_path_single), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    pixel_array_both = cv2.imread(str(data_path_both), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    # If it loaded as a 3-channel image, grab just the first channel
    if len(pixel_array_single .shape) == 3:
        pixel_array_single  = pixel_array_single [:, :, 0]
    if len(pixel_array_both.shape) == 3:
        pixel_array_both = pixel_array_both[:, :, 0]

    # Difference between the two images
    difference = cv2.absdiff(pixel_array_single, pixel_array_both).astype(np.uint16)

    output_dir = data_path_single.parent
    output_name = f"difference_" + pair_label + ".tiff"
    output_path = output_dir / output_name

    ok = cv2.imwrite(str(output_path), difference)
    if not ok:
        raise IOError(f"Could not write output image: {output_path}")

def visibility_test():
    # 1. Overall data access etc.
    # Data is stored in thesis-data/visibility/rabbit_ELEMENT, so build the base for bunny meshes
    rabbit_access = "thesis-data/visibility/rabbit_"
    # Image always comes out as rtimage_0_cam0 (backend setting)
    # Use this as a base for the newest image, then change the name to keep the data
    fresh_filename = "rtimage_0_cam0.tiff"
    # Target directory for rendered images
    target_dir = test_dir(BASE_TEST_DIR, "visibility")

    # 2. Camera
    image_width = 400
    image_height = 400
    output_format = output_format_phs6
    camera_center = np.array([0.0, 0.0, 50])
    camera_target = np.array([0.0, 0.0, 0.0])
    angle_vfov = 20
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)

    # 3. Iterate over pairs to generate images
    for pair in pairs:
        # Extract elements
        front_elem, back_elem = pair
        # Get rabbits
        rabbit_front_path = get_rabbit_path(rabbit_access, front_elem)
        rabbit_back_path = get_rabbit_path(rabbit_access, back_elem)
        bunny_front = simdata_csv_to_rtmesh(rabbit_front_path,world_position=np.array([-2.5, -0.5, -10]),target_size=12,size_axis=Axis.X)
        bunny_back = simdata_csv_to_rtmesh(rabbit_back_path,world_position=np.array([2.5, 1.0, -10.5]),target_size=12,size_axis=Axis.X)

        # Check positioning if needed
        #SceneVisualiser([bunny_front], cam)
        #SceneVisualiser([bunny_front, bunny_back], cam)

        # Set surfaces
        bunny_front.set_surface(surface_fill=np.array([1.0, 1.0, 1.0]))
        bunny_back.set_surface(surface_fill=np.array([0.5, 0.5, 0.5]))
        # Set scene and add camera
        scene = Scene()
        scene.set_background(np.array([0.0, 0.0, 0.0])) # Black background
        scene.add_camera(cam)
        # Generate naming for output so it's clear + we avoid overwriting
        pair_label = str(front_elem.label) + str(back_elem.label) # e.g., TRI3TRI6

        # 3.1. Image of just the front mesh
        scene.add_rtmesh(bunny_front)
        render_scene(image_height, image_width, scene, 1, target_dir, RenderType.STATIC, image_format=output_format)
        single_filename = "front_" + pair_label + ".tiff"
        # Rename so we don't overwrite it
        os.rename(target_dir.joinpath(fresh_filename), target_dir.joinpath(single_filename))

        # 3.2. Both bunnies
        scene.add_rtmesh(bunny_back)
        render_scene(image_height, image_width, scene, 1, target_dir, RenderType.STATIC, image_format=output_format)
        new_filename = "pair_" + pair_label + ".tiff"
        os.rename(target_dir.joinpath(fresh_filename), target_dir.joinpath(new_filename))

        # 3.3. Difference image
        difference_image(target_dir / single_filename, target_dir / new_filename, pair_label)

visibility_test()