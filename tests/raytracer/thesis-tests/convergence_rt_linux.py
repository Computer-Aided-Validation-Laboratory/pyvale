from global_utils import *
from enum import StrEnum
import csv
import os
import cv2
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

#import smplotlib # For nicer figures (imo), but no need to install if you don't want

from pyvale.sensorsim import EDim
from pyvale.sensorsim.imagetools import ImageTools

from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *

# VERSION WITHOUT BLENDER - SHOULD RUN ON LINUX 
# Make sure you have pre-processed UVs for the meshes (in the same folder as the meshes; so thesis-data/mesh-name)

# ================================================================================
# Positioning
# ================================================================================
# Factored out of the function to allow pre-UV unwrapping of the meshes so we can run on Linux without Blender - otherwise the entire supercomputer plan does not work

TANK_OUTER_LEN = 48.0 # Used to shift the tank and water so their front is at 0.0; also use for pipe to keep the overall scene positioning correct
BEAM_LEN = 48.0 # Used to drop the beam so that it is suspended from the top tank edge
TANK_MID_Z = -(TANK_OUTER_LEN/2) # z-coordinate corresponding to the centre of the tank
TANK_POSITION = np.array([0.0, 0.0, TANK_MID_Z]) # Just set it so that the front face is at z = 0.0; the rest can stay
WATER_POSITION = np.array([0.0, -3, TANK_MID_Z]) # Similar to tank, we just offset y a little bit to ensure there's overlap at the bottom for nested dielectrics
BEAM_POSITION = np.array([0.0, 4, 0.0]) # Doesn't really matter; x = 0.0 is key, we snap y and z based on the tank to reproduce the experiments
# Beam was not perfectly centered in the tank in reality due to the position of the breadboard holes. It was slightly to the right and to the front, so we offset it
BEAM_OFFSET = np.array([1.5, 0.0, 3.5]) 
CAMERA_DISTANCE = 110 # From original set-up
CAMERA_HEIGHT = 10.0
CAMERA__Z= CAMERA_DISTANCE + TANK_MID_Z + BEAM_OFFSET[2] # Wwe account here for the fact that the sample is at some -z position because it's centered within the tank and the offset
#CAMERA_DISTANCE = 260 # Use this to sanity check nested dielectric set-up
SCALE_PX_PER_MM = 45.06
VIEWPORT_Z = CAMERA_DISTANCE - 1 # Viewport position
CAMERA_POSITION = np.array([BEAM_OFFSET[0] - 0.5, CAMERA_HEIGHT, CAMERA_DISTANCE]) # Camera was slightly moved right to center on the beam, too
CAMERA_TARGET = np.array([BEAM_OFFSET[0] - 0.5, CAMERA_HEIGHT, VIEWPORT_Z])

# ================================================================================
# Helpers
# ================================================================================

# Number of anti-aliasing samples at which we end the test regardless of whether the convergence
# has been reached or not
SUBSAMPLE_LIMIT_MAX = 16384 # 2^14, so 14 runs per sample
RMSE_LIMIT_MIN = 1e-6 # When rmse < RMSE_LIMIT_MIN, we say it is converged

# Convenience enums for accessing the right meshes
class Tank(StrEnum):
    RECTANGLE = "rectangular-box"
    PIPE = "pipe"

class Refinement(StrEnum):
    COARSE = "coarse"
    MED_FINE = "med-fine"
    FINE = "fine"

class TestCase(StrEnum):
    AIR_UNLIT = "air_unlit"
    AIR_DIFFUSE = "air_diffuse"
    TANK = "tank"
    WATER = "water"

class Resolution(IntEnum):
    LOW = 128,
    HIGH = 1024

def iter_cases():
    # Iterates over test cases above for plotting
    for name, value in vars(TestCase).items():
        if isinstance(value, TestCase):
            yield name, value

def get_tank_path(tank_access: str, element: Element):
    # Tank access is sth like cwd/thesis-data/rectangular-box/coarse
    return full_path(tank_access + "/tank_surface_" + element.label + ".vtk") # full path to e.g., tank_surface_TRI3.vtk

def tank_uv_path(tank_path: Path, element: Element):
    return Path.with_name(tank_path, "tank_" + element.label + "_uvs.csv")

def get_fill_path(tank_access: str, element: Element):
    return full_path(tank_access + "/fill_surface_" + element.label + ".vtk")

def fill_uv_path(water_path: Path, element: Element):
    return Path.with_name(water_path, "fill_" + element.label + "_uvs.csv")

def sample_uv_path(sample_path: Path, element: Element):
    return Path.with_name(sample_path, "beam_" + element.label + "_uvs.csv")

# ================================================================================
# Convergence tester
# ================================================================================

def bitwise_compare(data_path_new: Path, data_path_prev: Path | None = None, bit_depth: BitDepth = BitDepth.BIT_12):
    """
    Checks if the 8-bit BMP images are bitwise identical.
    Created for the convergence tests, so it assumes that the size of the images is the same and does not check the format.

    If data_path_prev is None, it assumes that we want to compare the same image to itself to verify that the bitmap
    comparison works correctly (i.e., we get 100% similarity.)

    1. Finds the absolute difference between the two images. In the difference array:
        - 0 => Identical
        - != 0 => Different
    2. Counts number of non-zero entries in the difference array.
    3. Verifies if the images are bitwise identical based on the difference array and the total pixel count.
    4. Returns the similarity score.
    """
    max_value = 4095 # Max integer value for 12-bit uint; assign by default
    if bit_depth == BitDepth.BIT_8:
        max_value = 255 # Max integer value for 8-bit uint
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535 # Max integer value for 16-bit uint

    if data_path_prev is None:
        data_path_prev = data_path_new

    # cv2.IMREAD_ANYDEPTH forces OpenCV to keep the 16-bit depth instead of downsampling it to 8-bit
    pixel_array_new = cv2.imread(str(data_path_new), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    pixel_array_prev = cv2.imread(str(data_path_prev), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    # If it loaded as a 3-channel image, grab just the first channel
    if len(pixel_array_new .shape) == 3:
        pixel_array_new  = pixel_array_new [:, :, 0]
    if len(pixel_array_prev.shape) == 3:
        pixel_array_prev = pixel_array_prev[:, :, 0]

    # Difference between the two images
    difference = cv2.absdiff(pixel_array_new, pixel_array_prev)
    total_pixels = pixel_array_new.shape[0] * pixel_array_new.shape[1] # Pixel count in the image
    num_different = cv2.countNonZero(difference) # How many pixels are different
    num_identical = total_pixels - num_different # How many are identical
    
    # Calculate our metrics
    similarity_identical = num_identical / total_pixels # Similarity score based on how many pixels are exactly identical
    rmse = np.sqrt(np.mean((pixel_array_new - pixel_array_prev) ** 2)) # Root mean square error
    # RMSE similarity - based on the RMSE and the max. integer value for this picture (less sensitive to tiny per-pixel differences)
    similarity_rmse = 1.0 - rmse / max_value 

    return rmse, similarity_rmse, similarity_identical

# ================================================================================
# Rendering test 2.1: Convergence, RAY TRACER; version with pre-processed UVs (Linux/supercomputer)-compatible
# ================================================================================
def conv_test_rt(test_case: TestCase, resolution: Resolution = Resolution.HIGH, starting_subsamples: int | None = None, thread_count: int | None = None):
    # NOTE: Resolution is a single digit, because these cameras had square viewport
    # NOTE 2: starting_subsamples must be set for everything that is not AIR_UNLIT
    # 1. Set mesh data that we can set currently
    # Tank
    # Note: Tank has to be a rectangle if we want to test entirely consistent scenes, e.g., just QUAD4
    # This is because pipe didn't really work for quads, and they're a mixture of quads and triangles, which we currently don't support
    tank_access = "thesis-data/" + Tank.RECTANGLE + "/" + Refinement.COARSE # Point the correct mesh locatrions   
    # Set the sample path
    sample_name = "thesis-data/beam/" + Refinement.COARSE + "/beam_surface_"
    # Sample texture
    ref_texture = full_path("thesis-data/texture/speckle.tiff")
    beam_texture = ImageTools.load_image_greyscale(ref_texture) 

    # 2. Settings based on the selected case
    mat_type = MaterialType.UNLIT # Beam material
    if test_case == TestCase.AIR_UNLIT:
        print(f"--------------------------------\nTESTED CASE: AIR UNLIT\n--------------------------------")
        mat_type = MaterialType.UNLIT
        if starting_subsamples is None:
            starting_subsamples = 1
    else:
        # This helps us speed up - it is certain that we will need more subsampling for shading
        if starting_subsamples is None or starting_subsamples < 2:
            raise ValueError("Please base your starting subsample count on the UNLIT case, otherwise this will run for ages.")
        if test_case == TestCase.AIR_DIFFUSE:
            print(f"--------------------------------\nTESTED CASE: AIR DIFFUSE\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
        # For these cases we just accept that we do full shading
        elif test_case == TestCase.TANK:
            print(f"--------------------------------\nTESTED CASE: EMPTY TANK\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
        elif test_case == TestCase.WATER:
            print(f"--------------------------------\nTESTED CASE: TANK WITH WATER\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
    output_dir_name = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value

    # 3. Set camera data
    # Data for Photron Nova S6
    image_width = resolution
    image_height = resolution
    output_format = output_format_phs6
    camera_center = CAMERA_POSITION
    camera_target = CAMERA_TARGET
    # Angle vfov is in degrees
    import math
    distance_to_sample = 110
    #angle_vfov = vertical_fov_from_sensor(sensor_height_phs6, lens_focal_length_phs6)
    angle_vfov = vertical_fov_from_resolution(resolution, SCALE_PX_PER_MM, CAMERA_DISTANCE) # this works better (more truthfully for this)
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)

    # ------------------------------------------------
    # POSITIONING AND SINGLE RENDER TEST SECTION
    # ------------------------------------------------
    # Uncomment to render one test scene or see SceneVisualiser
    # Current - either beam needs to go slightly backwards, or camera further away
    #test_scene = Scene()
    #tank_path = get_tank_path(tank_access, Elements.TRI3)
    #test_tank = any_mesh_to_rtmesh(tank_path, world_position = TANK_POSITION, anchor = Anchor.CENTER) # -24 is half the tank width so its front is at z=0.0
    #water_path = get_fill_path(tank_access, element)
    #water_path = get_fill_path(tank_access, Elements.TRI3)
    #test_water = any_mesh_to_rtmesh(water_path, world_position = WATER_POSITION)
    #test_path = full_path(sample_name + Elements.TRI3.label + ".vtk")
    #test_beam = any_mesh_to_rtmesh(test_path, world_position = BEAM_POSITION, anchor = Anchor.BASE) 
    #snap_to(test_beam, test_tank, Axis.Y, align = (Axis.Z), gap = -BEAM_LEN + 2.0, stack_above = True)
    #test_beam.translate(BEAM_OFFSET)
    #blender_uv.add_rtmesh(test_beam)
    #blender_uv.smart_unwrap()
    #test_beam.set_surface(SurfType.TEXTURE, beam_texture, mat_type)
    #SceneVisualiser([tank, water], cam) # Test overlap
    #SceneVisualiser([test_tank, test_beam], cam) # Test beam position vs. tank and camera
    #test_tank.set_surface(SurfType.FIELD_COLOR,material = MaterialPresets.PLASTIC_ACRYLIC,material_type = MaterialType.REFRACTIVE,mesh_type = MeshType.SOLID,priority = 0)
    #test_water.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.WATER, material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID,priority = 1)
    #test_target = test_dir(BASE_TEST_DIR, output_dir_name)
    #test_scene.add_rtmesh([test_tank, test_water, test_beam])
    #test_scene.add_camera(cam)
    #render_scene(image_height, image_width, test_scene, 1, test_target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.BLENDED, image_format = output_format_phs6)


    # ------------------------------------------------
    # ACTUAL ITERATIVE TESTS FOR ALL ELEMENTS
    # ------------------------------------------------
    # Image always comes out as rtimage_0_cam0 (backend setting)
    # Use this as a base for the newest image, then change the name to keep the data
    
    fresh_filename = "rtimage_0_cam0.tiff"
    for name, element in iter_elements():
        # Announce element with exclamation marks so it stands out from subsample count notifications
        print(f"--------------------------------\n!!! TESTED ELEMENT: {element.label} !!!\n--------------------------------")
        # Create scene and add camera
        scene = Scene()
        scene.add_camera(cam)
        # Add meshes
        # We need to make tank RTMesh regardless of the case to snap the beam position correctly...
        tank_path = get_tank_path(tank_access, element)
        #tank_path = get_tank_path(tank_access, Elements.TRI3) # if pipe
        tank = any_mesh_to_rtmesh(tank_path, world_position = TANK_POSITION, anchor = Anchor.CENTER) # -24 is half the tank width so its front is at z=0.0
        if test_case == TestCase.TANK: # Tank with air
            tank.set_surface(SurfType.FIELD_COLOR,
                         material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE,
                         mesh_type = MeshType.SOLID)
            scene.add_rtmesh(tank)
        elif test_case == TestCase.WATER: # Tank with water
            tank.set_surface(SurfType.FIELD_COLOR,
                         material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE,
                         mesh_type = MeshType.SOLID,
                         priority = 0)
            scene.add_rtmesh(tank)
            water_path = get_fill_path(tank_access, element)
            #water_path = get_fill_path(tank_access, Elements.TRI3) # If pipe
            water = any_mesh_to_rtmesh(water_path, world_position = WATER_POSITION)
            water.set_surface(SurfType.FIELD_COLOR,
                          material = MaterialPresets.WATER,
                          material_type = MaterialType.REFRACTIVE,
                          mesh_type = MeshType.SOLID,
                          priority = 1)
            scene.add_rtmesh(water)
        # Find and add our mesh for the desired element type
        sample_path = full_path(sample_name + element.label + ".vtk")
        beam = any_mesh_to_rtmesh(sample_path, world_position = BEAM_POSITION, anchor = Anchor.BASE)
        snap_to(beam, tank, Axis.Y, align = (Axis.Z), gap = -BEAM_LEN + 2.0, stack_above = True)
        # Append texture
        beam.import_uvs(sample_uv_path(sample_path, element)) # Load pre-processed UVs
        beam.set_surface(SurfType.TEXTURE, beam_texture, mat_type)
        scene.add_rtmesh(beam)
        # Set target locations for the output
        target = test_dir(BASE_TEST_DIR, output_dir_name + "/" + element.label)
        csv_path = target / "convergence_log.csv"
        # Prepare to start the loop
        iteration_number = 0
        subsamples = starting_subsamples # Anti-aliasing samples
        # Open the CSV ready to append
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["iteration", "subsamples", "rmse", "sim_score_rmse", "sim_score_identical"])
            writer.writeheader()
            # Push data from Python buffer to disk
            csvfile.flush()
            os.fsync(csvfile.fileno())
            # Create the first image as our baseline
            render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count)
            # Create the updated filename and change file name
            new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
            os.rename(target.joinpath(fresh_filename), target.joinpath(new_filename))
            # Iteratively refine until we get convergence OR too many samples to be sensible
            while True:
                prev_filename = new_filename
                subsamples *=2
                iteration_number += 1
                render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count)
                 # Rename this file
                new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
                os.rename(target / fresh_filename, target / new_filename)
                # Compare this brand new image with the previous one
                rmse, sim_score_rmse, sim_score_identical = bitwise_compare(target / new_filename, target / prev_filename)
                print(f"-------------------------------- \nCURRENT SUBSAMPLE COUNT: {subsamples}"
                      f"\n\t RMSE: {rmse}\n--------------------------------")
                # Store data in CSV/log
                writer.writerow({
                    "iteration": iteration_number,
                    "subsamples": subsamples,
                    "rmse": rmse,
                    "sim_score_rmse": sim_score_rmse,
                    "sim_score_identical": sim_score_identical})
                csvfile.flush()
                os.fsync(csvfile.fileno())
                
                # Check if we can terminate for this element
                # RMSE condition - the main one
                if rmse < RMSE_LIMIT_MIN: # ~ Root mean square error ~ 0.0 - we converged
                    print("Images perfectly converged. Terminating this case.")
                    break
                # Fallback: subsample count
                if subsamples >= SUBSAMPLE_LIMIT_MAX:
                    print(f"Exceeded the maximum subsample limit of {SUBSAMPLE_LIMIT_MAX}. Terminating this case.")
                    break
        

conv_test_rt(TestCase.AIR_UNLIT, Resolution.LOW, None)




    



