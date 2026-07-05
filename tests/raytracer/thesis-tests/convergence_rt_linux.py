from global_utils import *
from convergence_common import *
import csv
import os
import timeit

#import smplotlib # For nicer figures (imo), but no need to install if you don't want

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
# Factored out of the function to allow pre-UV unwrapping of the meshes so we can run on Linux without Blender - otherwise the entire supercomputer plan does not work

# Number of anti-aliasing samples at which we end the test regardless of whether the convergence
# has been reached or not
SUBSAMPLE_LIMIT_MAX = 2**26

# ================================================================================
# Rendering test 2.1: Convergence, RAY TRACER; version with pre-processed UVs (Linux/supercomputer)-compatible
# ================================================================================
def conv_test_rt(test_case: TestCase,
                 resolution: Resolution = Resolution.HIGH,
                 starting_subsamples: int | None = None,
                 thread_count: int | None = None,
                 element_idx: int | None = None, # 0 = QUAD4, 1 = QUAD8, 2 = QUAD9, 3 = TRI3, 4 = TRI6; as in Elements in global_utils
                 single_image: bool = False,
                 subsample_limit: int | None = None): # If true, renders only one image at the given starting_subsamples
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

    SUBSAMPLE_LIMIT = SUBSAMPLE_LIMIT_MAX
    # Custom subsample limit - for convenience
    if subsample_limit is not None and subsample_limit > 1:
            SUBSAMPLE_LIMIT = subsample_limit
    if starting_subsamples is None:
            starting_subsamples = 1

    # 2. Settings based on the selected case
    mat_type = MaterialType.UNLIT # Beam material
    if test_case == TestCase.AIR_UNLIT:
        print(f"--------------------------------\nTESTED CASE: AIR UNLIT\n--------------------------------")
        mat_type = MaterialType.UNLIT
    else:

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
    #angle_vfov = vertical_fov_from_sensor(sensor_height_phs6, lens_focal_length_phs6)
    angle_vfov = vertical_fov_from_resolution(resolution, SCALE_PX_PER_MM, CAMERA_DISTANCE) # this works better (more truthfully for this)
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)

    # ------------------------------------------------
    # ACTUAL ITERATIVE TESTS FOR ALL ELEMENTS
    # ------------------------------------------------
    # Image always comes out as rtimage_0_cam0 (backend setting)
    # Use this as a base for the newest image, then change the name to keep the data
    
    fresh_filename = "rtimage_0_cam0.tiff"
    #for name, element in iter_elements():
    # Workaround to only render one element without rewriting much
    for idx, (name, element) in enumerate(iter_elements()):
        if element_idx is not None and idx != element_idx:
            continue  # Skip all elements except the one we want; this is to run array jobs etc.
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
        # Storing render times (separately since the rmse etc. are difference-based)
        time_csv_path = target / "render_time_log.csv"
        time_log_exists = os.path.isfile(time_csv_path)
        time_mode = "a" if time_log_exists else "w" # Append if time log already exists
        subsamples = starting_subsamples # Anti-aliasing samples
        if single_image:
            time = timeit.timeit(lambda: render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count), number=1)
            with open(time_csv_path, mode=time_mode, newline="", encoding="utf-8") as timefile:
                time_writer = csv.DictWriter(timefile, fieldnames=["subsamples","time (s)"])
                if not time_log_exists:
                    time_writer.writeheader()
                time_writer.writerow({"subsamples": subsamples, "time (s)": time})
                timefile.flush()
                os.fsync(timefile.fileno())
        else:
            csv_path = target / "convergence_log.csv"
            # No existence checking for that because we would have to fiddle with checking image files etc., and there are
            # convenience functions to re-run all convergence comparisons from existing images already
            # Prepare to start the loop
            iteration_number = 0
            # Open the CSV ready to append
            with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile, \
                open(time_csv_path, mode=time_mode, newline="", encoding="utf-8") as timefile:
                writer = csv.DictWriter(csvfile, fieldnames=["iteration", "subsamples", "rmse", "sim_score_rmse", "sim_score_identical"])
                writer.writeheader()
                # Push data from Python buffer to disk
                csvfile.flush()
                os.fsync(csvfile.fileno())
                # Create the first image as our baseline
                time = timeit.timeit(lambda: render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count), number=1)
                #render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count)
                # Create the updated filename and change file name
                new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
                os.rename(target.joinpath(fresh_filename), target.joinpath(new_filename))
                # Store the time
                time_writer = csv.DictWriter(timefile, fieldnames=["subsamples","time (s)"])
                if not time_log_exists:
                    time_writer.writeheader()
                time_writer.writerow({"subsamples": subsamples, "time (s)": time})
                timefile.flush()
                os.fsync(timefile.fileno())
                # Iteratively refine until we get convergence OR too many samples to be sensible
                while True:
                    prev_filename = new_filename
                    subsamples *=2
                    iteration_number += 1
                    time = timeit.timeit(lambda: render_scene(image_height, image_width, scene, subsamples, target, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format_phs6, omp_thread_count = thread_count), number=1)
                    # Rename this file
                    new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
                    os.rename(target / fresh_filename, target / new_filename)
                    # Store time data
                    time_writer.writerow({"subsamples": subsamples, "time (s)": time})
                    timefile.flush()
                    os.fsync(timefile.fileno())
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
                    if subsamples >= SUBSAMPLE_LIMIT:
                        print(f"Exceeded the maximum subsample limit of {SUBSAMPLE_LIMIT_MAX}. Terminating this case.")
                        break

#conv_test_rt(TestCase.AIR_UNLIT, Resolution.LOW, 1, None, 1, True)