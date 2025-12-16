"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import pytest
import numpy.testing as npt
import numpy as np
from scipy.spatial.transform import Rotation
import bpy

# Pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataset as dataset
import pyvale.mooseherder as mh
import pyvale.blender as blender

# NOTE: These test may fail if you are not running bpy 4.4.0
# It is a known issue that the render outputs between bpy 4.2.0 and 4.4.0 slightly
# differ. If you are running bpy 4.2.0, the following tests will fail:
# - test_camera_from_resolution
# - test_render_engine[Eeeve render engine]
# - test_stereo_convenience_cameras[Symmetric convenience function]
# - test_stereo_deformation
# These tests simply fail due to the version difference of bpy and the consequential
# difference in rendering.

@pytest.fixture
def sample_scene():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = sens.scale_length_units(1000.0,sim_data,disp_comps)
    render_mesh = sens.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=sens.EDim.TWOD,
                                        field_disp_keys=disp_comps)

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(pixels_num=np.array([20, 20]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 500),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15)
    camera = scene.add_camera(cam_data)
    light_data = blender.LightData(type=blender.LightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = scene.add_light(light_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(part=part,
                    speckle_path=speckle_path,
                    mat_data=material_data,
                    mm_px_resolution=mm_px_resolution)
    return render_mesh, part, cam_data, scene

@pytest.fixture
def sample_scene_no_light():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = sens.scale_length_units(1000.0,sim_data,disp_comps)
    render_mesh = sens.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=sens.EDim.TWOD,
                                        field_disp_keys=disp_comps)

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(pixels_num=np.array([20, 20]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 500),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15)
    camera = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    return cam_data, scene

@pytest.fixture
def sample_scene_no_cam():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = sens.scale_length_units(1000.0,sim_data,disp_comps)
    render_mesh = sens.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=2,
                                        field_disp_keys=disp_comps)

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    light_data = blender.LightData(type=blender.LightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = scene.add_light(light_data)

    return part, scene

@pytest.mark.parametrize(
    "energy, output",
    [
        pytest.param(0.5, "half_watt_lighting", id="Normal lighting - 0.5W"),
        pytest.param(3, "three_watt_lighting", id="Normal lighting - 3W")
    ]
)
def test_lighting_energy(energy, output, sample_scene_no_light, request, tmp_path):
    cam_data, scene = sample_scene_no_light
    light_data = blender.LightData(type=blender.LightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=energy)
    light = scene.add_light(light_data)
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)

@pytest.mark.parametrize(
    "pixels_num, output",
    [
        pytest.param(np.array([10, 20]), "vertical_cam", id="Vertical camera orientation"),
        pytest.param(np.array([20, 10]), "horizontal_cam", id="Horizontal camera orientation")
    ]
)
def test_camera_shape(pixels_num, output, request, sample_scene_no_cam, tmp_path):
    part, scene = sample_scene_no_cam
    cam_data = sens.CameraData(pixels_num=pixels_num,
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 500),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15)
    camera = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)

def test_camera_from_resolution(sample_scene_no_cam, cam_from_resolution, tmp_path):
    part, scene = sample_scene_no_cam
    pixels_num = np.array([20, 20])
    pixels_size = np.array([0.00345, 0.00345])
    working_dist = 500
    resolution = 0.1
    cam_data = sens.CameraTools.blender_camera_from_resolution(pixels_num,
                                                     pixels_size,
                                                     working_dist,
                                                     resolution)
    cam = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)

    npt.assert_allclose(image_array, cam_from_resolution, atol=2, rtol=0)

def test_deformation(sample_scene, deformed_images, tmp_path):
    (render_mesh, part, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir = tmp_path)
    image_arrays = scene.render_deformed_images(render_mesh,
                                                              sim_spat_dim=3,
                                                              render_data=render_data,
                                                              part=part,
                                                              stage_image=True)

    npt.assert_allclose(image_arrays[:, :, 10], deformed_images, atol=2, rtol=0)

@pytest.mark.parametrize(
    "samples, output",
    [
        pytest.param(4, "samples_four", id="Normal sample number - 4"),
        pytest.param(12, "samples_twelve", id="Normal sample number - 12")
    ],
)
def test_samples_happy(samples, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path,
                                    samples=samples)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)

def test_samples_unhappy(sample_scene, tmp_path):
    samples = 2.5
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path,
                                    samples=samples)
    with pytest.raises(TypeError):
        image_array = scene.render_single_image(stage_image=True,
                                                              render_data=render_data)


@pytest.mark.parametrize(
    "bounces, output",
    [
        pytest.param(2, "bounces_two", id="Normal bounces number - 2"),
        pytest.param(100, "bounces_hundred", id="Normal bounces number -100")
    ]
)
def test_max_bounces_happy(bounces, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path,
                                    max_bounces=bounces)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)

def test_max_bounces_unhappy(sample_scene, tmp_path):
    bounces = 2.5
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path,
                                    max_bounces=bounces)
    with pytest.raises(TypeError):
        image_array = scene.render_single_image(stage_image=True,
                                                              render_data=render_data)

@pytest.mark.parametrize(
        "engine, output",
        [
            pytest.param(blender.RenderEngine.CYCLES, "cycles_engine", id="Cycles render engine"),
            pytest.param(blender.RenderEngine.EEVEE, "eevee_engine", id="Eevee render engine")
        ]
)
def test_render_engine(engine, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    if engine == blender.RenderEngine.EEVEE:
        gpu_present = blender.Tools.check_for_GPU()
        if gpu_present is False:
            pytest.skip("Unsupported hardware for EEVEE")

    render_data = blender.RenderData(cam_data=cam_data,
                                    base_dir=tmp_path,
                                    engine=engine)
    image_array = scene.render_single_image(stage_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


@pytest.fixture
def half_watt_lighting():
    return np.array([[33, 40, 35, 37, 43, 46, 47, 48, 46, 50, 53, 52, 51, 51, 52, 45,
        46, 54, 58, 47],
       [37, 44, 32, 37, 40, 40, 42, 41, 41, 43, 44, 44, 45, 48, 49, 43,
        41, 48, 48, 42],
       [37, 40, 29, 41, 44, 42, 42, 38, 33, 32, 31, 32, 34, 40, 40, 37,
        39, 47, 47, 43],
       [37, 34, 28, 39, 36, 27, 20, 13,  5,  3,  3,  4,  6, 15, 24, 28,
        36, 46, 47, 42],
       [43, 35, 29, 34, 19,  3,  0,  1,  0,  1,  0,  0,  0,  0,  3, 19,
        33, 45, 48, 44],
       [45, 38, 36, 34,  4,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  3,
        25, 45, 48, 44],
       [46, 37, 41, 30,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        16, 43, 47, 40],
       [45, 37, 39, 22,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        10, 40, 45, 37],
       [47, 38, 36, 16,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
         8, 41, 47, 34],
       [46, 40, 38, 11,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
         7, 42, 48, 33],
       [43, 41, 41,  9,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
         9, 42, 46, 31],
       [36, 44, 46, 11,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        11, 42, 45, 33],
       [34, 45, 48, 16,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        15, 42, 45, 36],
       [36, 40, 45, 26,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        21, 43, 47, 36],
       [38, 40, 42, 33,  6,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  5,
        31, 46, 49, 36],
       [39, 42, 41, 40, 17,  0,  0,  0,  1,  1,  1,  1,  1,  0,  5, 26,
        39, 47, 49, 33],
       [37, 40, 36, 45, 24,  4,  1,  1,  1,  1,  1,  2,  5, 16, 31, 37,
        41, 45, 48, 29],
       [45, 40, 30, 45, 32, 25, 26, 25, 24, 30, 33, 34, 35, 42, 44, 42,
        37, 40, 49, 30],
       [46, 35, 29, 47, 43, 43, 49, 49, 45, 46, 47, 46, 47, 48, 45, 37,
        31, 36, 49, 34],
       [36, 27, 24, 37, 36, 33, 39, 49, 48, 37, 35, 37, 36, 34, 38, 25,
        20, 25, 42, 31]])

@pytest.fixture
def three_watt_lighting():
    return np.array([[ 97, 113, 102, 105, 118, 126, 127, 128, 126, 132, 137, 135, 134,
        135, 136, 123, 125, 140, 146, 127],
       [107, 121,  95, 107, 113, 114, 116, 116, 114, 119, 121, 122, 124,
        129, 130, 120, 115, 128, 129, 117],
       [106, 113,  89, 115, 121, 118, 118, 109,  99,  94,  93,  95, 101,
        112, 113, 107, 111, 127, 127, 120],
       [106, 101,  86, 110, 104,  84,  66,  51,  32,  23,  23,  24,  31,
         54,  77,  86, 106, 124, 126, 118],
       [118, 102,  88,  99,  64,  18,   1,   1,   0,   1,   0,   0,   0,
          1,  23,  64,  99, 124, 128, 120],
       [124, 108, 104,  98,  28,   0,   0,   1,   0,   0,   0,   0,   0,
          1,   1,  23,  79, 123, 129, 121],
       [124, 108, 115,  90,   7,   0,   1,   0,   1,   0,   1,   0,   0,
          0,   0,   0,  57, 118, 126, 113],
       [123, 107, 110,  72,   1,   0,   0,   1,   0,   0,   0,   0,   1,
          0,   1,   1,  44, 114, 125, 108],
       [127, 110, 106,  59,   1,   1,   1,   0,   0,   1,   1,   0,   1,
          1,   1,   0,  39, 114, 127, 101],
       [126, 112, 110,  47,   0,   1,   0,   1,   0,   1,   0,   1,   1,
          0,   1,   1,  37, 116, 129,  97],
       [119, 116, 115,  42,   1,   1,   0,   1,   1,   0,   0,   0,   0,
          1,   1,   1,  41, 118, 126,  93],
       [104, 121, 124,  47,   0,   1,   1,   0,   1,   0,   1,   0,   0,
          0,   0,   0,  47, 117, 123,  99],
       [100, 123, 129,  59,   1,   1,   1,   0,   0,   1,   1,   1,   0,
          0,   0,   1,  56, 117, 124, 104],
       [104, 113, 124,  80,   5,   1,   0,   1,   0,   1,   1,   1,   0,
          1,   0,   3,  71, 120, 127, 104],
       [109, 114, 117,  98,  33,   0,   0,   1,   0,   0,   0,   1,   1,
          0,   0,  31,  95, 125, 131, 104],
       [110, 117, 115, 112,  59,   1,   0,   0,   1,   1,   1,   1,   1,
          0,  27,  81, 110, 127, 130,  97],
       [108, 113, 105, 122,  77,  25,   9,   5,   5,   6,  10,  19,  29,
         58,  94, 107, 114, 123, 128,  90],
       [123, 114,  91, 123,  95,  81,  83,  79,  78,  91,  98, 100, 104,
        117, 121, 117, 109, 113, 130,  91],
       [126, 103,  88, 128, 120, 120, 130, 130, 123, 126, 128, 126, 128,
        129, 123, 106,  93, 104, 131, 102],
       [107,  84,  77, 106, 105,  97, 111, 129, 129, 106, 102, 106, 104,
         99, 110,  80,  67,  80, 117,  93]])

@pytest.fixture
def vertical_cam():
    return np.array([[47, 53, 56, 61, 64, 66, 66, 63, 63, 61],
       [53, 54, 55, 58, 61, 65, 68, 65, 61, 59],
       [65, 62, 62, 62, 59, 60, 66, 67, 66, 61],
       [67, 65, 65, 65, 61, 58, 61, 65, 66, 63],
       [65, 61, 62, 63, 59, 58, 60, 63, 63, 60],
       [66, 58, 60, 60, 56, 54, 57, 58, 60, 61],
       [67, 51, 47, 47, 42, 40, 43, 49, 56, 58],
       [59, 38, 23, 14,  7,  9, 17, 35, 52, 52],
       [63, 37,  8,  1,  0,  1,  2, 24, 47, 52],
       [68, 42,  4,  0,  0,  0,  1, 18, 38, 45],
       [61, 41,  5,  0,  1,  0,  0, 18, 37, 43],
       [54, 38,  6,  1,  1,  1,  1, 22, 41, 49],
       [50, 41, 27, 10,  4,  5, 14, 30, 44, 50],
       [61, 58, 51, 43, 38, 36, 37, 42, 55, 60],
       [57, 57, 57, 56, 56, 58, 55, 48, 56, 65],
       [53, 55, 56, 57, 54, 63, 65, 53, 59, 68],
       [48, 54, 52, 57, 52, 58, 65, 53, 60, 66],
       [45, 49, 47, 54, 53, 54, 60, 55, 61, 64],
       [50, 46, 48, 55, 56, 55, 57, 54, 61, 66],
       [53, 53, 50, 51, 54, 55, 56, 55, 66, 65]])

@pytest.fixture
def horizontal_cam():
    return np.array([[101,  96,  77,  41,   0,   0,   1,   0,   0,   0,   0,   0,   1,
          1,   0,   0,  21,  82,  87,  61],
       [ 88,  71,  58,  26,   1,   0,   1,   1,   0,   0,   0,   1,   1,
          1,   0,   1,  16,  67,  66,  51],
       [ 62,  46,  43,  10,   0,   0,   0,   1,   0,   0,   0,   0,   0,
          1,   1,   1,  11,  62,  66,  44],
       [ 49,  50,  58,  12,   1,   0,   0,   1,   0,   0,   0,   0,   1,
          0,   1,   1,   8,  59,  68,  41],
       [ 68,  65,  59,  11,   0,   1,   0,   1,   0,   1,   0,   1,   1,
          0,   1,   1,   7,  58,  72,  45],
       [ 76,  69,  58,  11,   0,   1,   1,   0,   1,   0,   1,   0,   0,
          0,   0,   0,   6,  57,  70,  59],
       [ 68,  64,  59,  16,   0,   1,   0,   1,   0,   1,   1,   1,   0,
          1,   0,   1,   8,  58,  64,  65],
       [ 62,  64,  61,  20,   1,   0,   0,   0,   1,   1,   1,   1,   1,
          0,   1,   0,  14,  61,  60,  69],
       [ 38,  59,  62,  28,   1,   0,   0,   0,   0,   0,   0,   1,   0,
          0,   1,   2,  25,  64,  61,  68],
       [ 61,  73,  77,  43,   0,   1,   0,   1,   0,   0,   0,   0,   0,
          1,   0,   6,  38,  72,  79,  72]])

@pytest.fixture
def cam_from_resolution():
    return np.array([[50, 66, 56, 49, 67, 82, 84, 77, 67, 84, 89, 90, 85, 83, 84, 57,
        62, 92, 86, 68],
       [59, 71, 48, 59, 66, 66, 65, 56, 46, 44, 46, 57, 60, 70, 86, 75,
        53, 80, 64, 54],
       [56, 63, 43, 67, 64, 42, 16,  3,  0,  1,  0,  1,  3, 11, 37, 57,
        55, 73, 68, 61],
       [55, 55, 41, 48, 20,  0,  1,  1,  0,  0,  0,  1,  1,  1,  0, 15,
        50, 70, 65, 52],
       [72, 53, 33,  8,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  1,  1,
        28, 73, 69, 62],
       [91, 64, 17,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  1,
         4, 63, 74, 66],
       [84, 51,  4,  1,  0,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
         1, 38, 71, 55],
       [77, 39,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
         0, 13, 59, 59],
       [73, 35,  0,  0,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
         0,  2, 44, 56],
       [74, 36,  1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
         0,  0, 21, 39],
       [72, 40,  1,  0,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
         0,  0,  4, 23],
       [60, 50,  3,  0,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
         0,  0, 12, 37],
       [53, 62, 14,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
         1,  2, 43, 65],
       [55, 57, 26,  1,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
         1, 17, 59, 59],
       [57, 60, 42,  2,  1,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0,  0,
         3, 52, 79, 57],
       [58, 65, 56, 21,  1,  0,  0,  0,  1,  1,  1,  1,  1,  0,  1,  0,
        33, 74, 72, 41],
       [55, 61, 53, 55, 19,  1,  0,  0,  1,  1,  0,  0,  0,  0,  2, 28,
        64, 68, 61, 41],
       [71, 63, 43, 67, 48, 33, 21,  4,  0,  0,  1,  5, 12, 36, 53, 64,
        59, 56, 75, 41],
       [69, 55, 41, 75, 71, 69, 72, 61, 46, 44, 50, 61, 71, 77, 66, 51,
        43, 55, 84, 56],
       [58, 37, 33, 54, 52, 45, 60, 80, 85, 57, 53, 62, 58, 51, 57, 34,
        26, 32, 62, 52]])

@pytest.fixture
def deformed_images():
    return np.array([[43, 62, 70, 75, 77, 73, 67, 59, 52, 69, 72, 73, 77, 81, 82, 69,
        69, 84, 80, 65],
       [47, 64, 67, 64, 63, 64, 66, 65, 52, 62, 66, 66, 69, 76, 78, 66,
        59, 75, 66, 59],
       [61, 69, 68, 67, 66, 68, 68, 56, 35, 31, 35, 45, 55, 70, 74, 64,
        59, 70, 65, 60],
       [55, 52, 61, 66, 69, 61, 39, 15,  1,  0,  0,  2, 10, 33, 48, 51,
        60, 66, 65, 60],
       [45, 41, 50, 59, 57, 23,  1,  1,  0,  1,  0,  0,  0,  0,  6, 26,
        49, 62, 66, 64],
       [59, 54, 49, 54, 24,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  6,
        40, 61, 66, 67],
       [72, 57, 49, 38,  2,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        30, 59, 64, 64],
       [77, 59, 44, 18,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        19, 57, 63, 58],
       [77, 64, 46, 10,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
        15, 59, 66, 51],
       [80, 69, 52,  7,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
        13, 61, 69, 47],
       [82, 71, 58,  8,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
        18, 64, 68, 43],
       [74, 74, 65, 14,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        24, 66, 65, 48],
       [58, 75, 73, 25,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        30, 66, 63, 48],
       [50, 67, 75, 40,  2,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  2,
        40, 67, 63, 47],
       [49, 64, 66, 52, 16,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 18,
        53, 70, 63, 46],
       [48, 64, 63, 60, 35,  1,  0,  0,  1,  1,  1,  1,  1,  0, 11, 43,
        62, 71, 61, 42],
       [53, 63, 54, 65, 46, 17,  3,  1,  2,  2,  2,  4,  9, 25, 49, 57,
        63, 65, 61, 38],
       [69, 65, 44, 62, 52, 42, 39, 37, 39, 47, 50, 51, 54, 64, 66, 60,
        57, 57, 65, 42],
       [70, 61, 46, 66, 72, 67, 72, 73, 70, 74, 73, 71, 72, 71, 65, 54,
        55, 58, 70, 52],
       [61, 47, 37, 48, 54, 51, 61, 75, 74, 55, 52, 59, 57, 51, 60, 46,
        43, 44, 59, 45]])

@pytest.fixture
def samples_four():
    return np.array([[45, 58, 67, 68, 61, 65, 69, 74, 57, 69, 76, 76, 73, 68, 66, 65,
        68, 72, 78, 77],
       [50, 56, 59, 65, 64, 58, 60, 69, 51, 63, 67, 66, 68, 68, 60, 60,
        61, 64, 61, 73],
       [60, 67, 67, 65, 65, 59, 61, 57, 36, 45, 50, 52, 57, 63, 60, 59,
        61, 64, 63, 71],
       [57, 56, 53, 59, 61, 49, 20,  6,  1,  1,  2,  3,  8, 26, 56, 65,
        60, 63, 65, 58],
       [56, 49, 44, 58, 55, 22,  0,  1,  0,  1,  0,  0,  0,  0, 17, 56,
        58, 65, 62, 54],
       [71, 68, 52, 53, 21,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1, 15,
        48, 68, 62, 54],
       [69, 60, 59, 41,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        31, 60, 59, 49],
       [65, 59, 60, 31,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        21, 59, 60, 44],
       [65, 60, 55, 13,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
        13, 61, 64, 41],
       [67, 60, 50,  4,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
         8, 61, 70, 50],
       [64, 61, 47,  1,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
         8, 64, 72, 59],
       [62, 66, 46,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        11, 66, 68, 66],
       [54, 68, 49,  3,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        17, 65, 64, 70],
       [58, 67, 60, 20,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        30, 68, 66, 71],
       [67, 67, 70, 50,  6,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 13,
        42, 66, 67, 65],
       [67, 58, 63, 61, 39,  3,  0,  0,  1,  1,  1,  1,  1,  0, 11, 39,
        48, 62, 63, 54],
       [66, 51, 54, 58, 55, 44, 21,  5,  3,  2,  3,  6,  9, 22, 47, 50,
        44, 50, 52, 43],
       [71, 49, 54, 60, 60, 59, 54, 46, 44, 45, 44, 41, 36, 55, 61, 52,
        43, 52, 73, 72],
       [59, 49, 55, 59, 63, 62, 63, 65, 68, 70, 67, 69, 59, 67, 61, 65,
        56, 48, 70, 69],
       [57, 50, 54, 57, 62, 61, 66, 66, 68, 54, 55, 70, 68, 65, 58, 63,
        59, 57, 69, 50]])

@pytest.fixture
def samples_twelve():
    return np.array([[58, 72, 71, 65, 55, 64, 64, 65, 66, 66, 65, 71, 77, 74, 64, 60,
        60, 68, 53, 58],
       [63, 62, 66, 64, 61, 64, 69, 67, 56, 57, 60, 61, 58, 63, 69, 67,
        58, 66, 48, 63],
       [64, 72, 71, 67, 69, 68, 64, 51, 32, 30, 40, 51, 53, 60, 65, 64,
        53, 67, 56, 60],
       [64, 70, 65, 64, 68, 59, 35,  5,  0,  0,  0,  3, 14, 32, 53, 61,
        50, 64, 59, 52],
       [57, 59, 54, 56, 55, 26,  1,  1,  0,  1,  0,  0,  0,  0,  8, 44,
        58, 61, 59, 53],
       [57, 58, 55, 52, 23,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1, 13,
        52, 61, 61, 54],
       [62, 55, 61, 35,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        32, 61, 64, 54],
       [65, 59, 50, 10,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        16, 60, 67, 57],
       [67, 65, 39,  1,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
         5, 52, 65, 55],
       [63, 61, 34,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
         2, 45, 63, 58],
       [62, 63, 37,  0,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
         1, 44, 61, 59],
       [63, 60, 46,  1,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
         4, 49, 59, 60],
       [60, 59, 55, 10,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        20, 55, 61, 65],
       [59, 58, 56, 30,  0,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  4,
        38, 54, 62, 64],
       [64, 64, 61, 54, 17,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 22,
        51, 59, 67, 63],
       [61, 63, 63, 66, 55, 17,  0,  0,  1,  1,  1,  1,  1,  1, 19, 43,
        60, 63, 68, 62],
       [57, 61, 61, 59, 53, 44, 27, 15, 12, 11,  9,  6, 16, 37, 48, 52,
        59, 58, 67, 62],
       [57, 60, 61, 61, 57, 54, 53, 48, 44, 47, 47, 43, 47, 51, 52, 59,
        56, 57, 65, 66],
       [59, 60, 55, 58, 56, 57, 62, 65, 63, 69, 61, 54, 53, 53, 61, 60,
        58, 53, 53, 66],
       [54, 57, 49, 52, 53, 55, 56, 62, 64, 65, 59, 60, 65, 63, 56, 52,
        60, 47, 63, 60]])

@pytest.fixture
def bounces_two():
    return np.array([[52, 62, 54, 57, 65, 71, 72, 72, 70, 75, 79, 78, 77, 78, 78, 69,
        70, 82, 87, 71],
       [57, 67, 50, 58, 62, 62, 64, 63, 62, 66, 67, 68, 69, 73, 74, 66,
        63, 72, 73, 64],
       [57, 62, 47, 63, 67, 65, 65, 59, 53, 50, 49, 51, 54, 61, 62, 57,
        60, 71, 72, 66],
       [57, 54, 45, 60, 55, 43, 33, 24, 13,  7,  7,  8, 13, 26, 39, 45,
        57, 70, 71, 65],
       [65, 54, 46, 53, 32,  6,  0,  1,  0,  1,  0,  0,  0,  0,  8, 32,
        52, 69, 72, 66],
       [69, 58, 56, 52, 10,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  7,
        41, 69, 73, 67],
       [69, 58, 63, 47,  2,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        28, 65, 71, 62],
       [69, 58, 60, 37,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        20, 62, 70, 58],
       [71, 60, 57, 29,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
        17, 63, 71, 54],
       [70, 61, 59, 21,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
        15, 64, 73, 51],
       [65, 63, 63, 19,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
        18, 65, 71, 49],
       [56, 67, 70, 22,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        21, 64, 68, 52],
       [53, 68, 73, 29,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        27, 65, 69, 56],
       [56, 62, 69, 42,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        36, 66, 71, 56],
       [59, 62, 64, 52, 14,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 12,
        50, 70, 74, 56],
       [60, 64, 63, 61, 29,  0,  0,  0,  1,  1,  1,  1,  1,  0, 10, 42,
        60, 71, 74, 52],
       [58, 62, 57, 68, 39,  9,  2,  1,  2,  2,  2,  6, 11, 28, 50, 57,
        63, 69, 72, 47],
       [69, 62, 47, 69, 50, 41, 43, 40, 40, 48, 52, 53, 55, 64, 67, 65,
        59, 62, 74, 48],
       [70, 55, 46, 72, 66, 66, 74, 74, 68, 71, 72, 71, 72, 73, 69, 57,
        49, 55, 74, 54],
       [57, 43, 40, 57, 56, 52, 61, 73, 73, 57, 54, 57, 56, 53, 59, 41,
        33, 41, 64, 49]])

@pytest.fixture
def bounces_hundred():
    return np.array([[52, 62, 54, 57, 65, 71, 72, 72, 70, 75, 79, 78, 77, 78, 78, 69,
        70, 82, 87, 71],
       [57, 67, 50, 58, 62, 62, 64, 63, 62, 66, 67, 68, 69, 73, 74, 66,
        63, 72, 73, 64],
       [57, 62, 47, 63, 67, 65, 65, 59, 53, 50, 49, 51, 54, 61, 62, 57,
        60, 71, 72, 66],
       [57, 54, 45, 60, 55, 43, 33, 24, 13,  7,  7,  8, 13, 26, 39, 45,
        57, 70, 71, 65],
       [65, 54, 46, 53, 32,  6,  0,  1,  0,  1,  0,  0,  0,  0,  8, 32,
        52, 69, 72, 66],
       [69, 58, 56, 52, 10,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  7,
        41, 69, 73, 67],
       [69, 58, 63, 47,  2,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        28, 65, 71, 62],
       [69, 58, 60, 37,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        20, 62, 70, 58],
       [71, 60, 57, 29,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
        17, 63, 71, 54],
       [70, 61, 59, 21,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
        15, 64, 73, 51],
       [65, 63, 63, 19,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
        18, 65, 71, 49],
       [56, 67, 70, 22,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        21, 64, 68, 52],
       [53, 68, 73, 29,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        27, 65, 69, 56],
       [56, 62, 69, 42,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        36, 66, 71, 56],
       [59, 62, 64, 52, 14,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 12,
        50, 70, 74, 56],
       [60, 64, 63, 61, 29,  0,  0,  0,  1,  1,  1,  1,  1,  0, 10, 42,
        60, 71, 74, 52],
       [58, 62, 57, 68, 39,  9,  2,  1,  2,  2,  2,  6, 11, 28, 50, 57,
        63, 69, 72, 47],
       [69, 62, 47, 69, 50, 41, 43, 40, 40, 48, 52, 53, 55, 64, 67, 65,
        59, 62, 74, 48],
       [70, 55, 46, 72, 66, 66, 74, 74, 68, 71, 72, 71, 72, 73, 69, 57,
        49, 55, 74, 54],
       [57, 43, 40, 57, 56, 52, 61, 73, 73, 57, 54, 57, 56, 53, 59, 41,
        33, 41, 64, 49]])

@pytest.fixture
def cycles_engine():
    return np.array([[52, 62, 54, 57, 65, 71, 72, 72, 70, 75, 79, 78, 77, 78, 78, 69,
        70, 82, 87, 71],
       [57, 67, 50, 58, 62, 62, 64, 63, 62, 66, 67, 68, 69, 73, 74, 66,
        63, 72, 73, 64],
       [57, 62, 47, 63, 67, 65, 65, 59, 53, 50, 49, 51, 54, 61, 62, 57,
        60, 71, 72, 66],
       [57, 54, 45, 60, 55, 43, 33, 24, 13,  7,  7,  8, 13, 26, 39, 45,
        57, 70, 71, 65],
       [65, 54, 46, 53, 32,  6,  0,  1,  0,  1,  0,  0,  0,  0,  8, 32,
        52, 69, 72, 66],
       [69, 58, 56, 52, 10,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  7,
        41, 69, 73, 67],
       [69, 58, 63, 47,  2,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        28, 65, 71, 62],
       [69, 58, 60, 37,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        20, 62, 70, 58],
       [71, 60, 57, 29,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
        17, 63, 71, 54],
       [70, 61, 59, 21,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
        15, 64, 73, 51],
       [65, 63, 63, 19,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
        18, 65, 71, 49],
       [56, 67, 70, 22,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        21, 64, 68, 52],
       [53, 68, 73, 29,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        27, 65, 69, 56],
       [56, 62, 69, 42,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        36, 66, 71, 56],
       [59, 62, 64, 52, 14,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 12,
        50, 70, 74, 56],
       [60, 64, 63, 61, 29,  0,  0,  0,  1,  1,  1,  1,  1,  0, 10, 42,
        60, 71, 74, 52],
       [58, 62, 57, 68, 39,  9,  2,  1,  2,  2,  2,  6, 11, 28, 50, 57,
        63, 69, 72, 47],
       [69, 62, 47, 69, 50, 41, 43, 40, 40, 48, 52, 53, 55, 64, 67, 65,
        59, 62, 74, 48],
       [70, 55, 46, 72, 66, 66, 74, 74, 68, 71, 72, 71, 72, 73, 69, 57,
        49, 55, 74, 54],
       [57, 43, 40, 57, 56, 52, 61, 73, 73, 57, 54, 57, 56, 53, 59, 41,
        33, 41, 64, 49]])

@pytest.fixture
def eevee_engine():
    return np.array([[85, 74, 53, 43, 53, 49, 43, 60, 57, 32, 39, 72, 63, 49, 38, 33,
        42, 60, 46, 46],
       [46, 44, 63, 40, 43, 60, 53, 74, 70, 19, 31, 53, 62, 44, 24, 20,
        47, 77, 59, 78],
       [75, 63, 77, 59, 34, 66, 60, 53, 61, 26, 43, 40, 53, 60, 55, 40,
        58, 53, 54, 52],
       [53, 66, 75, 29, 34, 56, 57, 23,  1,  0,  0,  1, 27, 51, 63, 49,
        78, 53, 54, 50],
       [42, 51, 63, 64, 37, 23,  1,  1,  0,  1,  0,  0,  0,  0, 11, 40,
        58, 61, 55, 40],
       [37, 42, 58, 40, 13,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1, 30,
        75, 52, 36, 23],
       [35, 53, 74, 56,  1,  0,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
        47, 70, 71, 53],
       [55, 77, 60,  4,  1,  0,  0,  1,  0,  0,  0,  0,  1,  0,  1,  1,
        27, 40, 47, 51],
       [59, 60, 52,  0,  1,  1,  1,  0,  0,  1,  1,  0,  1,  1,  1,  0,
         0, 38, 72, 53],
       [51, 63, 63,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  1,
         0, 28, 65, 69],
       [54, 80, 40,  0,  1,  1,  0,  1,  1,  0,  0,  0,  0,  1,  1,  1,
         0,  7, 24, 40],
       [54, 67, 28,  0,  0,  1,  1,  0,  1,  0,  1,  0,  0,  0,  0,  0,
         0, 23, 47, 75],
       [56, 45, 25, 21,  1,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,
        21, 56, 70, 46],
       [45, 61, 64, 61,  1,  1,  0,  1,  0,  1,  1,  1,  0,  1,  0,  1,
        34, 57, 65, 35],
       [24, 44, 53, 64, 27,  0,  0,  1,  0,  0,  0,  1,  1,  0,  0, 11,
        40, 54, 42, 37],
       [37, 69, 74, 44, 47, 25,  0,  0,  1,  1,  1,  1,  1,  2, 21, 33,
        53, 39, 56, 67],
       [60, 47, 75, 72, 53, 72, 13, 21, 13,  1,  0,  1,  4, 34, 70, 52,
        49, 66, 50, 69],
       [33, 56, 72, 62, 55, 68, 37, 71, 71, 34, 11, 36, 57, 54, 63, 37,
        73, 62, 57, 80],
       [48, 55, 72, 27, 31, 30, 27, 40, 64, 64, 40, 64, 81, 77, 62, 33,
        55, 57, 32, 48],
       [43, 58, 74, 28, 38, 51, 52, 46, 51, 65, 49, 58, 74, 49, 34, 59,
        38, 51, 46, 78]])




