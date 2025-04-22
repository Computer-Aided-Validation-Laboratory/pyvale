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
from pathlib import Path
import pyvale
import mooseherder as mh

@pytest.fixture
def sample_scene():
    data_path = Path.cwd() / 'tests/blender/test_out.e'
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)
    render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

    pyvale.BlenderScene.reset_scene()
    part = pyvale.BlenderScene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = pyvale.CameraData(pixels_num=np.array([20, 20]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 700),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=10)
    camera = pyvale.BlenderScene.add_camera(cam_data)
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = pyvale.BlenderScene.add_light(light_data)
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    return render_mesh, part, cam_data

@pytest.fixture
def sample_scene_no_light():
    data_path = Path.cwd() / 'tests/blender/test_out.e'
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)
    render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

    pyvale.BlenderScene.reset_scene()
    part = pyvale.BlenderScene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = pyvale.CameraData(pixels_num=np.array([20, 20]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 700),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=10)
    camera = pyvale.BlenderScene.add_camera(cam_data)
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    return cam_data

@pytest.fixture
def sample_scene_no_cam():
    data_path = Path.cwd() / 'tests/blender/test_out.e'
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    disp_comps = ("disp_x","disp_y")
    sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)
    render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

    pyvale.BlenderScene.reset_scene()
    part = pyvale.BlenderScene.add_part(render_mesh, sim_spat_dim=3)
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = pyvale.BlenderScene.add_light(light_data)

    return part

@pytest.mark.parametrize(
    "energy, output",
    [
        pytest.param(0.5, "half_watt_lighting", id="Normal lighting - 0.5W"),
        pytest.param(3, "three_watt_lighting", id="Normal lighting - 3W")
    ]
)
def test_lighting_energy(energy, output, sample_scene_no_light, request, tmp_path):
    cam_data = sample_scene_no_light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=energy)
    light = pyvale.BlenderScene.add_light(light_data)
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test')
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_array_equal(image_array, output)

@pytest.mark.parametrize(
    "pixels_num, output",
    [
        pytest.param(np.array([10, 20]), "vertical_cam", id="Vertical camera orientation"),
        pytest.param(np.array([20, 10]), "horizontal_cam", id="Horizontal camera orientation")
    ]
)
def test_camera_shape(pixels_num, output, request, sample_scene_no_cam, tmp_path):
    part = sample_scene_no_cam
    cam_data = pyvale.CameraData(pixels_num=pixels_num,
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 700),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=10)
    camera = pyvale.BlenderScene.add_camera(cam_data)
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test')
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_array_equal(image_array, output)

def test_camera_from_resolution(sample_scene_no_cam, cam_from_resolution, tmp_path):
    part = sample_scene_no_cam
    pixels_num = np.array([20, 20])
    pixels_size = np.array([0.00345, 0.00345])
    working_dist = 700
    resolution = 0.1
    cam_data = pyvale.CameraTools.blender_camera_from_resolution(pixels_num,
                                                     pixels_size,
                                                     working_dist,
                                                     resolution)
    cam = pyvale.BlenderScene.add_camera(cam_data)
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test')
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True, render_data=render_data)

    npt.assert_array_equal(image_array, cam_from_resolution)

def test_deformation(sample_scene, deformed_images, tmp_path):
    (render_mesh, part, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir = tmp_path,
                                    save_name='test')
    image_arrays = pyvale.BlenderScene.render_deformed_images(render_mesh,
                                                              sim_spat_dim=3,
                                                              render_data=render_data,
                                                              part=part,
                                                              bounce_image=True)

    npt.assert_array_equal(image_arrays, deformed_images)

@pytest.mark.parametrize(
    "samples, output",
    [
        pytest.param(4, "samples_four", id="Normal sample number - 4"),
        pytest.param(12, "samples_twelve", id="Normal sample number - 12")
    ],
)
def test_samples_happy(samples, output, request, sample_scene, tmp_path):
    (_, _, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test',
                                    samples=samples)
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_array_equal(image_array, output)

def test_samples_unhappy(sample_scene, tmp_path):
    samples = 2.5
    (_, _, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test',
                                    samples=samples)
    with pytest.raises(TypeError):
        image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                              render_data=render_data)


@pytest.mark.parametrize(
    "bounces, output",
    [
        pytest.param(2, "bounces_two", id="Normal bounces number - 2"),
        pytest.param(100, "bounces_hundred", id="Normal bounces number -100")
    ]
)
def test_max_bounces_happy(bounces, output, request, sample_scene, tmp_path):
    (_, _, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test',
                                    max_bounces=bounces)
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_array_equal(image_array, output)

def test_max_bounces_unhappy(sample_scene, tmp_path):
    bounces = 2.5
    (_, _, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test',
                                    max_bounces=bounces)
    with pytest.raises(TypeError):
        image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                              render_data=render_data)

@pytest.mark.parametrize(
        "engine, output",
        [
            pytest.param(pyvale.RenderEngine.CYCLES, "cycles_engine", id="Cycles render engine"),
            pytest.param(pyvale.RenderEngine.EEVEE, "eevee_engine", id="Eevee render engine")
        ]
)
def test_render_engine(engine, output, request, sample_scene, tmp_path):
    (_, _, cam_data) = sample_scene
    render_data = pyvale.RenderData(cam_data=cam_data,
                                    save_dir=tmp_path,
                                    save_name='test',
                                    engine=engine)
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)
    output = request.getfixturevalue(output)

    npt.assert_array_equal(image_array, output)

@pytest.fixture
def half_watt_lighting():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [ 1,  0,  0,  1,  0,  0,  1,  0,  3, 11, 14, 20, 18, 16, 16, 12,
         9,  7,  5,  2],
       [23, 29, 29, 34, 36, 34, 37, 40, 41, 39, 39, 38, 39, 37, 35, 33,
        33, 35, 34, 27],
       [40, 35, 34, 36, 36, 36, 40, 43, 38, 37, 39, 41, 42, 43, 44, 39,
        40, 44, 45, 37],
       [44, 41, 40, 41, 41, 38, 44, 42, 37, 37, 39, 40, 36, 45, 46, 39,
        41, 48, 47, 42],
       [44, 41, 44, 42, 37, 38, 47, 39, 38, 39, 43, 36, 35, 44, 45, 36,
        42, 46, 46, 41],
       [42, 38, 41, 43, 42, 42, 48, 39, 38, 42, 41, 33, 39, 44, 44, 35,
        41, 43, 44, 39],
       [42, 39, 40, 40, 39, 38, 46, 38, 41, 46, 41, 37, 44, 45, 41, 33,
        40, 43, 46, 34],
       [41, 39, 40, 39, 37, 38, 45, 42, 43, 45, 33, 39, 44, 44, 39, 34,
        39, 42, 44, 31],
       [41, 39, 41, 38, 39, 43, 47, 41, 41, 36, 33, 40, 44, 45, 39, 34,
        39, 42, 45, 30],
       [37, 41, 47, 45, 45, 49, 52, 39, 39, 36, 37, 39, 42, 46, 43, 37,
        39, 41, 43, 32],
       [36, 44, 48, 45, 46, 50, 51, 37, 38, 41, 41, 40, 38, 41, 42, 38,
        37, 41, 43, 34],
       [35, 37, 39, 35, 32, 37, 43, 36, 36, 40, 41, 41, 37, 40, 40, 35,
        36, 39, 41, 31],
       [36, 38, 40, 37, 33, 41, 41, 32, 31, 37, 40, 41, 38, 40, 38, 35,
        40, 43, 43, 30],
       [36, 38, 40, 38, 35, 39, 41, 35, 35, 43, 42, 41, 37, 36, 37, 37,
        40, 43, 41, 24],
       [23, 32, 33, 33, 29, 32, 36, 33, 35, 41, 40, 40, 35, 33, 34, 34,
        35, 36, 31, 12],
       [ 3,  5,  8, 10,  8,  9, 12, 12, 14, 19, 21, 19, 14, 13, 14, 14,
        11, 10,  6,  1],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def three_watt_lighting():
    return np.array([[  0,   0,   0,   1,   1,   0,   0,   1,   1,   0,   0,   1,   1,
          0,   0,   0,   1,   1,   1,   1],
       [  0,   1,   0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   1,
          1,   0,   0,   0,   0,   1,   1],
       [  3,   2,   1,   2,   1,   1,   2,   2,  23,  46,  53,  66,  62,
         57,  58,  50,  41,  35,  29,  21],
       [ 74,  90,  87, 100, 105, 101, 105, 112, 115, 112, 112, 108, 110,
        106, 103,  96,  98, 103, 100,  83],
       [112, 103,  99, 104, 103, 103, 113, 118, 110, 106, 110, 115, 117,
        120, 121, 110, 113, 121, 122, 108],
       [121, 116, 112, 114, 115, 109, 123, 117, 107, 106, 111, 112, 106,
        123, 125, 110, 115, 128, 128, 117],
       [121, 115, 121, 116, 108, 109, 126, 111, 109, 112, 119, 105, 103,
        121, 124, 105, 117, 125, 126, 116],
       [117, 109, 115, 119, 116, 117, 130, 109, 109, 117, 116,  99, 110,
        121, 120, 102, 116, 120, 123, 111],
       [117, 110, 113, 113, 110, 109, 125, 110, 115, 126, 114, 108, 121,
        124, 115,  99, 114, 119, 125, 102],
       [115, 111, 113, 112, 107, 109, 124, 117, 119, 122,  99, 110, 121,
        121, 111,  98, 111, 116, 121,  93],
       [114, 110, 115, 110, 110, 119, 128, 114, 114, 105,  98, 114, 121,
        122, 112, 100, 111, 118, 124,  91],
       [107, 116, 128, 123, 124, 131, 135, 112, 111, 105, 108, 111, 118,
        125, 119, 107, 112, 116, 121,  96],
       [106, 122, 129, 123, 124, 132, 134, 106, 109, 114, 115, 113, 108,
        116, 117, 108, 108, 115, 120, 101],
       [102, 107, 112, 102,  94, 107, 119, 104, 105, 112, 116, 115, 107,
        113, 114, 102, 104, 111, 114,  93],
       [105, 109, 112, 106,  98, 115, 116,  94,  94, 107, 113, 115, 110,
        114, 110, 102, 114, 119, 121,  91],
       [103, 109, 112, 109, 103, 112, 116, 103, 102, 118, 117, 115, 108,
        104, 105, 106, 112, 119, 115,  77],
       [ 76,  94,  97,  96,  89,  94, 105,  98, 102, 115, 114, 113, 103,
         97, 102, 100, 102, 104,  93,  49],
       [ 18,  30,  40,  42,  39,  43,  49,  49,  55,  65,  71,  65,  53,
         51,  53,  52,  49,  45,  32,   9],
       [  0,   0,   1,   1,   0,   0,   1,   1,   1,   1,   0,   0,   1,
          0,   1,   1,   0,   1,   0,   0],
       [  0,   0,   1,   0,   0,   1,   0,   1,   0,   0,   0,   0,   0,
          1,   0,   1,   0,   0,   1,   1]])

@pytest.fixture
def vertical_cam():
    return np.array([[ 0,  0,  1,  0,  1,  0,  1,  0,  1,  1],
       [ 0,  0,  0,  1,  0,  0,  1,  0,  0,  1],
       [ 1,  0,  0,  1,  0,  0,  1,  1,  0,  0],
       [ 1,  1,  1,  1,  0,  0,  1,  0,  0,  1],
       [ 1,  0,  1,  0,  0,  0,  0,  1,  0,  1],
       [ 1,  1,  1,  1,  1,  1,  1,  2,  1,  1],
       [53, 54, 63, 67, 70, 65, 62, 61, 57, 50],
       [60, 64, 70, 69, 68, 66, 67, 65, 63, 55],
       [66, 68, 73, 72, 69, 68, 69, 68, 67, 60],
       [68, 70, 71, 71, 70, 71, 71, 69, 66, 60],
       [68, 71, 72, 72, 73, 71, 70, 70, 65, 61],
       [62, 69, 69, 70, 71, 70, 69, 67, 63, 59],
       [47, 59, 65, 64, 65, 65, 65, 63, 61, 54],
       [28, 43, 50, 50, 52, 53, 54, 53, 51, 37],
       [ 2,  8, 15, 19, 23, 25, 23, 18, 14,  5],
       [ 1,  1,  1,  0,  1,  1,  1,  1,  1,  1],
       [ 0,  1,  1,  0,  1,  0,  0,  0,  0,  1],
       [ 1,  1,  1,  0,  0,  0,  0,  1,  0,  1],
       [ 0,  1,  0,  1,  0,  0,  1,  1,  0,  0],
       [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  1]])

@pytest.fixture
def horizontal_cam():
    return np.array([[71, 71, 66, 59, 59, 59, 60, 56, 56, 54, 52, 54, 58, 58, 58, 59,
        62, 67, 71, 56],
       [62, 61, 59, 57, 57, 57, 59, 59, 58, 57, 57, 58, 60, 63, 63, 60,
        62, 64, 64, 57],
       [61, 61, 60, 63, 60, 61, 61, 61, 61, 61, 61, 61, 62, 64, 64, 61,
        65, 68, 68, 54],
       [65, 62, 63, 64, 61, 62, 62, 63, 63, 63, 62, 62, 64, 63, 62, 62,
        66, 67, 69, 52],
       [67, 64, 64, 64, 63, 64, 63, 64, 65, 64, 63, 64, 65, 64, 63, 63,
        67, 68, 69, 54],
       [67, 65, 66, 64, 63, 65, 65, 66, 67, 66, 64, 64, 63, 64, 64, 62,
        65, 68, 68, 60],
       [67, 65, 67, 66, 65, 67, 67, 69, 68, 68, 66, 66, 65, 66, 65, 64,
        65, 66, 65, 63],
       [64, 66, 66, 64, 65, 66, 68, 67, 67, 69, 68, 67, 66, 64, 64, 63,
        64, 65, 61, 65],
       [55, 59, 65, 63, 64, 64, 64, 66, 67, 69, 69, 69, 66, 67, 65, 65,
        64, 66, 60, 62],
       [66, 76, 72, 66, 64, 62, 60, 58, 59, 59, 61, 62, 63, 67, 66, 65,
        64, 64, 66, 59]])

@pytest.fixture
def cam_from_resolution():
    return np.array([[53, 58, 57, 59, 61, 63, 66, 67, 67, 68, 69, 68, 68, 68, 71, 69,
        70, 71, 74, 63],
       [55, 59, 56, 56, 58, 61, 63, 62, 63, 64, 64, 65, 66, 67, 67, 66,
        65, 65, 66, 63],
       [58, 60, 59, 61, 61, 63, 65, 64, 64, 66, 64, 66, 65, 66, 66, 65,
        65, 67, 66, 64],
       [62, 60, 60, 61, 61, 62, 64, 65, 64, 64, 64, 64, 64, 65, 65, 64,
        64, 65, 66, 64],
       [62, 60, 60, 62, 62, 63, 64, 65, 64, 64, 64, 64, 64, 65, 66, 65,
        65, 67, 68, 64],
       [61, 60, 61, 62, 62, 62, 63, 64, 63, 62, 63, 63, 63, 65, 66, 64,
        65, 67, 69, 66],
       [62, 61, 62, 63, 61, 62, 64, 64, 64, 63, 64, 62, 62, 64, 65, 63,
        65, 66, 67, 63],
       [60, 60, 62, 64, 63, 63, 65, 65, 63, 64, 63, 61, 63, 65, 65, 63,
        64, 65, 67, 61],
       [63, 61, 62, 63, 63, 63, 65, 65, 65, 66, 64, 63, 65, 66, 66, 62,
        64, 66, 69, 58],
       [63, 63, 64, 63, 63, 64, 65, 66, 65, 66, 62, 64, 66, 66, 65, 64,
        65, 66, 69, 57],
       [63, 63, 64, 63, 64, 66, 66, 66, 65, 64, 61, 63, 65, 67, 65, 63,
        65, 66, 69, 56],
       [62, 64, 66, 65, 65, 67, 68, 64, 64, 63, 63, 64, 64, 66, 66, 63,
        64, 65, 67, 57],
       [63, 64, 65, 65, 66, 68, 69, 63, 63, 64, 65, 65, 63, 64, 65, 64,
        64, 65, 67, 60],
       [60, 61, 62, 62, 61, 64, 65, 62, 61, 64, 65, 66, 63, 64, 64, 62,
        63, 65, 67, 58],
       [60, 61, 62, 61, 60, 62, 64, 61, 60, 64, 65, 65, 63, 63, 62, 61,
        62, 65, 68, 56],
       [61, 61, 62, 61, 60, 62, 64, 61, 61, 67, 66, 65, 63, 61, 62, 61,
        62, 64, 68, 53],
       [61, 61, 61, 62, 60, 61, 63, 62, 62, 66, 64, 65, 63, 60, 60, 60,
        61, 63, 68, 49],
       [63, 60, 60, 62, 61, 61, 62, 62, 62, 64, 63, 64, 61, 62, 62, 60,
        59, 61, 69, 50],
       [68, 58, 59, 61, 62, 62, 64, 65, 65, 65, 63, 63, 64, 63, 63, 59,
        58, 60, 64, 50],
       [63, 60, 56, 58, 59, 60, 62, 66, 64, 61, 59, 59, 59, 61, 59, 57,
        53, 54, 60, 44]])

@pytest.fixture
def deformed_images():
    return np.array([[[ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 1,  1],
        [ 1,  1]],

       [[ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1]],

       [[ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 8,  8],
        [22, 22],
        [25, 25],
        [33, 33],
        [30, 30],
        [28, 28],
        [28, 28],
        [23, 23],
        [18, 18],
        [15, 15],
        [11, 11],
        [ 6,  6]],

       [[37, 37],
        [47, 47],
        [46, 46],
        [53, 53],
        [56, 56],
        [54, 54],
        [57, 57],
        [61, 61],
        [63, 63],
        [61, 61],
        [61, 61],
        [59, 59],
        [60, 60],
        [57, 57],
        [55, 55],
        [51, 51],
        [52, 52],
        [55, 55],
        [53, 53],
        [43, 43]],

       [[61, 61],
        [55, 55],
        [53, 53],
        [56, 56],
        [55, 55],
        [56, 55],
        [61, 61],
        [65, 65],
        [59, 59],
        [57, 57],
        [60, 60],
        [63, 63],
        [64, 64],
        [66, 66],
        [67, 67],
        [60, 60],
        [62, 62],
        [67, 67],
        [68, 68],
        [58, 58]],

       [[67, 67],
        [63, 63],
        [61, 61],
        [63, 63],
        [63, 63],
        [59, 59],
        [68, 68],
        [64, 64],
        [57, 57],
        [57, 57],
        [61, 61],
        [61, 61],
        [57, 57],
        [68, 68],
        [70, 70],
        [60, 60],
        [63, 63],
        [72, 72],
        [72, 72],
        [64, 64]],

       [[67, 67],
        [63, 63],
        [67, 67],
        [64, 64],
        [58, 58],
        [59, 59],
        [71, 71],
        [60, 60],
        [59, 59],
        [61, 61],
        [66, 66],
        [56, 56],
        [55, 55],
        [67, 67],
        [69, 69],
        [56, 56],
        [64, 64],
        [70, 70],
        [70, 70],
        [63, 63]],

       [[64, 64],
        [59, 59],
        [63, 63],
        [66, 66],
        [64, 64],
        [64, 64],
        [73, 73],
        [60, 60],
        [59, 59],
        [64, 64],
        [63, 63],
        [52, 52],
        [60, 60],
        [67, 67],
        [67, 67],
        [54, 54],
        [64, 64],
        [66, 66],
        [68, 68],
        [60, 60]],

       [[65, 65],
        [60, 60],
        [62, 62],
        [62, 62],
        [60, 60],
        [59, 59],
        [70, 70],
        [59, 59],
        [63, 63],
        [70, 71],
        [63, 63],
        [58, 58],
        [67, 67],
        [69, 69],
        [63, 63],
        [52, 52],
        [62, 62],
        [66, 66],
        [70, 70],
        [54, 54]],

       [[63, 63],
        [60, 60],
        [62, 62],
        [61, 61],
        [58, 58],
        [59, 59],
        [70, 70],
        [65, 65],
        [66, 66],
        [68, 68],
        [52, 52],
        [60, 60],
        [67, 67],
        [67, 67],
        [60, 60],
        [53, 52],
        [60, 60],
        [64, 64],
        [67, 67],
        [49, 49]],

       [[63, 63],
        [60, 60],
        [63, 63],
        [59, 59],
        [60, 60],
        [65, 65],
        [72, 72],
        [63, 63],
        [63, 63],
        [56, 56],
        [52, 52],
        [62, 62],
        [67, 67],
        [68, 68],
        [61, 61],
        [53, 53],
        [60, 61],
        [65, 65],
        [69, 69],
        [48, 48]],

       [[58, 58],
        [64, 64],
        [72, 72],
        [69, 69],
        [69, 69],
        [74, 74],
        [78, 78],
        [61, 61],
        [61, 61],
        [56, 56],
        [58, 58],
        [61, 61],
        [65, 65],
        [70, 70],
        [66, 66],
        [58, 58],
        [61, 61],
        [64, 64],
        [67, 67],
        [51, 51]],

       [[57, 57],
        [67, 67],
        [73, 73],
        [68, 68],
        [69, 69],
        [76, 76],
        [77, 77],
        [57, 57],
        [59, 59],
        [62, 63],
        [63, 63],
        [62, 62],
        [59, 59],
        [64, 64],
        [65, 65],
        [59, 59],
        [58, 58],
        [63, 63],
        [66, 66],
        [54, 54]],

       [[54, 54],
        [57, 57],
        [61, 61],
        [54, 54],
        [50, 50],
        [58, 58],
        [65, 65],
        [56, 56],
        [56, 57],
        [61, 61],
        [63, 63],
        [63, 63],
        [57, 58],
        [62, 62],
        [62, 62],
        [54, 54],
        [56, 56],
        [61, 61],
        [63, 63],
        [49, 50]],

       [[56, 56],
        [59, 59],
        [61, 61],
        [57, 57],
        [52, 52],
        [63, 63],
        [63, 63],
        [50, 50],
        [50, 50],
        [58, 58],
        [62, 62],
        [63, 63],
        [60, 60],
        [62, 62],
        [59, 60],
        [54, 54],
        [62, 62],
        [66, 66],
        [67, 67],
        [48, 49]],

       [[55, 55],
        [59, 59],
        [61, 61],
        [59, 59],
        [55, 55],
        [61, 61],
        [64, 64],
        [55, 55],
        [55, 55],
        [65, 65],
        [64, 64],
        [63, 63],
        [58, 58],
        [56, 56],
        [57, 57],
        [57, 57],
        [61, 61],
        [65, 65],
        [63, 63],
        [39, 39]],

       [[39, 39],
        [50, 50],
        [51, 51],
        [51, 51],
        [47, 47],
        [50, 50],
        [56, 56],
        [52, 52],
        [55, 55],
        [63, 63],
        [62, 62],
        [62, 62],
        [55, 55],
        [51, 51],
        [54, 54],
        [54, 54],
        [54, 54],
        [56, 56],
        [49, 49],
        [23, 23]],

       [[ 6,  6],
        [11, 11],
        [17, 17],
        [19, 19],
        [17, 17],
        [19, 19],
        [23, 23],
        [23, 23],
        [26, 26],
        [32, 33],
        [35, 36],
        [33, 33],
        [25, 25],
        [24, 24],
        [25, 25],
        [25, 25],
        [22, 22],
        [20, 20],
        [13, 13],
        [ 2,  2]],

       [[ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0]],

       [[ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 1,  1],
        [ 0,  0],
        [ 0,  0],
        [ 1,  1],
        [ 1,  1]]])

@pytest.fixture
def samples_four():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [23, 21, 20, 21, 19, 21, 23, 25, 27, 26, 22, 22, 20, 21, 22, 23,
        24, 24, 19, 14],
       [45, 47, 49, 53, 55, 57, 58, 57, 56, 55, 54, 53, 52, 55, 56, 57,
        55, 56, 55, 44],
       [56, 55, 56, 59, 60, 61, 60, 61, 59, 58, 57, 58, 60, 63, 64, 63,
        63, 64, 61, 51],
       [63, 63, 59, 60, 59, 58, 59, 61, 57, 57, 58, 60, 63, 66, 65, 64,
        64, 66, 64, 55],
       [63, 60, 60, 60, 57, 59, 60, 60, 57, 58, 57, 58, 59, 61, 62, 62,
        64, 64, 62, 55],
       [61, 60, 62, 62, 59, 59, 59, 61, 57, 56, 57, 57, 58, 57, 61, 58,
        62, 62, 61, 55],
       [61, 59, 60, 60, 60, 59, 59, 60, 59, 58, 57, 58, 59, 59, 57, 56,
        62, 63, 63, 54],
       [60, 58, 60, 59, 59, 59, 58, 60, 60, 58, 55, 57, 59, 60, 57, 57,
        61, 64, 66, 58],
       [60, 59, 60, 59, 60, 60, 59, 60, 60, 58, 56, 57, 58, 61, 57, 57,
        61, 66, 69, 63],
       [58, 59, 60, 59, 59, 60, 59, 59, 59, 59, 60, 60, 59, 60, 61, 60,
        63, 68, 68, 66],
       [58, 61, 59, 59, 60, 61, 61, 61, 60, 61, 62, 62, 60, 58, 60, 62,
        64, 67, 66, 67],
       [60, 60, 60, 59, 59, 60, 59, 61, 60, 62, 60, 60, 58, 58, 59, 60,
        61, 65, 64, 66],
       [63, 62, 64, 62, 61, 61, 60, 61, 60, 61, 58, 59, 59, 60, 60, 58,
        59, 62, 62, 61],
       [65, 61, 62, 60, 58, 60, 62, 62, 61, 61, 58, 57, 59, 59, 57, 54,
        56, 58, 58, 51],
       [58, 56, 57, 55, 55, 59, 59, 61, 60, 59, 55, 53, 52, 51, 48, 45,
        45, 43, 42, 32],
       [31, 29, 32, 34, 35, 33, 32, 30, 28, 26, 23, 22, 17, 16, 16, 12,
         9,  9, 13,  6],
       [ 3,  1,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def samples_twelve():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [25, 28, 28, 28, 26, 24, 23, 22, 20, 21, 19, 18, 17, 15, 12, 11,
        13, 18, 21, 17],
       [55, 57, 57, 58, 60, 59, 58, 58, 56, 55, 54, 54, 54, 53, 50, 51,
        52, 55, 57, 48],
       [58, 59, 57, 57, 58, 59, 58, 59, 58, 59, 59, 61, 62, 57, 57, 57,
        58, 60, 60, 54],
       [60, 59, 58, 60, 59, 59, 60, 61, 61, 60, 59, 56, 55, 57, 60, 59,
        60, 61, 62, 55],
       [61, 57, 61, 59, 58, 58, 61, 61, 62, 60, 58, 56, 55, 58, 60, 60,
        62, 62, 63, 56],
       [61, 58, 60, 60, 59, 59, 57, 59, 60, 61, 58, 58, 60, 58, 60, 61,
        63, 65, 67, 60],
       [61, 59, 58, 60, 60, 59, 56, 61, 60, 61, 58, 59, 60, 59, 59, 60,
        63, 65, 65, 58],
       [60, 59, 60, 58, 58, 58, 56, 60, 59, 60, 59, 57, 57, 54, 56, 59,
        58, 58, 63, 61],
       [59, 59, 61, 58, 59, 59, 57, 59, 57, 56, 56, 55, 56, 56, 57, 59,
        60, 59, 62, 63],
       [59, 57, 60, 59, 58, 60, 59, 57, 58, 57, 59, 55, 56, 59, 60, 60,
        60, 60, 59, 59],
       [58, 58, 60, 59, 59, 60, 58, 57, 60, 59, 60, 56, 55, 57, 59, 61,
        60, 63, 63, 64],
       [57, 57, 58, 59, 58, 58, 58, 60, 60, 59, 59, 56, 54, 57, 56, 58,
        57, 56, 61, 62],
       [58, 59, 61, 61, 62, 59, 59, 58, 59, 59, 56, 55, 57, 60, 59, 56,
        59, 60, 63, 61],
       [59, 60, 64, 65, 64, 62, 61, 60, 59, 58, 57, 55, 53, 54, 58, 54,
        60, 64, 65, 62],
       [51, 57, 57, 55, 53, 52, 52, 54, 55, 55, 55, 56, 56, 58, 56, 51,
        53, 54, 55, 48],
       [14, 16, 16, 15, 13, 13, 13, 16, 18, 19, 20, 21, 20, 20, 20, 18,
        15, 13, 13, 15],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def bounces_two():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [ 1,  1,  0,  1,  0,  0,  1,  1,  8, 22, 25, 33, 30, 28, 28, 23,
        18, 15, 11,  6],
       [37, 47, 46, 53, 56, 54, 57, 61, 63, 61, 61, 59, 60, 57, 55, 51,
        52, 55, 53, 43],
       [61, 55, 53, 56, 55, 56, 61, 65, 59, 57, 60, 63, 64, 66, 67, 60,
        62, 67, 68, 58],
       [67, 63, 61, 63, 63, 59, 68, 64, 57, 57, 61, 61, 57, 68, 70, 60,
        63, 72, 72, 64],
       [67, 63, 67, 64, 58, 59, 71, 60, 59, 61, 66, 56, 55, 67, 69, 56,
        64, 70, 70, 63],
       [64, 59, 63, 66, 64, 64, 73, 60, 59, 64, 63, 52, 60, 67, 67, 54,
        64, 66, 68, 60],
       [65, 60, 62, 62, 60, 59, 70, 59, 63, 70, 63, 58, 67, 69, 63, 52,
        62, 66, 70, 54],
       [63, 60, 62, 61, 58, 59, 70, 65, 66, 68, 52, 60, 67, 67, 60, 53,
        60, 64, 67, 49],
       [63, 60, 63, 59, 60, 65, 72, 63, 63, 56, 52, 62, 67, 68, 61, 53,
        60, 65, 69, 48],
       [58, 64, 72, 69, 69, 74, 78, 61, 61, 56, 58, 61, 65, 70, 66, 58,
        61, 64, 67, 51],
       [57, 67, 73, 68, 69, 76, 77, 57, 59, 62, 63, 62, 59, 64, 65, 59,
        58, 63, 66, 54],
       [54, 57, 61, 54, 50, 58, 65, 56, 56, 61, 63, 63, 57, 62, 62, 54,
        56, 61, 63, 49],
       [56, 59, 61, 57, 52, 63, 63, 50, 50, 58, 62, 63, 60, 62, 59, 54,
        62, 66, 67, 48],
       [55, 59, 61, 59, 55, 61, 64, 55, 55, 65, 64, 63, 58, 56, 57, 57,
        61, 65, 63, 39],
       [39, 50, 51, 51, 47, 50, 56, 52, 55, 63, 62, 62, 55, 51, 54, 54,
        54, 56, 49, 23],
       [ 6, 11, 17, 19, 17, 19, 23, 23, 26, 32, 35, 33, 25, 24, 25, 25,
        22, 20, 13,  2],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def bounces_hundred():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [ 1,  1,  0,  1,  0,  0,  1,  1,  8, 22, 25, 33, 30, 28, 28, 23,
        18, 15, 11,  6],
       [37, 47, 46, 53, 56, 54, 57, 61, 63, 61, 61, 59, 60, 57, 55, 51,
        52, 55, 53, 43],
       [61, 55, 53, 56, 55, 56, 61, 65, 59, 57, 60, 63, 64, 66, 67, 60,
        62, 67, 68, 58],
       [67, 63, 61, 63, 63, 59, 68, 64, 57, 57, 61, 61, 57, 68, 70, 60,
        63, 72, 72, 64],
       [67, 63, 67, 64, 58, 59, 71, 60, 59, 61, 66, 56, 55, 67, 69, 56,
        64, 70, 70, 63],
       [64, 59, 63, 66, 64, 64, 73, 60, 59, 64, 63, 52, 60, 67, 67, 54,
        64, 66, 68, 60],
       [65, 60, 62, 62, 60, 59, 70, 59, 63, 70, 63, 58, 67, 69, 63, 52,
        62, 66, 70, 54],
       [63, 60, 62, 61, 58, 59, 70, 65, 66, 68, 52, 60, 67, 67, 60, 53,
        60, 64, 67, 49],
       [63, 60, 63, 59, 60, 65, 72, 63, 63, 56, 52, 62, 67, 68, 61, 53,
        60, 65, 69, 48],
       [58, 64, 72, 69, 69, 74, 78, 61, 61, 56, 58, 61, 65, 70, 66, 58,
        61, 64, 67, 51],
       [57, 67, 73, 68, 69, 76, 77, 57, 59, 62, 63, 62, 59, 64, 65, 59,
        58, 63, 66, 54],
       [54, 57, 61, 54, 50, 58, 65, 56, 56, 61, 63, 63, 57, 62, 62, 54,
        56, 61, 63, 49],
       [56, 59, 61, 57, 52, 63, 63, 50, 50, 58, 62, 63, 60, 62, 59, 54,
        62, 66, 67, 48],
       [55, 59, 61, 59, 55, 61, 64, 55, 55, 65, 64, 63, 58, 56, 57, 57,
        61, 65, 63, 39],
       [39, 50, 51, 51, 47, 50, 56, 52, 55, 63, 62, 62, 55, 51, 54, 54,
        54, 56, 49, 23],
       [ 6, 11, 17, 19, 17, 19, 23, 23, 26, 32, 35, 33, 25, 24, 25, 25,
        22, 20, 13,  2],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def cycles_engine():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [ 1,  1,  0,  1,  0,  0,  1,  1,  8, 22, 25, 33, 30, 28, 28, 23,
        18, 15, 11,  6],
       [37, 47, 46, 53, 56, 54, 57, 61, 63, 61, 61, 59, 60, 57, 55, 51,
        52, 55, 53, 43],
       [61, 55, 53, 56, 55, 56, 61, 65, 59, 57, 60, 63, 64, 66, 67, 60,
        62, 67, 68, 58],
       [67, 63, 61, 63, 63, 59, 68, 64, 57, 57, 61, 61, 57, 68, 70, 60,
        63, 72, 72, 64],
       [67, 63, 67, 64, 58, 59, 71, 60, 59, 61, 66, 56, 55, 67, 69, 56,
        64, 70, 70, 63],
       [64, 59, 63, 66, 64, 64, 73, 60, 59, 64, 63, 52, 60, 67, 67, 54,
        64, 66, 68, 60],
       [65, 60, 62, 62, 60, 59, 70, 59, 63, 70, 63, 58, 67, 69, 63, 52,
        62, 66, 70, 54],
       [63, 60, 62, 61, 58, 59, 70, 65, 66, 68, 52, 60, 67, 67, 60, 53,
        60, 64, 67, 49],
       [63, 60, 63, 59, 60, 65, 72, 63, 63, 56, 52, 62, 67, 68, 61, 53,
        60, 65, 69, 48],
       [58, 64, 72, 69, 69, 74, 78, 61, 61, 56, 58, 61, 65, 70, 66, 58,
        61, 64, 67, 51],
       [57, 67, 73, 68, 69, 76, 77, 57, 59, 62, 63, 62, 59, 64, 65, 59,
        58, 63, 66, 54],
       [54, 57, 61, 54, 50, 58, 65, 56, 56, 61, 63, 63, 57, 62, 62, 54,
        56, 61, 63, 49],
       [56, 59, 61, 57, 52, 63, 63, 50, 50, 58, 62, 63, 60, 62, 59, 54,
        62, 66, 67, 48],
       [55, 59, 61, 59, 55, 61, 64, 55, 55, 65, 64, 63, 58, 56, 57, 57,
        61, 65, 63, 39],
       [39, 50, 51, 51, 47, 50, 56, 52, 55, 63, 62, 62, 55, 51, 54, 54,
        54, 56, 49, 23],
       [ 6, 11, 17, 19, 17, 19, 23, 23, 26, 32, 35, 33, 25, 24, 25, 25,
        22, 20, 13,  2],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])

@pytest.fixture
def eevee_engine():
    return np.array([[ 0,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  1,  1,  0,  0,  0,
         1,  1,  1,  1],
       [ 0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,
         0,  0,  1,  1],
       [42, 40, 29,  8,  8, 30, 42, 40, 40, 30, 25, 44, 43, 44, 34, 28,
        38, 17, 24, 21],
       [52, 65, 73, 28, 35, 63, 68, 50, 59, 64, 45, 76, 61, 68, 68, 49,
        78, 53, 54, 49],
       [42, 51, 63, 63, 58, 79, 69, 49, 72, 71, 35, 65, 52, 52, 38, 48,
        58, 61, 55, 40],
       [37, 42, 58, 50, 61, 51, 35, 50, 73, 66, 49, 68, 74, 75, 58, 65,
        80, 52, 35, 22],
       [35, 53, 74, 72, 79, 62, 61, 56, 81, 49, 41, 51, 58, 48, 62, 47,
        80, 70, 71, 53],
       [55, 77, 64, 47, 51, 44, 42, 62, 70, 45, 62, 66, 51, 43, 49, 69,
        56, 48, 47, 51],
       [59, 60, 54, 36, 63, 69, 51, 56, 37, 67, 78, 65, 34, 44, 76, 66,
        58, 52, 71, 53],
       [51, 62, 71, 60, 66, 63, 74, 55, 35, 55, 86, 64, 44, 24, 60, 85,
        74, 51, 64, 68],
       [54, 79, 55, 68, 60, 54, 60, 85, 60, 69, 55, 44, 34, 48, 59, 81,
        54, 39, 24, 39],
       [54, 67, 54, 79, 56, 65, 45, 73, 54, 56, 41, 52, 55, 47, 54, 47,
        47, 47, 47, 75],
       [56, 45, 48, 86, 73, 45, 38, 52, 58, 60, 25, 70, 88, 82, 61, 45,
        52, 65, 70, 45],
       [45, 60, 64, 78, 58, 43, 40, 55, 60, 70, 41, 75, 54, 61, 43, 72,
        78, 57, 65, 35],
       [24, 44, 53, 71, 54, 51, 60, 71, 59, 48, 53, 76, 59, 48, 55, 71,
        50, 54, 42, 37],
       [37, 69, 74, 44, 53, 49, 54, 52, 77, 37, 52, 49, 46, 76, 72, 70,
        54, 38, 55, 67],
       [59, 43, 71, 69, 51, 71, 40, 51, 77, 49, 48, 42, 47, 75, 72, 52,
        43, 64, 47, 63],
       [16, 12, 26, 28, 29, 26,  6,  8, 29, 26,  2, 18, 17, 21, 23, 14,
        29, 40, 39, 39],
       [ 0,  0,  1,  1,  0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,
         0,  1,  0,  0],
       [ 0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,
         0,  0,  1,  1]])




