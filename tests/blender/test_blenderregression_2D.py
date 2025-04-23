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
    image_array = pyvale.BlenderScene.render_single_image(bounce_image=True,
                                                          render_data=render_data)

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
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def three_watt_lighting():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1],
       [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1],
       [2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1],
       [2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1],
       [2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 0],
       [2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1],
       [2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1],
       [1, 1, 2, 2, 2, 2, 3, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 0],
       [1, 2, 2, 2, 3, 3, 3, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 1, 1],
       [2, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1],
       [2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1],
       [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1],
       [1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def vertical_cam():
    return np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
       [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
       [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
       [1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
       [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
       [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
       [1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
       [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
       [1, 1, 0, 0, 0, 0, 1, 1, 0, 1],
       [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
       [0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
       [1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
       [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 0, 1, 0, 0, 0, 0, 1],
       [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1]])

@pytest.fixture
def horizontal_cam():
    return np.array([[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def cam_from_resolution():
    return np.array([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 1, 1, 1, 1],
       [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
       [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
       [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
       [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
       [2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]])

@pytest.fixture
def deformed_images():
    return np.array([[[0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]],

       [[0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1]],

       [[1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0]],

       [[1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1]],

       [[1, 1],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1]],

       [[1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0]],

       [[1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 1]],

       [[1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0]],

       [[1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1]],

       [[1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [1, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0]],

       [[1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1]],

       [[1, 1],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 0],
        [1, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 0],
        [1, 1]],

       [[1, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0]],

       [[0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0]],

       [[1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1]]])

@pytest.fixture
def samples_four():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def samples_twelve():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def bounces_two():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
       [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def bounces_hundred():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
       [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]])

@pytest.fixture
def cycles_engine():
    return np.array([[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1],
       [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
       [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],
       [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
       [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
       [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
       [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
       [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
       [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
       [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
       [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1]],)

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




