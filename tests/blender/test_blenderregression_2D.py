# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial.transform import Rotation

import pyvale.blender as blender
import pyvale.dataset as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens

GOLD_DIR = Path(__file__).parent / "2D_gold"


@pytest.fixture
def sample_scene():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=sens.EDim.TWOD,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=(0, 0, 500),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    camera = scene.add_camera(cam_data)
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return render_mesh, part, cam_data, scene


@pytest.fixture
def sample_scene_no_light():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=sens.EDim.TWOD,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    cam_data = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=(0, 0, 500),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    camera = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return cam_data, scene


@pytest.fixture
def sample_scene_no_cam():
    data_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()
    disp_comps = ("disp_x", "disp_y")
    sim_data = sens.scale_length_units(1000.0, sim_data, disp_comps)
    render_mesh = sens.create_render_mesh(
        sim_data,
        ("disp_y", "disp_x"),
        sim_spat_dim=2,
        field_disp_keys=disp_comps,
    )

    scene = blender.Scene()
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)

    return part, scene


# ------------------------------------------------------------------------------
# Gold Fixtures loading from 2D_gold directory
# ------------------------------------------------------------------------------
@pytest.fixture
def half_watt_lighting():
    return np.load(GOLD_DIR / "half_watt_lighting.npy")


@pytest.fixture
def three_watt_lighting():
    return np.load(GOLD_DIR / "three_watt_lighting.npy")


@pytest.fixture
def vertical_cam():
    return np.load(GOLD_DIR / "vertical_cam.npy")


@pytest.fixture
def horizontal_cam():
    return np.load(GOLD_DIR / "horizontal_cam.npy")


@pytest.fixture
def cam_from_resolution():
    return np.load(GOLD_DIR / "cam_from_resolution.npy")


@pytest.fixture
def samples_four():
    return np.load(GOLD_DIR / "samples_four.npy")


@pytest.fixture
def samples_twelve():
    return np.load(GOLD_DIR / "samples_twelve.npy")


@pytest.fixture
def bounces_two():
    return np.load(GOLD_DIR / "bounces_two.npy")


@pytest.fixture
def bounces_hundred():
    return np.load(GOLD_DIR / "bounces_hundred.npy")


@pytest.fixture
def cycles_engine():
    return np.load(GOLD_DIR / "cycles_engine.npy")


@pytest.fixture
def eevee_engine():
    return np.load(GOLD_DIR / "eevee_engine.npy")


@pytest.fixture
def deformed_images():
    return np.load(GOLD_DIR / "deformed_images.npy")


# ------------------------------------------------------------------------------
# Regression test suites
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "energy, output",
    [
        pytest.param(0.5, "half_watt_lighting", id="Normal lighting - 0.5W"),
        pytest.param(3, "three_watt_lighting", id="Normal lighting - 3W"),
    ],
)
def test_lighting_energy(
    energy, output, sample_scene_no_light, request, tmp_path
):
    cam_data, scene = sample_scene_no_light
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=energy,
    )
    light = scene.add_light(light_data)
    render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


@pytest.mark.parametrize(
    "pixels_num, output",
    [
        pytest.param(
            np.array([10, 20]),
            "vertical_cam",
            id="Vertical camera orientation",
        ),
        pytest.param(
            np.array([20, 10]),
            "horizontal_cam",
            id="Horizontal camera orientation",
        ),
    ],
)
def test_camera_shape(
    pixels_num, output, request, sample_scene_no_cam, tmp_path
):
    part, scene = sample_scene_no_cam
    cam_data = sens.CameraData(
        pixels_num=pixels_num,
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=(0, 0, 500),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    camera = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


def test_camera_from_resolution(
    sample_scene_no_cam, cam_from_resolution, tmp_path
):
    part, scene = sample_scene_no_cam
    pixels_num = np.array([20, 20])
    pixels_size = np.array([0.00345, 0.00345])
    working_dist = 500
    resolution = 0.1
    cam_data = sens.CameraTools.blender_camera_from_resolution(
        pixels_num, pixels_size, working_dist, resolution
    )
    cam = scene.add_camera(cam_data)
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )

    npt.assert_allclose(image_array, cam_from_resolution, atol=2, rtol=0)


def test_deformation(sample_scene, deformed_images, tmp_path):
    (render_mesh, part, cam_data, scene) = sample_scene
    render_data = blender.RenderData(cam_data=cam_data, base_dir=tmp_path)
    image_arrays = scene.render_deformed_images(
        render_mesh,
        sim_spat_dim=3,
        render_data=render_data,
        part=part,
        stage_image=True,
    )

    npt.assert_allclose(image_arrays[:, :, 10], deformed_images, atol=2, rtol=0)


@pytest.mark.parametrize(
    "samples, output",
    [
        pytest.param(4, "samples_four", id="Normal sample number - 4"),
        pytest.param(12, "samples_twelve", id="Normal sample number - 12"),
    ],
)
def test_samples_happy(samples, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(
        cam_data=cam_data, base_dir=tmp_path, samples=samples
    )
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


def test_samples_unhappy(sample_scene, tmp_path):
    samples = 2.5
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(
        cam_data=cam_data, base_dir=tmp_path, samples=samples
    )
    with pytest.raises(TypeError):
        image_array = scene.render_single_image(
            stage_image=True, render_data=render_data
        )


@pytest.mark.parametrize(
    "bounces, output",
    [
        pytest.param(2, "bounces_two", id="Normal bounces number - 2"),
        pytest.param(100, "bounces_hundred", id="Normal bounces number -100"),
    ],
)
def test_max_bounces_happy(bounces, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(
        cam_data=cam_data, base_dir=tmp_path, max_bounces=bounces
    )
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


def test_max_bounces_unhappy(sample_scene, tmp_path):
    bounces = 2.5
    (_, _, cam_data, scene) = sample_scene
    render_data = blender.RenderData(
        cam_data=cam_data, base_dir=tmp_path, max_bounces=bounces
    )
    with pytest.raises(TypeError):
        image_array = scene.render_single_image(
            stage_image=True, render_data=render_data
        )


@pytest.mark.parametrize(
    "engine, output",
    [
        pytest.param(
            blender.RenderEngine.CYCLES,
            "cycles_engine",
            id="Cycles render engine",
        ),
        pytest.param(
            blender.RenderEngine.EEVEE,
            "eevee_engine",
            id="Eevee render engine",
        ),
    ],
)
def test_render_engine(engine, output, request, sample_scene, tmp_path):
    (_, _, cam_data, scene) = sample_scene
    if engine == blender.RenderEngine.EEVEE:
        gpu_present = blender.Tools.check_for_GPU()
        if gpu_present is False:
            pytest.skip("Unsupported hardware for EEVEE")

    render_data = blender.RenderData(
        cam_data=cam_data, base_dir=tmp_path, engine=engine
    )
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)
