# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================
from pathlib import Path
import numpy as np
import numpy.testing as npt
import pytest
import yaml
from scipy.spatial.transform import Rotation

import pyvale.blender as blender
import pyvale.dataset as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens

GOLD_DIR = Path(__file__).parent / "3D_gold"


@pytest.fixture
def sample_scene_no_cam():
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
    light_data = blender.LightData(
        type=blender.LightType.POINT,
        pos_world=(0, 0, 400),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        energy=1,
    )
    light = scene.add_light(light_data)
    cam_data_0 = sens.CameraData(
        pixels_num=np.array([20, 20]),
        pixels_size=np.array([0.00345, 0.00345]),
        pos_world=np.array([0, 0, 500]),
        rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
        roi_cent_world=(0, 0, 0),
        focal_length=15,
    )
    material_data = blender.MaterialData()
    speckle_path = dataset.dic_pattern_5mpx_path()
    mm_px_resolution = sens.CameraTools.calculate_mm_px_resolution(cam_data_0)
    scene.add_speckle(
        part=part,
        speckle_path=speckle_path,
        mat_data=material_data,
        mm_px_resolution=mm_px_resolution,
    )
    return cam_data_0, part, render_mesh, scene


@pytest.fixture
def sample_stereo_scene(sample_scene_no_cam):
    cam_data_0, part, render_mesh, scene = sample_scene_no_cam
    stereo_system = sens.CameraTools.faceon_stereo_cameras(
        cam_data_0=cam_data_0, stereo_angle=15.0
    )
    cam0, cam1 = scene.add_stereo_system(stereo_system)
    return (stereo_system, part, render_mesh, scene)


# ------------------------------------------------------------------------------
# Gold Fixtures loading from 3D_gold directory
# ------------------------------------------------------------------------------
@pytest.fixture
def stereo_symmetric():
    return np.load(GOLD_DIR / "stereo_symmetric.npy")


@pytest.fixture
def stereo_faceon():
    return np.load(GOLD_DIR / "stereo_faceon.npy")


@pytest.fixture
def deformed_images():
    return np.load(GOLD_DIR / "deformed_images.npy")


@pytest.fixture
def calib_dict():
    yaml_path = GOLD_DIR / "calib_dict.yaml"
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------
# Regression test suites
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "placement, output",
    [
        pytest.param(
            "symmetric", "stereo_symmetric", id="Symmetric convenience function"
        ),
        pytest.param(
            "faceon", "stereo_faceon", id="Face-on convenience function"
        ),
    ],
)
def test_stereo_convenience_cameras(
    placement, output, request, sample_scene_no_cam, tmp_path
):
    (cam_data_0, _, _, scene) = sample_scene_no_cam
    if placement == "symmetric":
        stereo_system = sens.CameraTools.symmetric_stereo_cameras(
            cam_data_0=cam_data_0, stereo_angle=15.0
        )
    elif placement == "faceon":
        stereo_system = sens.CameraTools.faceon_stereo_cameras(
            cam_data_0=cam_data_0, stereo_angle=15.0
        )
    cam0, cam1 = scene.add_stereo_system(stereo_system)
    render_data = blender.RenderData(
        cam_data=(stereo_system.cam_data_0, stereo_system.cam_data_1),
        base_dir=tmp_path,
    )
    image_array = scene.render_single_image(
        stage_image=True, render_data=render_data
    )
    output = request.getfixturevalue(output)

    npt.assert_allclose(image_array, output, atol=2, rtol=0)


def test_stereo_deformation(sample_stereo_scene, deformed_images, tmp_path):
    (stereo_system, part, render_mesh, scene) = sample_stereo_scene
    render_data = blender.RenderData(
        cam_data=(stereo_system.cam_data_0, stereo_system.cam_data_1),
        base_dir=tmp_path,
    )
    image_arrays = scene.render_deformed_images(
        render_mesh=render_mesh,
        sim_spat_dim=3,
        render_data=render_data,
        part=part,
        stage_image=True,
    )
    image_array = image_arrays[:, :, 120:]
    npt.assert_allclose(image_array, deformed_images, atol=2, rtol=0)


def test_cal_images():
    calibration_data = blender.CalibrationData(
        angle_lims=(-10, 10),
        angle_step=5,
        plunge_lims=(-5, 5),
        plunge_step=5,
    )
    number_cal_images = blender.Tools.number_calibration_images(
        calibration_data
    )

    assert number_cal_images == 675


def test_calib_file(tmp_path, sample_stereo_scene, calib_dict):
    (stereo_system, _, _, _) = sample_stereo_scene

    stereo_system.save_calibration(base_dir=tmp_path)

    output = tmp_path / "calibration/calibration.yaml"
    output_dict = yaml.safe_load(output.read_text())

    assert calib_dict == output_dict


def test_cameras_from_calib(tmp_path, sample_stereo_scene):
    (stereo_system, _, _, _) = sample_stereo_scene
    stereo_system.save_calibration(base_dir=tmp_path)

    output = tmp_path / "calibration/calibration.yaml"

    params = yaml.safe_load(output.read_text())

    camerastereo = sens.CameraStereo.from_calibration(
        calib_path=output,
        pos_world_0=np.array([0, 0, 500]),
        rot_world_0=Rotation.from_euler("xyz", [0, 0, 0]),
        focal_length=15.0,
    )

    npt.assert_array_almost_equal(
        camerastereo.stereo_dist, stereo_system.stereo_dist
    )
    assert camerastereo.stereo_rotation.approx_equal(
        stereo_system.stereo_rotation
    )
