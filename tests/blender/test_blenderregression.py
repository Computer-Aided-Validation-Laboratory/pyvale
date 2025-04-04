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
    data_path = pyvale.DataSet.thermomechanical_2d_output_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    pyvale.BlenderScene.reset_scene()
    part = pyvale.BlenderScene.add_part(sim_data)
    cam_data = pyvale.CameraData(pixels_num=np.array([400, 250]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 400),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15)
    camera = pyvale.BlenderScene.add_camera(cam_data)
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = pyvale.BlenderScene.add_light(light_data)
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    cam_data=cam_data)
    return cam_data

@pytest.mark.parameterize(
    "samples, output",
    [
        pytest.param(4, j, id="Normal sample number"),
        pytest.param(2., j, id='Non-integer sample number'),
        pytest.param(1025, j, id='Sample number too large')
    ],
)
def test_samples(samples, output, sample_scene):
    render_data = pyvale.RenderData(cam_data=sample_scene,
                                    save_dir=Path.cwd()/'src/pyvale/tests/blender',
                                    save_name='test',
                                    samples=samples)
    image_array = pyvale.BlenderScene.render_single_image(save=False, render_data=render_data)

    npt.assert_array_equal(image_array, output)

