"""Example to render a set of calibration images
"""

import os
from pathlib import Path
import numpy as np
import mooseherder as mh
from dev_sceneblender import BlenderScene
from dev_partblender import *
from dev_camerablender import CameraData
from dev_lightingblender import LightData, LightType
from dev_objectmaterial import MaterialData
from dev_stereo import StereoData
from dev_render import RenderData
from dev_calibration import CalibrationData, Calibration

def main() -> None:
    simcase = 'cal'
    data_path = Path('/home/lorna/mooseherder/scripts/moose/moose-mech-simple_out.e')
    data_reader = mh.ExodusReader(data_path)
    sim_data = data_reader.read_all_sim_data()


    dir = Path.cwd() / 'dev/lsdev/blender_files'
    name = 'case' + str(simcase)
    filename = name + '.blend'
    filepath = dir / filename
    all_files = os.listdir(dir)
    for ff in all_files:
        if filename == ff:
            os.remove(dir / ff)

    filepath = str(filepath)

    # Creating scene
    # --------------------------------------------------------------------------
    scene = BlenderScene()

    part_location = (0, 0, 0)
    angle = np.radians(90)
    part_rotation = (0, 0, 0)

    part, pv_surf, spat_dim, components = scene.add_part(sim_data=sim_data)
    scene.set_part_location(part=part, location=part_location)
    scene.set_part_rotation(part=part, rotation=part_rotation)

    sensor_px = (2464, 2056)
    cam_position = (0, 0, 250)
    focal_length = 15.0
    cam_data_0 = CameraData(sensor_px=sensor_px,
                          position=cam_position,
                          focal_length=focal_length,
                          part_dimension=part.dimensions)
    cam_data_1 = cam_data_0
    calib_filepath = Path.cwd() / 'dev/lsdev/rendered_images/stereo/calibration2'
    stereo_data = StereoData(cam_data_0=cam_data_0,
                             cam_data_1=cam_data_1,
                             base = 35.0,
                             angle_deg=7.0,
                             calib_file=False,
                             calib_filepath=calib_filepath)
    scene.add_stereo_system(stereo_data, scene)

    type = LightType.SPOT
    light_position = (0, 0, 200)
    energy = 700 * (10)**3
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy,
                           part_dimension=part.dimensions)

    light = scene.add_light(light_data)

    cal = True
    mat_data = MaterialData(cal)
    image_path = '/home/lorna/pyvale/dev/lsdev/cal_target.tiff'
    mat = scene.add_material(mat_data, part, image_path, cam_data_0)


    #---------------------------------------------------------------------------
    # Set up rendering and render reference image
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/stereo/calibration2'
    output_path = image_path


    render_data = RenderData(samples=1)

    cal_data = CalibrationData(part=part,
                               image_path=image_path,
                               output_path=output_path,
                               render_data=render_data,
                               cam_data=[cam_data_0, cam_data_1],)
    calibration = Calibration(cal_data)
    calibration.perform_calibration()

    # Save Blender file
    # --------------------------------------------------------------------------
    scene.save_model(filepath)

if __name__ == "__main__":
    main()