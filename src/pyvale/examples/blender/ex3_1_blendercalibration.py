"""
================================================================================
Example: Simple Blender scene with no deformation

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import pyvale
import mooseherder as mh

def main() -> None:
    data_path = Path('/home/lorna/mooseherder/scripts/moose/moose-mech-simple_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    # Creating the scene
    # --------------------------------------------------------------------------
    pyvale.BlenderScene.reset_scene()

    part = pyvale.BlenderScene.add_part(sim_data)
    # Set the part location
    part_location = np.array([0, 0, 0])
    pyvale.BlenderTools.move_blender_part(part=part, pos_world=part_location)
    # Set part rotation
    part_rotation = Rotation.from_euler("xyz", [0, 0, 0])
    pyvale.BlenderTools.rotate_blender_part(part=part, rot_world=part_rotation)

    # Add the camera
    cam_data_0 = pyvale.CameraData(pixels_num=np.array([2464, 2056]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([0, 0, 300]),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=20)
    # Set this to "symmetric" to get a symmetric stereo system or set this to
    # "faceon" to get a face-on stereo system
    stereo_system = "faceon"
    if stereo_system == "symmetric":
        cam_data_1 = pyvale.blender_symmetric_stereo(cam_data_0=cam_data_0,
                                                 stereo_angle=15.0)
    if stereo_system == "faceon":
        cam_data_1 = pyvale.blender_faceon_stereo(cam_data_0=cam_data_0,
                                                 stereo_angle=15.0)

    # Add the light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 200),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=400 * 10**3)
    light = pyvale.BlenderScene.add_light(light_data)

    # Apply the speckle pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = Path('/home/lorna/pyvale/dev/lsdev/cal_target.tiff')
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    cam_data=cam_data_0,
                                    cal=True)

    # Rendering image
    # --------------------------------------------------------------------------

    save_dir = Path.cwd() / 'src/pyvale/data/blender_images/calibration'
    save_name = 'cal'
    render_data = pyvale.RenderData(cam_data=(cam_data_0, cam_data_1),
                                    save_dir=save_dir,
                                    save_name=save_name,
                                    samples=4)
    calibration_data = pyvale.CalibrationData(angle_lims=(-6, 6),
                                              angle_step=10,
                                              plunge_lims=(-5, 5),
                                              plunge_step=5)

    pyvale.BlenderTools.calibration_images(render_data, calibration_data, part)

    # Save Blender file
    # --------------------------------------------------------------------------
    blender_path = Path.cwd() / 'src/pyvale/data/blender_files/cal.blend'
    pyvale.BlenderTools.save_blender_file(blender_path, override=True)

if __name__ == "__main__":
    main()