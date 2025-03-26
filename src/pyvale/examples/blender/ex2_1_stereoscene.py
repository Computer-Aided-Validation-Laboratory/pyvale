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
    data_path = pyvale.DataSet.thermomechanical_2d_output_path()
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

    # Add the stereo camera system
    cam_data_0 = pyvale.CameraData(pixels_num=np.array([1540, 1040]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([0, 0, 400]),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15.0)
    # Set this to "symmetric" to get a symmetric stereo system or set this to
    # "faceon" to get a face-on stereo system
    stereo_system = "faceon"
    if stereo_system == "symmetric":
        cam_data_1 = pyvale.blender_symmetric_stereo(cam_data_0=cam_data_0,
                                                 stereo_angle=15.0)
    if stereo_system == "faceon":
        cam_data_1 = pyvale.blender_faceon_stereo(cam_data_0=cam_data_0,
                                                 stereo_angle=15.0)



    # Generate calibration file
    stereo_data = pyvale.CameraStereoData(cam_data_0, cam_data_1)
    calib_filepath = Path.cwd() / 'src/pyvale/data/blender'
    pyvale.BlenderTools.generate_calib_file(stereo_data, calib_filepath)


    # Add the light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 200),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=400 * 10**3)
    light = pyvale.BlenderScene.add_light(light_data)

    # Apply the speckle pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    # NOTE: If you wish to use a bigger camera, you will need to generate a
    # bigger speckle pattern generator
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    cam_data=cam_data_0)

    # Rendering image
    # --------------------------------------------------------------------------
    # Set this to True to render image of the current scene
    render_image = True
    if render_image is True:
        save_dir = Path.cwd() / 'src/pyvale/data/blender/blender_images'
        save_name = 'ex_2_1'
        render_data = pyvale.RenderData(cam_data=(cam_data_0, cam_data_1),
                                        save_dir=save_dir,
                                        save_name=save_name,
                                        samples=1)

        pyvale.BlenderScene.render_single_image(save=True, render_data=render_data)

    # Save Blender file
    # --------------------------------------------------------------------------
    blender_path = Path.cwd() / 'src/pyvale/data/blender_files/ex2_1.blend'
    pyvale.BlenderTools.save_blender_file(blender_path, override=True)

if __name__ == "__main__":
    main()