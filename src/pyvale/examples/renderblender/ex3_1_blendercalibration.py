# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import pyvale

def main() -> None:
    #NOTE: All lengths are to be specified in mm

    # Set the save path
    # --------------------------------------------------------------------------
    # All the files saved will be saved to a subfolder within this specified
    # base directory.
    # This base directory can be specified by:
    base_dir = Path("./")
    # If no base directory is specified, it will be set as your home directory

    # Creating the scene
    # --------------------------------------------------------------------------
    scene = pyvale.BlenderScene()

    # Add the calibration target
    # A rectangular calibration target of the specified size is added to the scene
    target = scene.add_cal_target(target_size=np.array([150, 100, 10]))

    # Add the camera
    cam_data_0 = pyvale.CameraData(pixels_num=np.array([1540, 1040]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([0, 0, 400]),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15.0)
    # Set this to "symmetric" to get a symmetric stereo system or set this to
    # "faceon" to get a face-on stereo system
    stereo_setup = "faceon"
    if stereo_setup == "symmetric":
        stereo_system = pyvale.CameraTools.symmetric_stereo_cameras(
            cam_data_0=cam_data_0,
            stereo_angle=15.0)
    if stereo_setup == "faceon":
        stereo_system = pyvale.CameraTools.faceon_stereo_cameras(
            cam_data_0=cam_data_0,
            stereo_angle=15.0)

    scene.add_stereo_system(stereo_system)

    # Generate calibration file
    stereo_system.save_calibration(base_dir)

    # Add the light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 200),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = scene.add_light(light_data)
    # The light can be moved and rotated:
    light.location = (0, 0, 210)
    light.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle

    # Apply the calibration target pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = Path.cwd() / "src/pyvale/data/cal_target.tiff"
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data_0)
    scene.add_speckle(part=target,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution,
                                    cal=True)
    # NOTE: The `cal` flag has to be set to True in order to scale the
    # calibration target pattern correctly

    # Rendering calibration images
    # --------------------------------------------------------------------------
    save_dir = Path.cwd() / "blenderimages"
    save_name = "cal"
    render_data = pyvale.RenderData(cam_data=(stereo_system.cam_data_0,
                                              stereo_system.cam_data_1),
                                    base_dir=base_dir)
    # NOTE: The number of threads used to render the images is set within
    # RenderData, it is defaulted to 4 threads

    # The desired limits for the calibration target movement are to be set within
    # the CalibrationData dataclass
    calibration_data = pyvale.CalibrationData(angle_lims=(-10, 10),
                                              angle_step=5,
                                              plunge_lims=(-5, 5),
                                              plunge_step=5)

    # The number of calibration images that will be rendered can be calculated
    number_calibration_images = pyvale.BlenderTools.number_calibration_images(calibration_data)
    print()
    print(80*"-")
    print("Number of calibration images to be rendered:", number_calibration_images)
    print(80*"-")

    # The calibration images can then be rendered
    pyvale.BlenderTools.render_calibration_images(render_data,
                                                  calibration_data,
                                                  target)

    print()
    print(80*"-")
    print("Save directory of the images:", (render_data.base_dir / "calimages"))
    print(80*"-")
    print()

    # Save Blender file
    # --------------------------------------------------------------------------
    # The file that will be saved is a Blender project file. This can be opened
    # with the Blender GUI to view the scene.
    pyvale.BlenderTools.save_blender_file(base_dir)

if __name__ == "__main__":
    main()