"""
================================================================================
Example: Simple stereo Blender scene with deformation

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
    data_path = pyvale.DataSet.render_mechanical_3d_path()
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    disp_comps = ("disp_x","disp_y", "disp_z")

    # Scale m -> mm
    # NOTE: All lengths are to be specified in mm
    sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)

    render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

    # Creating the scene
    # --------------------------------------------------------------------------
    # When Blender is started, default objects are present within the scene
    # The following function is used to clear the scene
    pyvale.BlenderScene.reset_scene()

    part = pyvale.BlenderScene.add_part(render_mesh, sim_spat_dim=3)
    # Set the part location
    part_location = np.array([0, 0, 0])
    pyvale.BlenderTools.move_blender_obj(part=part, pos_world=part_location)
    # Set part rotation
    part_rotation = Rotation.from_euler("xyz", [0, 0, 0])
    pyvale.BlenderTools.rotate_blender_obj(part=part, rot_world=part_rotation)

    # Add the stereo camera system
    cam_data_0 = pyvale.CameraData(pixels_num=np.array([1540, 1040]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=np.array([0, 0, 400]),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15.0)
    # Set this to "symmetric" to get a symmetric stereo system or set this to
    # "faceon" to get a face-on stereo system
    stereo_system = "symmetric"
    if stereo_system == "symmetric":
        stereo_data = pyvale.CameraTools.symmetric_stereo_cameras(
            cam_data_0=cam_data_0,
            stereo_angle=15.0)
    if stereo_system == "faceon":
        stereo_data = pyvale.CameraTools.faceon_stereo_cameras(
            cam_data_0=cam_data_0,
            stereo_angle=15.0)

    cam0, cam1 = pyvale.BlenderScene.add_stereo_system(stereo_data)

    # Generate calibration file
    calib_filepath = Path.cwd() / "blenderimages"
    pyvale.BlenderTools.generate_calib_file(stereo_data, calib_filepath)


    # Add the light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = pyvale.BlenderScene.add_light(light_data)

    # The light can also be moved and rotated:
    light.location = (0, 0, 410)
    light.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle

    # Apply the speckle pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    # NOTE: If you wish to use a bigger camera, you will need to generate a
    # bigger speckle pattern generator

    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data_0)
    pyvale.BlenderScene.add_speckle(part=part,
                                    speckle_path=speckle_path,
                                    mat_data=material_data,
                                    mm_px_resolution=mm_px_resolution)

    # Deform and render images
    # --------------------------------------------------------------------------
    # Set this to True to render image of the deforming part
    render_opts = True
    if render_opts:
        save_dir = Path.cwd() / "blenderimages"
        # NOTE: If no save directory is specified, this is where the images will
        # be saved
        save_name = "ex2_2"
        render_data = pyvale.RenderData(cam_data=(stereo_data.cam_data_0,
                                                  stereo_data.cam_data_1),
                                        save_dir=save_dir,
                                        save_name=save_name,
                                        threads=8)
        # NOTE: The number of threads used to render the images is set within
        # RenderData, it is defaulted to 4 threads

        pyvale.BlenderScene.render_deformed_images(render_mesh=render_mesh,
                                                   sim_spat_dim=3,
                                                   render_data=render_data,
                                                   part=part,
                                                   bounce_image=False)
        # NOTE: If bounce_image is set to True, the image will be saved to disk,
        # converted to an array, deleted and the image array will be returned.

        print()
        print(80*"-")
        print("Save directory of the image:", render_data.save_dir)
        print(80*"-")
        print()

    # Save Blender file
    # --------------------------------------------------------------------------
    # The file that will be saved is a Blender project file. This can be opened
    # with the Blender GUI to view the scene.
    blender_path = Path.cwd() / "blenderfiles"
    pyvale.BlenderTools.save_blender_file(blender_path)

    print()
    print(80*"-")
    print("Save directory of Blender project file:", blender_path)
    print(80*"-")

if __name__ == "__main__":
    main()
