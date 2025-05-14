# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

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
    sim_data = pyvale.scale_length_units(sim_data,disp_comps,1000.0)

    render_mesh = pyvale.create_render_mesh(sim_data,
                                        ("disp_y","disp_x"),
                                        sim_spat_dim=3,
                                        field_disp_keys=disp_comps)

    # Set the save path
    # --------------------------------------------------------------------------
    # All the files saved will be saved to a subfolder within this specified
    # base directory.
    # This base directory can be specified by:
    base_dir = Path.cwd()
    # If no base directory is specified, it will be set as your home directory

    # Creating the scene
    # --------------------------------------------------------------------------
    scene = pyvale.BlenderScene()

    # It should be noted that the mesh will be centred to allow for the cameras
    # to be centred on the mesh.
    part = scene.add_part(render_mesh, sim_spat_dim=3)
    # Set the part location
    part_location = np.array([0, 0, 0])
    pyvale.BlenderTools.move_blender_obj(part=part, pos_world=part_location)
    # Set part rotation
    part_rotation = Rotation.from_euler("xyz", [0, 0, 0])
    pyvale.BlenderTools.rotate_blender_obj(part=part, rot_world=part_rotation)

    # Add the camera
    cam_data = pyvale.CameraData(pixels_num=np.array([1540, 1040]),
                                 pixels_size=np.array([0.00345, 0.00345]),
                                 pos_world=(0, 0, 400),
                                 rot_world=Rotation.from_euler("xyz", [0, 0, 0]),
                                 roi_cent_world=(0, 0, 0),
                                 focal_length=15.0)
    camera = scene.add_camera(cam_data)

    # The camera can be moved and rotated this:
    camera.location = (0, 0, 410)
    camera.rotation_euler = (0, 0, 0) # NOTE: The default is an XYZ Euler angle

    # Add the light
    light_data = pyvale.BlenderLightData(type=pyvale.BlenderLightType.POINT,
                                         pos_world=(0, 0, 400),
                                         rot_world=Rotation.from_euler("xyz",
                                                                       [0, 0, 0]),
                                         energy=1)
    light = scene.add_light(light_data)

    # The light can also be moved and rotated:
    light.location = (0, 0, 410)
    light.rotation_euler = (0, 0, 0)

    # Apply the speckle pattern
    material_data = pyvale.BlenderMaterialData()
    speckle_path = pyvale.DataSet.dic_pattern_5mpx_path()
    # NOTE: If you wish to use a bigger camera, you will need to generate a
    # bigger speckle pattern generator
    mm_px_resolution = pyvale.CameraTools.calculate_mm_px_resolution(cam_data)
    scene.add_speckle(part=part,
                      speckle_path=speckle_path,
                      mat_data=material_data,
                      mm_px_resolution=mm_px_resolution)

    # Deform and render images
    # --------------------------------------------------------------------------
    # Set this to True to render image of the deforming part
    render_opts = True
    if render_opts:
        # NOTE: If no save directory is specified, this is where the images will
        # be saved
        render_data = pyvale.RenderData(cam_data=cam_data,
                                        base_dir=base_dir,
                                        threads=8)
        # NOTE: The number of threads used to render the images is set within
        # RenderData, it is defaulted to 4 threads

        scene.render_deformed_images(render_mesh,
                                     sim_spat_dim=3,
                                     render_data=render_data,
                                     part=part,
                                     stage_image=False)
        # NOTE: If bounce_image is set to True, the image will be saved to disk,
        # converted to an array, deleted and the image array will be returned.

        print()
        print(80*"-")
        print("Save directory of the image:", render_data.base_dir)
        print(80*"-")
        print()

    # Save Blender file
    # --------------------------------------------------------------------------
    # The file that will be saved is a Blender project file. This can be opened
    # with the Blender GUI to view the scene.
    pyvale.BlenderTools.save_blender_file(base_dir)

if __name__ == "__main__":
    main()