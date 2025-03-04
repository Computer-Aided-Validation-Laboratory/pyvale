"""Example to render images of an object either deforming or performing rigid
body motion using stereo DIC
"""

import os
from pathlib import Path
import numpy as np
import time
import mooseherder as mh
import bpy
from dev_sceneblender import BlenderScene
from dev_partblender import *
from dev_camerablender import CameraData
from dev_lightingblender import LightData, LightType
from dev_objectmaterial import MaterialData
from dev_stereo import StereoData
from dev_render import RenderData, Render
from dev_deform_part import DeformMesh, DeformPart

def main() -> None:
    simcase = 18
    if simcase in [13, 16, 17]:
        data_path = Path('src/pyvale/data/case' + str(simcase) + '_out.e')
    else:
        data_path = Path('src/pyvale/simcases/case' + str(simcase) + '_out.e')
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
    calib_filepath = Path.cwd() / 'dev/lsdev/rendered_images/Stereo DIC/cal'
    stereo_data = StereoData(cam_data_0=cam_data_0,
                             cam_data_1=cam_data_1,
                             base = 35.0,
                             angle_deg=7.0,
                             calib_file=True,
                             calib_filepath=calib_filepath)
    scene.add_stereo_system(stereo_data, scene)

    type = LightType.POINT
    light_position = (0, 0, 200)
    energy = 600 * (10)**3
    light_data = LightData(type=type,
                           position=light_position,
                           energy=energy,
                           part_dimension=part.dimensions)

    light = scene.add_light(light_data)

    mat_data = MaterialData()
    image_path = '/home/lorna/pyvale/dev/lsdev/speckle_3000.bmp'
    mat = scene.add_material(mat_data, part, image_path, cam_data_0)


    #---------------------------------------------------------------------------
    # Set up rendering
    render_start_time = time.perf_counter()
    image_path = Path.cwd() / 'dev/lsdev/rendered_images/Stereo DIC/cal/'
    output_path = Path.cwd() / 'dev/lsdev/rendered_images/Stereo DIC/cal/'
    render_data = RenderData(samples=4)


    # --------------------------------------------------------------------------
    # Deform or RBM
    deform = False
    rbm = False

    #---------------------------------------------------------------------------
    # Deform mesh

    # Reference image
    if deform is True:
        cam_count = 0
        cam_data = [cam_data_0, cam_data_1]
        render_counter = 0
        render_name = 'ref_image'
        for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
            bpy.context.scene.camera = cam
            cam_data_render = cam_data[cam_count]
            render = Render(render_data, image_path=image_path, output_path=output_path, cam_data=cam_data_render)

            render.render_image(render_name, render_counter, part, cam_count)
            cam_count += 1


    # Deformation images
        timesteps = sim_data.time.shape[0]
        meshdeformer = DeformMesh(pv_surf, spat_dim, components)
        nodes = pv_surf.points
        for timestep in range(1, timesteps):
            deformed_nodes = meshdeformer.add_displacement(timestep, nodes)

            if deformed_nodes is not None:
                partdeformer = DeformPart(part, deformed_nodes)
                part = partdeformer.deform_part()
                partdeformer.set_new_frame()
                print(f"{timestep=}")
                print(f"{part.dimensions=}")

                cam_count = 0
                cam_data = [cam_data_0, cam_data_1]
                render_name = 'case18'
                for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
                    bpy.context.scene.camera = cam
                    cam_data_render = cam_data[cam_count]
                    render = Render(render_data, image_path=image_path, output_path=output_path, cam_data=cam_data_render)
                    render.render_image(render_name, timestep, part, cam_count)
                    cam_count += 1

    # --------------------------------------------------------------------------
    # Rigid Body Motion
    if rbm is True:
        render_name = 'rbm_x'
        step = 0.1
        x_max = 1
        x_lims = [0, x_max]
        n_steps = int((x_lims[1] - x_lims[0]) / step)

        for x in range(n_steps):
            x_location = (x * step) + x_lims[0]
            part.location[0] = x_location
            cam_count = 0
            cam_data = [cam_data_0, cam_data_1]
            for cam in [obj for obj in bpy.data.objects if obj.type == 'CAMERA']:
                    bpy.context.scene.camera = cam
                    cam_data_render = cam_data[cam_count]
                    render = Render(render_data, image_path=image_path, output_path=output_path, cam_data=cam_data_render)
                    render.render_image(render_name, x, part, cam_count)
                    cam_count += 1


    render_end_time = time.perf_counter()
    time_render = render_end_time - render_start_time
    print('Time taken to render images: ' + str(time_render) + 's')
    report = open((output_path / 'output.txt'), 'a', encoding='utf-8')
    report.write('\nTime taken to render images: ' + str(time_render) + 's')
    report.close()

    # Save Blender file
    # --------------------------------------------------------------------------
    scene.save_model(filepath)

if __name__ == "__main__":
    main()