"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import bpy
from multiprocessing import cpu_count
import mooseherder as mh
import pyvale
from pyvale.core.cameradata import CameraData
from pyvale.core.blenderlightdata import BlenderLightData
from pyvale.core.blendertools import BlenderTools, BlenderError
from pyvale.core.simtools import SimTools
from pyvale.core.blendermaterialdata import BlenderMaterialData
from pyvale.core.camerastereodata import CameraStereoData
from pyvale.core.blenderrenderdata import RenderData, RenderEngine

# NOTE: This module is a feature under development

class BlenderScene():
    """Interface (abstract base class) for a scene within Blender.

    #TODO: Add to this

    """

    @staticmethod
    def reset_scene():
        bpy.ops.wm.read_factory_settings(use_empty=True)

        bpy.context.scene.unit_settings.scale_length = 0.001
        bpy.context.scene.unit_settings.length_unit = 'MILLIMETERS'

        new_world = bpy.data.worlds.new('World')
        bpy.context.scene.world = new_world
        new_world.use_nodes = True
        node_tree = new_world.node_tree
        nodes = node_tree.nodes

        nodes.clear()
        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.inputs[0].default_value = [0.5, 0.5, 0.5, 1]
        bg_node.inputs[1].default_value = 0

    @staticmethod
    def add_camera(cam_data:CameraData):
        new_cam = bpy.data.cameras.new('Camera')
        camera = bpy.data.objects.new('Camera', new_cam)
        bpy.context.collection.objects.link(camera)

        camera.location = (cam_data.pos_world[0],
                           cam_data.pos_world[1],
                           cam_data.pos_world[2])
        camera.rotation_mode = 'XYZ'
        rotation_euler = cam_data.rot_world.as_euler("xyz", degrees=False)
        camera.rotation_euler = rotation_euler



        camera['sensor_px'] = (2464, 2056)
        camera['px_size'] = (cam_data.pixels_size / 1000)
        camera['k1'] = cam_data.k1
        camera['k2'] = cam_data.k2
        camera['k3'] = cam_data.k3
        camera['p1'] = cam_data.p1
        camera['p2'] = cam_data.p2
        camera['c0'] = cam_data.c0
        camera['c1'] = cam_data.c1

        new_cam.lens = cam_data.focal_length
        new_cam.sensor_width = cam_data.sensor_size[0] / 1000
        new_cam.sensor_height = cam_data.sensor_size[1] / 1000

        if cam_data.object_distance is not None:
            new_cam.dof.focus_distance = cam_data.object_distance
            new_cam.dof.use_dof = True
            new_cam.dof.aperture_fstop = cam_data.fstop

        bpy.context.scene.camera = camera

        return camera # Do I need this return?

    @staticmethod
    def add_stereo_system(cam_data_0: CameraData, cam_data_1: CameraData):
        # Can i use method defined in this namespace?
        cam0 = BlenderScene.add_camera(cam_data_0)
        cam1 = BlenderScene.add_camera(cam_data_1)
        return cam0, cam1

    @staticmethod
    def add_light(light_data: BlenderLightData):
        # TODO: Make method compatible for different light types
        type = light_data.type.value
        name = type.capitalize() + 'Light'
        light = bpy.data.lights.new(name=name, type=type)
        light_ob = bpy.data.objects.new(name=name, object_data=light)

        light_ob.location = (light_data.pos_world[0],
                                   light_data.pos_world[1],
                                   light_data.pos_world[2])

        light_ob.rotation_mode = 'XYZ'
        rotation_euler = light_data.rot_world.as_euler("xyz", degrees=False)
        light_ob.rotation_euler = rotation_euler

        light.energy = light_data.energy
        light.shadow_soft_size = 1.5 # Add to dataclass

        bpy.context.collection.objects.link(light_ob)

        return light_ob

    @staticmethod
    def add_part(sim_data: mh.SimData):
        spat_dim = SimTools.get_mesh_spat_dim(sim_data)
        components = SimTools.get_simulation_components(sim_data)
        sim_data.coords = sim_data.coords * 1000 # Change from m to mm
        sim_data.coords = SimTools.centre_mesh_nodes(sim_data.coords)
        (pv_grid, _) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                      components,
                                                      spat_dim)
        pv_surf = SimTools.conv_pvgrid_to_pvsurf(pv_grid)

        vertices = pv_surf.points
        elements_per_face = SimTools.surf_mesh_elements_per_face(pv_surf)
        faces = pv_surf.faces.reshape(-1, elements_per_face)
        faces = np.delete(faces, 0, axis=1)

        mesh = bpy.data.meshes.new("Part")
        mesh.from_pydata(vertices, [], faces)
        part = bpy.data.objects.new("Part", mesh)
        bpy.context.scene.collection.objects.link(part)

        return part

    @staticmethod
    def add_speckle(part,
                    speckle_path: Path | None,
                    mat_data: BlenderMaterialData | None,
                    cam_data: CameraData):
        # Add way to only take 1 of camera data
        # Work out way to get FOV if camera is not perp
        BlenderTools.clear_material_nodes(part)
        (FOV_x, _) = BlenderTools.calculate_FOV(cam_data)
        if mat_data is None:
            mat_data = BlenderMaterialData()
        # TODO: Add option for if speckle_path is None to generate speckle pattern
        # and add
        BlenderTools.add_image_texture(mat_data=mat_data, image_path=speckle_path)
        BlenderTools.uv_unwrap_part(part, FOV_x)




    @staticmethod
    def deform_all_timesteps(sim_data: mh.SimData, part):
        timesteps = sim_data.time.shape[0]
        spat_dim = SimTools.get_mesh_spat_dim(sim_data)
        components = SimTools.get_simulation_components(sim_data)
        sim_data.coords = sim_data.coords
        sim_data.coords = SimTools.centre_mesh_nodes(sim_data.coords)
        (pv_grid, _) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                 components,
                                                 spat_dim)
        pv_surf = SimTools.conv_pvgrid_to_pvsurf(pv_grid)

        for timestep in range(1, timesteps):
            deformed_nodes = SimTools.get_deformed_nodes(timestep,
                                                         pv_surf,
                                                         spat_dim,
                                                         components)
            if deformed_nodes is not None:
                BlenderTools.deform_single_timestep(part, deformed_nodes)
                BlenderTools.set_new_frame(part)
        # Do I need to return anything from this?
        # Do I need this function? - Maybe for viewing in Blender?


    @staticmethod
    def render_single_image(save: bool, render_data: RenderData):
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = int(cpu_count())
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
            bpy.context.scene.cycles.use_denoising = False # Only turned off to make rendering faster
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        if isinstance(render_data.cam_data, tuple):
            cam_count = 0
            image_count = 0
            for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
                bpy.context.scene.camera = cam
                cam_data_render = render_data.cam_data[cam_count]
                bpy.context.scene.render.resolution_x = cam_data_render.pixels_num[0]
                bpy.context.scene.render.resolution_y = cam_data_render.pixels_num[1]
                filename = render_data.save_name + "_" + str(image_count) + "_" + str(cam_count) + ".tiff"
                bpy.context.scene.render.filepath = str(render_data.save_dir / filename)
                if save is True:
                    bpy.ops.render.render(write_still=True)
                else:
                    bpy.ops.render.render(write_still=False)
                    # Add to this
                cam_count += 1
        else:
            image_count = 0
            bpy.context.scene.render.resolution_x = render_data.cam_data.pixels_num[0]
            bpy.context.scene.render.resolution_y = render_data.cam_data.pixels_num[1]
            filename = render_data.save_name + "_" + str(image_count) + ".tiff"
            bpy.context.scene.render.filepath = str(render_data.save_dir / filename)
            if save is True:
                bpy.ops.render.render(write_still=True)
            else:
                bpy.ops.render.render(write_still=False)
                # Add to this

    @staticmethod
    def render_deformed_images(sim_data: mh.SimData,
                               render_data:RenderData,
                               part,
                               save: bool | None = False):
        timesteps = sim_data.time.shape[0]
        spat_dim = SimTools.get_mesh_spat_dim(sim_data)
        components = SimTools.get_simulation_components(sim_data)
        sim_data.coords = sim_data.coords
        sim_data.coords = SimTools.centre_mesh_nodes(sim_data.coords)
        (pv_grid, _) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                 components,
                                                 spat_dim)
        pv_surf = SimTools.conv_pvgrid_to_pvsurf(pv_grid)

        # Render parameters
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = int(cpu_count())
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
            bpy.context.scene.cycles.use_denoising = False # Only turned off to make rendering faster
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        for timestep in range(0, timesteps):
            deformed_nodes = SimTools.get_deformed_nodes(timestep,
                                                         pv_surf,
                                                         spat_dim,
                                                         components)
            if deformed_nodes is not None:
                BlenderTools.deform_single_timestep(part, deformed_nodes)
                BlenderTools.set_new_frame(part)

                if isinstance(render_data.cam_data, tuple):
                    cam_count = 0
                    for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
                        bpy.context.scene.camera = cam
                        cam_data_render = render_data.cam_data[cam_count]
                        bpy.context.scene.render.resolution_x = cam_data_render.pixels_num[0]
                        bpy.context.scene.render.resolution_y = cam_data_render.pixels_num[1]
                        filename = render_data.save_name + "_" + str(timestep) + "_" + str(cam_count) + ".tiff"
                        bpy.context.scene.render.filepath = str(render_data.save_dir / filename)
                        if save is True:
                            bpy.ops.render.render(write_still=True)
                        else:
                            bpy.ops.render.render(write_still=False)
                            # Add to this
                        cam_count += 1
                else:
                    bpy.context.scene.render.resolution_x = render_data.cam_data.pixels_num[0]
                    bpy.context.scene.render.resolution_y = render_data.cam_data.pixels_num[1]
                    filename = render_data.save_name + "_" + str(timestep) + ".tiff"
                    bpy.context.scene.render.filepath = str(render_data.save_dir / filename)
                    if save is True:
                        bpy.ops.render.render(write_still=True)
                    else:
                        bpy.ops.render.render(write_still=False)
                        # Add to this











