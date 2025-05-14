# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================
import numpy as np
from pathlib import Path
import bpy
from pyvale.cameradata import CameraData
from pyvale.blenderlightdata import BlenderLightData
from pyvale.blendertools import BlenderTools
from pyvale.simtools import SimTools
from pyvale.blendermaterialdata import BlenderMaterialData
from pyvale.blenderrenderdata import RenderData, RenderEngine
from pyvale.camerastereo import CameraStereo
from pyvale.rendermesh import RenderMeshData
from pyvale.pyvaleexceptions import BlenderError


class BlenderScene():
    """Namespace for creating a scene within Blender.
    Methods include adding an object, camera, light and adding a speckle pattern,
    as well as deforming the object, and then rendering the scene.
    """

    def __init__(self) -> None:
        self.reset_scene()

    def reset_scene(self) -> None:
        """This method creates a new, empty scene.
        The units are then set to milimetres, and all nodes are cleared from the
        scene. This method will be called when the class is initialised.
        """
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

    def add_camera(self, cam_data:CameraData) -> bpy.data.objects:
        """Method to add a camera object within Blender.

        Parameters
        ----------
        cam_data : CameraData
            A dataclass containing the necessary parameters to create the camera
            object in Blender.

        Returns
        -------
        bpy.data.objects
            The Blender camera object that is created.
        """
        new_cam = bpy.data.cameras.new('Camera')
        camera = bpy.data.objects.new('Camera', new_cam)
        bpy.context.collection.objects.link(camera)

        camera.location = (cam_data.pos_world[0],
                           cam_data.pos_world[1],
                           cam_data.pos_world[2])
        camera.rotation_mode = 'XYZ'
        rotation_euler = cam_data.rot_world.as_euler("xyz", degrees=False)
        camera.rotation_euler = rotation_euler

        pixels_num = (int(cam_data.pixels_num[0]), int(cam_data.pixels_num[1]))
        camera['sensor_px'] = pixels_num
        camera['px_size'] = cam_data.pixels_size
        camera['k1'] = cam_data.k1
        camera['k2'] = cam_data.k2
        camera['k3'] = cam_data.k3
        camera['p1'] = cam_data.p1
        camera['p2'] = cam_data.p2
        camera['c0'] = cam_data.c0
        camera['c1'] = cam_data.c1

        new_cam.lens_unit = 'MILLIMETERS'
        new_cam.lens = cam_data.focal_length
        new_cam.sensor_fit = 'HORIZONTAL'
        new_cam.sensor_width = cam_data.sensor_size[0]
        new_cam.sensor_height = cam_data.sensor_size[1]

        if cam_data.fstop is not None:
            new_cam.dof.focus_distance = cam_data.image_dist
            new_cam.dof.use_dof = True
            new_cam.dof.aperture_fstop = cam_data.fstop

        bpy.context.scene.camera = camera
        return camera

    def add_stereo_system(self, stereo_system: CameraStereo) -> tuple[bpy.data.objects,
                                                           bpy.data.objects]:
        # TODO: Correct docstring
        """A method to add a stereo camera system within Blender, given an
        instance of the CameraStereo class (that describes a stereo system).

        Parameters
        ----------
        stereo_system: CameraStereo
            An instance of the CameraStereo class, describing a stereo system.

        Returns
        -------
        tuple[bpy.data.objects, bpy.data.objects]
            A tuple of the Blender camera objects: camera 0 and camera 1.
        """
        cam0 = self.add_camera(stereo_system.cam_data_0)
        cam1 = self.add_camera(stereo_system.cam_data_1)
        return cam0, cam1

    def add_light(self, light_data: BlenderLightData) -> bpy.data.objects:
        """A method to add a light object within Blender.

        Parameters
        ----------
        light_data : BlenderLightData
            A dataclass contain the necessary parameters to create a Blender
            light object.

        Returns
        -------
        bpy.data.objects
            The Blender light object that is created.
        """
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

        light.energy = light_data.energy * 10**6
        light.shadow_soft_size = light_data.shadow_soft_size

        bpy.context.collection.objects.link(light_ob)

        return light_ob

    def add_part(self,
                 render_mesh: RenderMeshData,
                 sim_spat_dim: int) -> bpy.data.objects:
        """A method to add a part mesh into Blender, given a RenderMeshData object.
        This is done by taking the mesh information from the RenderMeshData
        object and converting it into a form that is accepted by Blender. It
        should be noted that the object is placed at the origin and centred
        around its geometric centroid.

        Parameters
        ----------
        render_mesh: RenderMeshData
            A dataclass containing the mesh information of the skinned
            simulation mesh.
        sim_spat_dim: int
            The spatial dimension of the simulation mesh.

        Returns
        -------
        bpy.data.objects
            The Blender part object that is created.
        """
        nodes_centred = SimTools.centre_mesh_nodes(render_mesh.coords,
                                              sim_spat_dim)
        vertices = np.delete(nodes_centred, 3, axis=1)
        faces = render_mesh.connectivity

        mesh = bpy.data.meshes.new("Part")
        mesh.from_pydata(vertices, [], faces)
        part = bpy.data.objects.new("Part", mesh)

        bpy.context.scene.collection.objects.link(part)

        return part

    def add_cal_target(self, target_size: np.ndarray) -> bpy.data.objects:
        """A function to add a calibration target object to a Blender scene.

        Parameters
        ----------
        target_size : np.ndarray
            The dimensions of the calibration target, with the
            shape=(width, height, depth).

        Returns
        -------
        bpy.data.objects
            A Blender part object of the calibration target.
        """
        nodes = [
            (-target_size[0] / 2, target_size[1] / 2, 0),
            (-target_size[0] / 2, -target_size[1] / 2, 0),
            (target_size[0] / 2, -target_size[1] / 2, 0),
            (target_size[0] / 2, target_size[1] / 2, 0),
        ]
        elements = [(0, 1, 2, 3)]
        thickness = target_size[2]
        mesh = bpy.data.meshes.new("part")
        mesh.from_pydata(nodes, [], elements)
        target = bpy.data.objects.new("specimen", mesh)
        bpy.context.scene.collection.objects.link(target)
        target.modifiers.new(name="solidify", type="SOLIDIFY")
        target.modifiers["solidify"].thickness = thickness
        target.location = (0, 0, -target_size[2])
        return target

    def add_speckle(self,
                    part: bpy.data.objects,
                    speckle_path: Path | None,
                    mat_data: BlenderMaterialData | None,
                    mm_px_resolution: float,
                    cal: bool = False) -> None:
        """A method to add a speckle pattern to an existing mesh object within
        Blender. The speckle pattern can either be passed in as an image file
        that is saved to the disc, or can be generated dynamically (this is
        currently not an option but this method has the capaibility to link up
        to a speckle pattern generator)

        Parameters
        ----------
        part : bpy.data.objects
            The Blender part object, to which the speckle is to be applied.
        speckle_path : Path | None
            The filepath containing the speckle pattern image. If this is None,
            there will be capability to generate a speckle pattern.
        mat_data : BlenderMaterialData | None
            A dataclass containin the material parameters. If this is None, it
            is initialised within the method.
        mm_px_resolution: float
            The mm/px resolution of the camera. This is required in order to
            scale the speckle image.
        cal : bool, optional
            A flag that can be set if a calibration target image is added to
            a Blender part object. When set to True, the part object is UV
            unwrapped differently to ensure the correct scaling, by default False
        """
        BlenderTools.clear_material_nodes(part)
        if mat_data is None:
            mat_data = BlenderMaterialData()
        if speckle_path.exists():
            BlenderTools.add_image_texture(mat_data=mat_data, image_path=speckle_path)
        else:
            speckle_pattern = np.array() # Generate speckle pattern array
            BlenderTools.add_image_texture(mat_data=mat_data, image_array=speckle_pattern)
        BlenderTools.uv_unwrap_part(part, mm_px_resolution, cal)

    def _debug_deform(self,
                      render_mesh: RenderMeshData,
                     sim_spat_dim:int,
                     part: bpy.data.objects) -> None:
        """A method to deform the Blender mesh object using the simulation results.
        This is done by taking the displacements to the nodes, and applying it
        in Blender. It should be noted that this only deforms the mesh without
        rendering any images, mainly useful for debugging code.

        Parameters
        ----------
        sim_data : mh.SimData
            A dataclass containing the simulation information i.e. the displacements
            to all the nodes in the mesh.
        part : bpy.data.objects
            The Blender part object which is to be deformed, normally as sample
            object.
        """
        render_mesh.coords = SimTools.centre_mesh_nodes(render_mesh.coords,
                                                        sim_spat_dim)
        timesteps = render_mesh.fields_render.shape[1]


        for timestep in range(1, timesteps):
            deformed_nodes = SimTools.get_deformed_nodes(timestep,
                                                         render_mesh)
            if deformed_nodes is not None:
                BlenderTools.deform_single_timestep(part, deformed_nodes)
                BlenderTools.set_new_frame(part)

    def render_single_image(self,
                            render_data: RenderData,
                            stage_image: bool | None = True) -> None | np.ndarray:
        """A method to render an images(s) of the current scene in Blender.
        Depending on the number of cameras, either one or two images will be
        rendered.

        Parameters
        ----------
        render_data : RenderData
            A dataclass containing the parameters needed to render an image.
        stage_image : bool | None, optional
            A flag that can be set to either save the rendered to disk or not.
            If set to False, an array of the image or stack of images will be
            returned, by default True. In order to output these images as an
            array, the image will first be saved to the disk and then bounced
            back as an array.

        Returns
        -------
        None | np.ndarray
            Nothing is returned if the image(s) is saved to disk (when save set
            to True). When save is set to False, the image array is returned.
            For a 2D system, an array with shape=(pixels_num_y, pixels_num_x) is
            returned. For a 3D system, a stack of arrays with
            shape=(pixels_num_y, pixels_num_x, 2) is returned.
        """
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = str(render_data.bit_size)
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = render_data.threads
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        if not render_data.base_dir.is_dir():
            raise BlenderError("The specified save directory does not exist")

        save_dir = render_data.base_dir / "blenderimages"
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(render_data.cam_data, tuple):
            cam_count = 0
            image_count = 0
            image_arrays = []
            for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
                bpy.context.scene.camera = cam
                cam_data_render = render_data.cam_data[cam_count]
                bpy.context.scene.render.resolution_x = cam_data_render.pixels_num[0]
                bpy.context.scene.render.resolution_y = cam_data_render.pixels_num[1]
                filename = "blenderimage_" + str(image_count) + "_" + str(cam_count) + ".tiff"
                filepath = save_dir / filename
                bpy.context.scene.render.filepath = str(filepath)
                if stage_image:
                    bpy.ops.render.render(write_still=True)
                    image_array = BlenderTools.save_render_as_array(filepath)
                    image_arrays.append(image_array)
                else:
                    bpy.ops.render.render(write_still=True)
                cam_count += 1
            if stage_image:
                image_arrays = np.dstack(image_arrays)
                return image_arrays
        else:
            image_count = 0
            bpy.context.scene.render.resolution_x = render_data.cam_data.pixels_num[0]
            bpy.context.scene.render.resolution_y = render_data.cam_data.pixels_num[1]
            filename = "blenderimage_" + str(image_count) + ".tiff"
            filepath = save_dir / filename
            bpy.context.scene.render.filepath = str(filepath)
            if stage_image:
                bpy.ops.render.render(write_still=True)
                image_array = BlenderTools.save_render_as_array(filepath)
                return image_array
            else:
                bpy.ops.render.render(write_still=True)

    def render_deformed_images(self,
                               render_mesh: RenderMeshData,
                               sim_spat_dim: int,
                               render_data:RenderData,
                               part: bpy.data.objects,
                               stage_image: bool | None = True) -> None | np.ndarray:
        """A method to deform the mesh object at all timesteps, and render
        image(s) at each timestep

        Parameters
        ----------
        render_mesh : RenderMeshData
            A dataclass containing the skimmed mesh and simulation information
            needed to deform the sample.
        sim_spat_dim: int
            The spatial dimension of the simulation.
        render_data : RenderData
            A dataclass containing the parameters necessary to render an image.
        part : bpy.data.objects
            The Blender part object to be deformed.
        stage_image : bool | None, optional
            A flag that can be set to save the rendered image to disk or not,
            by default True. In order to output these images as an
            array, the image will first be saved to the disk and then bounced
            back as an array.

        Returns
        -------
        None | np.ndarray
            Either nothing is returned if the image is saved
                to disk or a stack of image arrays are returned with the following
                dimensions: shape=(pixels_num_y, pixels_num_x, (num_timesteps + 1)
                for 2D setups and shape=(pixels_num_y, pixels_num_x, (num_timesteps + 1)*2)
                for 3D setups. The additional image is the reference image. For
                3D setups, the images in the stack alternate between camera 0 and
                camera 1.
        """
        render_mesh.coords = SimTools.centre_mesh_nodes(render_mesh.coords,
                                                        sim_spat_dim)
        timesteps = render_mesh.fields_render.shape[1]

        # Render parameters
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = str(render_data.bit_size)
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = render_data.threads
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        if not render_data.base_dir.is_dir():
            raise BlenderError("The specified save directory does not exist")

        save_dir = render_data.base_dir / "blenderimages"
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)

        image_arrays = []
        for timestep in range(0, timesteps):
            deformed_nodes = SimTools.get_deformed_nodes(timestep,
                                                         render_mesh)
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
                        filename = "blenderimage_" + str(timestep) + "_" + str(cam_count) + ".tiff"
                        filepath = save_dir / filename
                        bpy.context.scene.render.filepath = str(filepath)
                        if stage_image:
                            bpy.ops.render.render(write_still=True)
                            image_array = BlenderTools.save_render_as_array(filepath)
                            image_arrays.append(image_array)
                        else:
                            bpy.ops.render.render(write_still=True)
                        cam_count += 1
                else:
                    bpy.context.scene.render.resolution_x = render_data.cam_data.pixels_num[0]
                    bpy.context.scene.render.resolution_y = render_data.cam_data.pixels_num[1]
                    filename = "blenderimage_" + str(timestep) + ".tiff"
                    filepath = save_dir / filename
                    bpy.context.scene.render.filepath = str(filepath)
                    if stage_image:
                        bpy.ops.render.render(write_still=True)
                        image_array = BlenderTools.save_render_as_array(filepath)
                        image_arrays.append(image_array)
                    else:
                        bpy.ops.render.render(write_still=True)
        if stage_image:
            image_arrays = np.dstack(image_arrays)
            # TODO: Potentially change the way images are stacked for stereo systems
            # Change it so it suits Joel's code
            return image_arrays












