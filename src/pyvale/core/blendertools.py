"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image
from multiprocessing import cpu_count
import bpy
import pyvale
from pyvale.core.blendermaterialdata import BlenderMaterialData
from pyvale.core.camerastereodata import CameraStereoData
from pyvale.core.blenderrenderdata import RenderData, RenderEngine
from pyvale.core.blendercalibrationdata import CalibrationData

# NOTE: This module is a feature under development

class BlenderError(Exception):
    pass

class BlenderTools():
    """Namespace for tools used within the pyvale Blender module.
    """

    @staticmethod
    def save_blender_file(filepath: Path, override: bool = False) -> None:
        """A method to save the current Blender scene to a Blender .blend filepath

        Parameters
        ----------
        filepath : Path
            The filepath to which the Blender file should be saved.
        override : bool, optional
            A flag which can be set to True or False. If set to True, if the
            specified filepath already exists, this file will be automatically
            overwritten. If set to False and the specified filepath already exists
            an error will be thrown. If the specified filepath does not exist,
            the file will be saved normally, by default False

        Raises
        ------
        BlenderError
            "A file already exists with this filepath". This error is thrown
            when override is set to False, and the specified filepath already
            exists.
        """
        if filepath.exists():
                if override is True:
                    filepath.unlink()
                else:
                    raise BlenderError("A file already exists with this filepath")
        filepath = str(filepath)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)

    @staticmethod
    def move_blender_part(pos_world: np.ndarray, part: bpy.data.objects) -> None:
        """A method to move the part object within Blender.

        Parameters
        ----------
        pos_world : np.ndarray
            A array describing the vector position to which the part should be
            moved to.
        part : bpy.data.objects
            The Blender part object to be moved.
        """
        z_location = int(part.dimensions[2])
        part.location = (pos_world[0], pos_world[1], (pos_world[2] - z_location))

    @staticmethod
    def rotate_blender_part(rot_world: Rotation, part: bpy.data.objects) -> None:
        """A method to rotate the part object within Blender.

        Parameters
        ----------
        rot_world : Rotation
            The rotation that is to be applied to the part object.
        part : bpy.data.objects
            The Blender part object to be rotated.
        """
        part.rotation_mode = "XYZ"
        part_rotation = rot_world.as_euler("xyz", degrees=False)
        part.rotation_euler = part_rotation

    @staticmethod
    def set_new_frame(part: bpy.data.objects) -> None:
        """A method to set a new frame within Blender (needed to differenciate
        the timesteps).

        Parameters
        ----------
        part : bpy.data.objects
            The Blender part object, normally the sample object. This is passed
            in to ensure it is the active object within the scene.
        """
        frame_incr = 20
        ob = bpy.context.view_layer.objects.active
        if ob is None:
            bpy.context.objects.active = part

        current_frame = bpy.context.scene.frame_current
        current_frame += frame_incr
        bpy.context.scene.frame_set(current_frame)

        bpy.data.shape_keys["Key"].eval_time = current_frame
        part.data.shape_keys.keyframe_insert("eval_time", frame=current_frame)
        bpy.context.scene.frame_end = current_frame

    @staticmethod
    def deform_single_timestep(part: bpy.data.objects,
                               deformed_nodes: np.ndarray) -> bpy.data.objects:
        """A method to deform the part for a single timestep, given the node
        positions the nodes will move to.

        Parameters
        ----------
        part : bpy.data.objects
            The Blender part object to be deformed, normally the sample object.
        deformed_nodes : np.ndarray
            An array of the deformed positions of each node in the surface mesh.

        Returns
        -------
        bpy.data.objects
            The deformed Blender part object.
        """
        if part.data.shape_keys is None:
            part.shape_key_add()
            BlenderTools.set_new_frame(part)
        shape_key = part.shape_key_add()
        part.data.shape_keys.use_relative = False

        n_nodes_layer = int(len(part.data.vertices))
        for i in range(len(part.data.vertices)):
            if i < n_nodes_layer:
                shape_key.data[i].co = deformed_nodes[i]
        return part

    @staticmethod
    def clear_material_nodes(part: bpy.data.objects) -> None:
        """A method to clear any existing material nodes from the specified
        Blender object.

        Parameters
        ----------
        part : bpy.data.objects
            The Blender part object to which a material will be applied.
        """
        part.select_set(True)
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        part.active_material = mat
        tree = mat.node_tree
        nodes = tree.nodes
        nodes.clear()

    @staticmethod
    def uv_unwrap_part(part: bpy.data.objects,
                       resolution: float,
                       cal: bool = False) -> None:
        """A method to UV unwrap the Blender object, in order to apply a speckle
        image texture.

        Parameters
        ----------
        part : bpy.data.objects
            The Blender part object to be unwrapped, normally the sample object.
        resolution : float
            The mm/px resolution of the rendered image, used to size the UV unwrapping.
        cal : bool, optional
            A flag that can be set when UV unwrapping a calibration target as the
            sizing differs, by default False
        """
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        cube_size = resolution * 1500
        # TODO: Add capability here to uv unwrap non-rectangular objects
        if cal is not True:
            bpy.ops.uv.cube_project(scale_to_bounds = False,
                                    correct_aspect=True,
                                    cube_size = cube_size)
        else:
            bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.mode_set(mode="OBJECT")
        part.select_set(False)

    @staticmethod
    def add_image_texture(mat_data: BlenderMaterialData,
                          image_path: Path | None = None,
                          image_array: np.ndarray | None = None) -> None:
        """A method to add an image texture to a Blender object, this will
        primarily be used for applying a speckle pattern to a sample object.

        Parameters
        ----------
        mat_data : BlenderMaterialData
            A dataclass containing the material parameters, including roughness
        image_path : Path | None, optional
            The filepath for the speckle image file. If provided, that image will
            be used, by default None
        image_array : np.ndarray | None, optional
            An 2D array of a speckle image. If provided, this image will be used,
            by default None

        Raises
        ------
        BlenderError
            "Image texture filepath does not exist". This error is thrown when
            neither a filepath nor an image array have been provided
        """
        mat_nodes = bpy.data.materials["Material"].node_tree.nodes
        bsdf = mat_nodes.new(type="ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        bsdf.inputs["Roughness"].default_value = mat_data.roughness
        bsdf.inputs["Metallic"].default_value = mat_data.metallic

        node_tree = bpy.data.materials["Material"].node_tree
        tex_image = node_tree.nodes.new(type="ShaderNodeTexImage")
        tex_image.location = (0, 0)

        if image_array is None:
            if image_path.exists:
                tex_image.image = bpy.data.images.load(str(image_path))
            else:
                raise BlenderError("Image texture filepath does not exist")

        if image_array is not None:
            size = image_array.shape
            image = Image.fromarray(image_array).convert("RGBA")
            new_image_array = np.array(image)
            blender_image = bpy.data.images.new("Speckle",
                                                width=size[0],
                                                height=size[1])
            pixels = new_image_array.flatten()
            blender_image.pixels = pixels
            blender_image.update()
            tex_image.image = blender_image


        tex_image.interpolation = mat_data.interpolant

        output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
        output.location = (0, 0)

        node_tree.links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
        node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        obj = bpy.data.objects.get("Part")
        if obj:
            obj.active_material = bpy.data.materials["Material"]

    @staticmethod
    def save_render_as_array(filepath: Path) -> np.ndarray:
        """Method to save a rendered image as an array. This method write the
        image to the disk and then extracts it

        Parameters
        ----------
        filepath : Path
            The filepath to which the image is saved

        Returns
        -------
        np.ndarray
            The rendered image as an array with the following dimensions:
            shape=(pixels_num_y, pixels_num_x)
        """
        image = Image.open(filepath)
        image_array = np.asarray(image)
        filepath.unlink()
        return image_array

    @staticmethod
    def generate_calib_file(stereo_data: CameraStereoData,
                            calib_filepath: Path,
                            calib_filename: str | None = None) -> None:
        """A method to generate a .caldat calibration file, compatible with
        MatchID. The intrinsic and extrinsic parameters are calculated directly
        from Blender

        Parameters
        ----------
        stereo_data : CameraStereoData
            A dataclass containing the parameters of the stereo setup, most
            importantly the individual CameraData classes
        calib_filepath : Path
            The file directory to which the calibration file is to be saved to
        calib_filename : str | None, optional
            The filename the calibration file should be saved as. If None then
            filename is saved as calib.caldat, by default None
        """
        if Path(calib_filepath).is_dir() is False:
            Path.mkdir(calib_filepath)
        if calib_filename is not None:
            calib_filepath = calib_filepath / calib_filename
        else:
            calib_filepath = calib_filepath / 'calib.caldat'
        with open(calib_filepath, "w") as file:
            file.write(f'Cam0_Fx [pixels]; {stereo_data.cam_data_0.focal_length/ stereo_data.cam_data_0.pixels_size[0]}\n')
            file.write(f'Cam0_Fy [pixels]; {stereo_data.cam_data_0.focal_length/ stereo_data.cam_data_0.pixels_size[1]}\n')
            file.write("Cam0_Fs [pixels];0\n")
            file.write(f'Cam0_Kappa 1;{stereo_data.cam_data_0.k1}\n')
            file.write(f'Cam0_Kappa 2;{stereo_data.cam_data_0.k2}\n')
            file.write(f'Cam0_Kappa 3;{stereo_data.cam_data_0.k3}\n')
            file.write(f'Cam0_P1;{stereo_data.cam_data_0.p1}\n')
            file.write(f'Cam0_P2;{stereo_data.cam_data_0.p2}\n')
            file.write(f'Cam0_Cx [pixels];{stereo_data.cam_data_0.c0}\n')
            file.write(f'Cam0_Cy [pixels];{stereo_data.cam_data_0.c1}\n')
            file.write(f'Cam1_Fx [pixels]; {stereo_data.cam_data_1.focal_length/ stereo_data.cam_data_1.pixels_size[0]}\n')
            file.write(f'Cam1_Fy [pixels]; {stereo_data.cam_data_1.focal_length/ stereo_data.cam_data_1.pixels_size[1]}\n')
            file.write("Cam1_Fs [pixels];0\n")
            file.write(f'Cam1_Kappa 1;{stereo_data.cam_data_1.k1}\n')
            file.write(f'Cam1_Kappa 2;{stereo_data.cam_data_1.k2}\n')
            file.write(f'Cam1_Kappa 3;{stereo_data.cam_data_1.k3}\n')
            file.write(f'Cam1_P1;{stereo_data.cam_data_1.p1}\n')
            file.write(f'Cam1_P2;{stereo_data.cam_data_1.p2}\n')
            file.write(f'Cam1_Cx [pixels];{stereo_data.cam_data_1.c0}\n')
            file.write(f'Cam1_Cy [pixels];{stereo_data.cam_data_1.c1}\n')
            file.write(f"Tx [mm];{stereo_data.stereo_dist[0]}\n")
            file.write(f"Ty [mm];{stereo_data.stereo_dist[1]}\n")
            file.write(f"Tz [mm];{stereo_data.stereo_dist[2]}\n")
            stereo_rotation = stereo_data.stereo_rotation.as_euler("xyz", degrees=True)
            file.write(f"Theta [deg];{stereo_rotation[0]}\n")
            file.write(f"Phi [deg];{stereo_rotation[1]}\n")
            file.write(f"Psi [deg];{stereo_rotation[2]}")

    def calibration_images(render_data: RenderData,
                           calibration_data: CalibrationData,
                           part: bpy.data.objects,
                           render: bool = True) -> int:
        """A method to render a set of calibration images, which can be used to
        calculate the intrinsic and extrinsic parameters.

        Parameters
        ----------
        render_data : RenderData
            The
        calibration_data : CalibrationData
            A dataclass containing the parameters by which to move the calibration
            target. These inclcude the plungle depth and rotation angle.
        part : bpy.data.objects
            The Blender part object, in this instance the calibration target.
        render : bool, optional
            A flag that can be set to render/not render the images. This can be
            set to False to just return the number of calibration images that
            will be rendered, by default True.

        Returns
        -------
        int
            The number of calibration images that will be rendered. This is
            dependant on the values set within the CalibrationData dataclass.
        """
        # Render parameters
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = str(render_data.bit_size)
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = int(cpu_count())
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        render_counter = 0
        plunge_steps = int(((calibration_data.plunge_lims[1] -
                             calibration_data.plunge_lims[0]) /
                             calibration_data.plunge_step) + 1)
        for ii in range(plunge_steps):
            plunge = calibration_data.plunge_lims[0] + calibration_data.plunge_step * ii
            # Plunge
            (FOV_x, FOV_y) = pyvale.blender_FOV(render_data.cam_data[0])
            x_limit = int(round((FOV_x / 2) - (part.dimensions[0] / 2)))

            y_limit = int(round((FOV_y / 2) - (part.dimensions[1] / 2)))

            for x in np.arange(-1, 2):
                x *= x_limit
                # Move in x-dir
                for y in np.arange(-1, 2):
                    y *= y_limit
                    # Move in y-dir
                    part.location = ((x, y, plunge))
                    part.location[2] = plunge
                    angle_steps = int(((calibration_data.angle_lims[1] -
                                   calibration_data.angle_lims[0]) /
                                     calibration_data.angle_step) + 1)
                    for jj in range(angle_steps):
                        angle = calibration_data.angle_lims[0] + calibration_data.angle_step * jj

                        # Rotate around x-axis
                        rotation  = (np.radians(angle), 0, 0)
                        part.rotation_mode = 'XYZ'
                        part.rotation_euler = rotation
                        for kk in range(angle_steps):
                            angle = calibration_data.angle_lims[0] + calibration_data.angle_step * kk
                            # Rotate around y-axis
                            rotation  = (0, np.radians(angle), 0)
                            part.rotation_mode = 'XYZ'
                            part.rotation_euler = rotation

                            if isinstance(render_data.cam_data, tuple):
                                cam_count = 0
                                for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
                                    bpy.context.scene.camera = cam
                                    cam_data_render = render_data.cam_data[cam_count]
                                    bpy.context.scene.render.resolution_x = cam_data_render.pixels_num[0]
                                    bpy.context.scene.render.resolution_y = cam_data_render.pixels_num[1]
                                    filename = render_data.save_name + "_" + str(render_counter) + "_" + str(cam_count) + ".tiff"
                                    bpy.context.scene.render.filepath = str(render_data.save_dir / filename)
                                    if render is True:
                                        bpy.ops.render.render(write_still=True)
                                    cam_count += 1
                            render_counter += 1
        print('Total number of calibration images = ' + str(render_counter))
        return render_counter




