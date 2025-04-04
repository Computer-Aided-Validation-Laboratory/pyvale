"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import pyvista as pv
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from PIL import Image
from multiprocessing import cpu_count
import bpy
import pyvale
import mooseherder as mh
from pyvale.core.cameradata import CameraData
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
        """A method to save the current Blender file to a .blend filepath

        Args:
            filepath (Path): The filepath to which the Blender file should be
                saved
            override (bool, optional): A flag which can be set to True or False.
                If set to True, if the specified filepath already exists, this
                file will automatically be overwritten. If set to False and the
                specified filepath already exists, an error will be thrown.
                Defaults to False.

        Raises:
            BlenderError: "A file already exists with this filepath"
        """
        if filepath.exists():
                if override is True:
                    filepath.unlink()
                else:
                    raise BlenderError("A file already exists with this filepath")
        filepath = str(filepath)
        bpy.ops.wm.save_as_mainfile(filepath=filepath)

    @staticmethod
    def move_blender_part(pos_world: np.ndarray, part) -> None:
        """A method to move the part object within Blender

        Args:
            pos_world (np.ndarray): The position, as a vector, to which the part
                should be moved to
            part (bpy.data.objects['Part']): The Blender mesh object to be
                moved
        """
        z_location = int(part.dimensions[2])
        part.location = (pos_world[0], pos_world[1], (pos_world[2] - z_location))

    @staticmethod
    def rotate_blender_part(rot_world: Rotation, part) -> None:
        """A method to rotate the part object within Blender

        Args:
            rot_world (Rotation): The rotation that the part should have
            part (bpy.data.objects['Part']): The Blender mesh object to be
                rotated
        """
        part.rotation_mode = "XYZ"
        part_rotation = rot_world.as_euler("xyz", degrees=False)
        part.rotation_euler = part_rotation

    @staticmethod
    def set_new_frame(part) -> None:
        """A method to set a new frame within Blender (needed to differentiate
        the timesteps)

        Args:
            part (bpy.data.objects['Part']): The Blender mesh object, to ensure
                that it is the active object
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
    def deform_single_timestep(part, deformed_nodes: np.ndarray):
        """A method to deform the part for a single timestep, given the node
        positions the nodes will move to

        Args:
            part (bpy.data.objects['Part']): The Blender mesh object to be
                deformed
            deformed_nodes (np.ndarray): The deformed positions of each node in
                the surface mesh

        Returns:
            part: The deformed Blender mesh object
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
    def clear_material_nodes(part) -> None:
        """A method to clear any existing material nodes from the specified
        Blender object

        Args:
            part (bpy.data.objects['Part']): The Blender object to which a
                material will be applied
        """
        part.select_set(True)
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        part.active_material = mat
        tree = mat.node_tree
        nodes = tree.nodes
        nodes.clear()

    @staticmethod
    def uv_unwrap_part(part, resolution: float, cal: bool = False) -> None:
        """A method to UV unwrap the Blender object, in order to apply a speckle
        image texture

        Args:
            part (bpy.data.objects['Part']): The Blender object to be unwrapped
            FOV_x (float): The horizontal field of view, in order to scale the
                speckle image texture for an optimal number of pixels per speckle
            cal (bool, optional): A flag that can set to True or False. If set
                to True, the uv unwrap scales to the bounds of the object, which
                is needed when applying the calibration target image texture.
                Defaults to False.
        """
        part.select_set(True)
        bpy.context.view_layer.objects.active = part
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        cube_size = resolution * 1500 #TODO: Change this back

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
        """A method to add an image texture to a Blender object

        Args:
            mat_data (BlenderMaterialData): A dataclass containing the material
                parameters
            image_path (Path | None, optional): The filepath for the speckle
                image file. Defaults to None.
            image_array (np.ndarray | None, optional): A speckle image array.
                Defaults to None.

        Raises:
            BlenderError: "Image texture filepath does not exist"
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
        image = Image.open(filepath)
        image_array = np.asarray(image)
        filepath.unlink()
        return image_array

    @staticmethod
    def generate_calib_file(stereo_data: CameraStereoData,
                            calib_filepath: Path,
                            calib_filename: str | None = None) -> None:
        """A method to generate a .caldat calibration file, compatible with
        MatchID

        Args:
            stereo_data (CameraStereoData): A dataclass containing the parameters
                of the stereo setup
            calib_filepath (Path): The file directory to which the calibration
                file is to be saved to
            calib_filename (str | None, optional): The filename the calibration
                should be saved as. Defaults to None.
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
                           part):
        # Render parameters
        bpy.context.scene.render.engine = render_data.engine.value
        bpy.context.scene.render.image_settings.color_mode = "BW"
        bpy.context.scene.render.image_settings.color_depth = render_data.bit_size
        bpy.context.scene.render.threads_mode = "FIXED"
        bpy.context.scene.render.threads = int(cpu_count())
        bpy.context.scene.render.image_settings.file_format = "TIFF"

        if render_data.engine == RenderEngine.CYCLES:
            bpy.context.scene.cycles.samples = render_data.samples
            bpy.context.scene.cycles.max_bounces = render_data.max_bounces
        elif render_data.engine == RenderEngine.EEVEE:
            bpy.context.scene.eevee.taa_render_samples = render_data.samples

        render_counter = 0
        for plunge in range(calibration_data.plunge_lims[0],
                            calibration_data.plunge_lims[1],
                            calibration_data.plunge_step):
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
                    for ii in range(angle_steps):
                        angle = calibration_data.angle_lims[0] + calibration_data.angle_step * ii

                        # Rotate around x-axis
                        print(f"{angle=}")
                        rotation  = (np.radians(angle), 0, 0)
                        part.rotation_mode = 'XYZ'
                        part.rotation_euler = rotation
                        for jj in range(angle_steps):
                            angle = calibration_data.angle_lims[0] + calibration_data.angle_step * jj
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
                                    bpy.ops.render.render(write_still=True)
                                    cam_count += 1
                            render_counter += 1
        print('Total number of calibration images = ' + str(render_counter))




