import os
from dataclasses import dataclass
import bpy
import numpy as np
from PIL import Image
import cv2

@dataclass
class MaterialData():
    roughness: float | None = 1
    metallic: float | None = 0
    interpolant = 'Cubic'
    cal: bool = False
    # TODO: add other material properties to here

class MaterialBlender():
    def __init__(self, MaterialData, object, image_path):
        self.mat_data = MaterialData
        self.object = object
        self.image_path = image_path
        self.mat = None
        self.tree = None
        self.nodes = None

    def _uv_unwrap(self, FOV_mm):
        """Method to UV unwrap object before adding material.
           Object needs to be unwrapped for image texture to apply to it
        """
        self.object.select_set(True)
        bpy.context.view_layer.objects.active = self.object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        FOV_mm = FOV_mm
        cube_size = FOV_mm / 1
        if self.mat_data.cal is not True:
            bpy.ops.uv.cube_project(scale_to_bounds = False,
                                    correct_aspect=True,
                                    cube_size = cube_size)
        else:
            bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.mode_set(mode='OBJECT')
        self.object.select_set(False)

    def _clear_nodes(self):
        """Method to clear any existing material nodes
        """
        self.object.select_set(True)
        self.mat = bpy.data.materials.new(name='Material')
        self.mat.use_nodes = True
        self.object.active_material = self.mat
        self.tree = self.mat.node_tree
        self.nodes = self.tree.nodes
        self.nodes.clear()

    def _set_image_texture(self):
        bsdf = self.nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Roughness'].default_value = self.mat_data.roughness
        bsdf.inputs['Metallic'].default_value = self.mat_data.metallic

        print(f"{self.mat=}")
        tex_image = self.nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (0, 0)

        if os.path.exists(self.image_path):
            tex_image.image = bpy.data.images.load(self.image_path)
        else:
            image_array = np.zeros((3000, 3000))
            image_array[:1500] = 1
            img = Image.fromarray(image_array).convert('RGBA')
            new_image_array = np.array(img)
            print(f"{new_image_array.shape=}")
            blender_img = bpy.data.images.new('Speckle', width=3000, height=3000)
            pixels = new_image_array.flatten()
            blender_img.pixels = pixels


            blender_img.pixels = pixels
            blender_img.update()
            tex_image.image = blender_img
            blender_img.filepath_raw = '/home/lorna/pyvale/dev/lsdev/rendered_images/speckle.png'
            blender_img.file_format = 'PNG'
            # blender_img.save()  # Only sets image tex correctly if save image??



        tex_image.interpolation = self.mat_data.interpolant

        output = self.nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (0, 0)

        self.tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        self.tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        obj = bpy.data.objects.get('part')
        if obj:
            obj.active_material = self.mat

        return tex_image

    def add_material(self, FOV_mm):
        self._clear_nodes()
        self._set_image_texture()
        self._uv_unwrap(FOV_mm)

        return self.mat







