import os
from dataclasses import dataclass
import bpy

@dataclass
class MaterialData():
    roughness: float | None = 0.5
    metallic: float | None = 0
    interpolant = 'Cubic'
    # TODO: add other material properties to here

class BlenderMaterial():
    def __init__(self, MaterialData, object, image_path):
        self.mat_data = MaterialData
        self.object = object
        self.image_path = image_path
        self.mat = None
        self.tree = None
        self.nodes = None

    def _uv_unwrap(self):
        self.object.select_set(True)
        bpy.context.view_layer.objects.active = self.object
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
        self.object.select_set(False)

    def _clear_nodes(self):
        self.object.select_set(True)
        self.mat = bpy.data.materials.new(name='Material') # add this to init?
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

        tex_image = self.nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (0, 0)

        if os.path.exists(self.image_path):
            tex_image.image = bpy.data.images.load(self.image_path)
        else:
            print('Failed to load image')

        tex_image.interpolation = self.mat_data.interpolant

        output = self.nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (0, 0)

        self.tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        self.tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        obj = bpy.data.objects.get('part')
        if obj:
            obj.active_material = self.mat

    def add_material(self):
        self._uv_unwrap()
        self._clear_nodes()
        self._set_image_texture()

        return self.mat






