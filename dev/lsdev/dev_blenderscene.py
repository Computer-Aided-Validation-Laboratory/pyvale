import bpy
from mooseherder import SimData
from dev_blendercamera import CameraData, CameraBlender
from dev_lightingblender import LightData, BlenderLight
from dev_partblender import BlenderPart
from dev_objectmaterial import MaterialData, BlenderMaterial

class BlenderScene:
    def __init__(self):
        bpy.ops.wm.read_factory_settings(use_empty=True) #Blender starts with empty scene, without default objects
        self.default_settings()

    def default_settings(self):
        """Sets the default settings in Blender:
        - Sets the units to mm

        # part = partmaker.add_thickness(part=part)
        - Sets the background to black
        """
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

    def add_light(self, light_data: LightData):
        lightmaker = BlenderLight(light_data)
        light = lightmaker.add_light()
        return light

    def add_camera(self, cam_data: CameraData):
        cameramaker = CameraBlender(cam_data)
        camera = cameramaker.add_camera()

        return camera

    def add_part(self, sim_data: SimData):
        partmaker = BlenderPart(sim_data)
        part = partmaker.simdata_to_part()

        self._set_origin(part)

        return part

    def set_part_location(self, part, location: tuple):
        part.location = location

    def set_part_roation(self, part, rotation: tuple):
        part.rotation_mode = 'XYZ'
        part.rotation_euler = rotation

    def add_material(self, mat_data: MaterialData, part, image_path: str):
        materialmaker = BlenderMaterial(mat_data, part, image_path)
        mat = materialmaker.add_material()

        return mat

    def _set_origin(self, part):
        # Not sure if this is necessary
        bpy.ops.object.select_all(action='DESELECT')
        part.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

    def save_model(self, filepath: str):
        '''
        Method to save the blender model to a file
        '''
        if filepath is not None:
            bpy.ops.wm.save_as_mainfile(filepath=filepath)




