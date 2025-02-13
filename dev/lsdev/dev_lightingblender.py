from dataclasses import dataclass
from enum import Enum
import numpy as np
import bpy

class LightType(Enum):
    POINT = 'POINT'
    SUN = 'SUN'
    SPOT = 'SPOT'
    AREA = 'AREA'

@dataclass
class LightData():
    type: LightType | None = LightType.POINT
    position: np.ndarray | None = (0, 0, 10)
    orientation: np.ndarray | None = (0, 0, 0)
    energy: int | None = 10
    part_dimension: np.ndarray | None = None


class LightBlender():
    def __init__(self, LightData):
        self.light_data = LightData

    def create_light(self):
        # TODO: Add different options for different light types
        type = self.light_data.type.value
        name = type.capitalize() + 'Light'
        light = bpy.data.lights.new(name=name, type=type)
        light_ob = bpy.data.objects.new(name=name, object_data=light)

        light_ob.location = (self.light_data.position[0],
                                   self.light_data.position[1],
                                   self.light_data.position[2])

        light_ob.rotation_mode = 'XYZ'
        light_ob.rotation_euler = self.light_data.orientation

        light.energy = self.light_data.energy

        bpy.context.collection.objects.link(light_ob)

        return light_ob



