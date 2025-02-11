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
        self._light_ob = None
        self._light = None

    def _create_light(self):
        # TODO: Add different options for different light types
        type = self.light_data.type.value
        name = type.capitalize() + 'Light'
        self._light = bpy.data.lights.new(name=name, type=type)
        self._light_ob = bpy.data.objects.new(name=name, object_data=self._light)
        bpy.context.collection.objects.link(self._light_ob)

    def _set_location(self):
        self._light_ob.location = (self.light_data.position[0],
                                   self.light_data.position[1],
                                   self.light_data.position[2])

    def _set_rotation(self):
        self._light_ob.rotation_mode = 'XYZ'
        self._light_ob.rotation_euler = self.light_data.orientation

    def _set_energy(self):
        self._light.energy = self.light_data.energy

    def add_light(self):
        self._create_light()
        self._set_location()
        self._set_rotation()
        self._set_energy()

        return self._light_ob


