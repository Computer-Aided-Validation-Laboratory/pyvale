
# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

from dataclasses import dataclass
import numpy as np

#TODO: Docstrings

@dataclass(slots=True)
class SensorDescriptor:
    name: str = 'Measured Value'
    units: str = r"-"
    time_units: str = r"s"
    symbol: str = r"m"
    tag: str = 'S'
    components: tuple[str,...] | None = None

    def create_label(self, comp_ind: int | None = None) -> str:
        label = ""
        if self.name != "":
            label = label + rf"{self.name} "


        symbol = rf"${self.symbol}$ "
        if comp_ind is not None and self.components is not None:
            symbol = rf"${self.symbol}_{{{self.components[comp_ind]}}}$ "
        if symbol != "":
            label = label + symbol

        if self.units != "":
            label = label + "\n" + rf"[${self.units}$]"

        return label

    def create_label_flat(self, comp_ind: int | None = None) -> str:
        label = ""
        if self.name != "":
            label = label + rf"{self.name} "


        symbol = rf"${self.symbol}$ "
        if comp_ind is not None and self.components is not None:
            symbol = rf"${self.symbol}_{{{self.components[comp_ind]}}}$ "
        if symbol != "":
            label = label + symbol

        if self.units != "":
            label = label + " " + rf"[${self.units}$]"

        return label

    def create_sensor_tags(self,n_sensors: int) -> list[str]:
        z_width = int(np.log10(n_sensors))+1

        sensor_names = list()
        for ss in range(n_sensors):
            num_str = f'{ss+1}'.zfill(z_width)
            sensor_names.append(f'{self.tag}{num_str}')

        return sensor_names


class SensorDescriptorFactory:
    @staticmethod
    def temperature_descriptor() -> SensorDescriptor:
        descriptor = SensorDescriptor()
        descriptor.name = 'Temp.'
        descriptor.symbol = 'T'
        descriptor.units = r'^{\circ}C'
        descriptor.tag = 'TC'
        return descriptor

    @staticmethod
    def displacement_descriptor() -> SensorDescriptor:
        descriptor = SensorDescriptor()
        descriptor.name = 'Disp.'
        descriptor.symbol = 'u'
        descriptor.units = r'm'
        descriptor.tag = 'DS'
        descriptor.components = ('x','y','z')
        return descriptor

    @staticmethod
    def strain_descriptor(spat_dims: int = 3) -> SensorDescriptor:
        descriptor = SensorDescriptor()
        descriptor.name = 'Strain'
        descriptor.symbol = r'\varepsilon'
        descriptor.units = r'-'
        descriptor.tag = 'SG'

        if spat_dims == 2:
            descriptor.components = ('xx','yy','xy')
        else:
            descriptor.components = ('xx','yy','zz','xy','yz','xz')

        return descriptor
