
# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module is used to create sensor descriptors which are strings used to label
plots and visualisations for virtual sensor simulations.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SensorDescriptor:
    """Dataclass for storing string descriptors for sensor array vis2ualisation.
    Used for labelling matplotlib and pyvista plots with the sensor name,
    physical units and other descriptors.
    """

    name: str = "Measured Value"
    """String describing the field that the sensor measures e.g. temperature
    , strain etc. Defaults to 'Measured Value'.
    """

    units: str = r"-"
    """String describing the sensor measurement units. Defaults to '-'. Latex
    symbols can be used with a raw string.
    """

    time_units: str = r"s"
    """String describing time units. Defaults to 's'.
    """

    symbol: str = r"m"
    """Symbol for describing the field the sensor measures. For example 'T' for
    temperature of r'\epsilon' for strain. Latex symbols can be used with a raw
    string.
    """

    tag: str = "S"
    """String shorthand tag used to label sensors on pyvista plots. Defaults to
    'S'.
    """

    components: tuple[str,...] | None = None
    """Tuple of strings describing the field components. Defaults to None which
    is used for scalar fields. For vector fields use ('x','y','z') for 3D and
    for tensor fields use ('xx','yy','zz','xy','yz','xz').
    """


    def create_label(self, comp_ind: int | None = None) -> str:
        """Creates an axis label for a matplotlib plot based on the sensor
        descriptor string. The axis label takes the form: 'name, symbol [units]'
        This version creates a label with line breaks which is useful for
        vertical colourbars.

        Parameters
        ----------
        comp_ind : int | None, optional
            Index of the field component to create a label for, by default None.
            If None the first field component is used.

        Returns
        -------
        str
            Axis label for field component in the form: 'name, symbol [units]'.
        """
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
        """Creates an axis label for a matplotlib plot based on the sensor
        descriptor string. The axis label takes the form: 'name, symbol [units]'
        This version creates a label with no line breaks which is useful for
        axis labels on plots.

        Parameters
        ----------
        comp_ind : int | None, optional
            Index of the field component to create a label for, by default None.
            If None the first field component is used.

        Returns
        -------
        str
            Axis label for field component in the form: 'name, symbol [units]'.
        """
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
        """Creates a list of numbered sensor tags for labelling sensor locations
        or for graph legends. Tags are shorthand names for sensors such as TC
        for thermocouples or SG for strain gauges.

        Parameters
        ----------
        n_sensors : int
            The number of sensors to create tags for.

        Returns
        -------
        list[str]
            A list of sensor tags
        """
        z_width = int(np.log10(n_sensors))+1

        sensor_names = list()
        for ss in range(n_sensors):
            num_str = f"{ss+1}".zfill(z_width)
            sensor_names.append(f"{self.tag}{num_str}")

        return sensor_names


class SensorDescriptorFactory:
    """A factory for building common sensor descriptors for scalar, vector and
    tensor fields. Builds descriptors for thermcouples, displacement sensors
    and strain sensors.
    """

    @staticmethod
    def temperature_descriptor() -> SensorDescriptor:
        """Creates a generic temperature sensor descriptor. Assumes the sensor
        is measuring a temperature in degrees C.

        Returns
        -------
        SensorDescriptor
            The default temperature sensor descriptor.
        """
        descriptor = SensorDescriptor(name="Temp.",
                                      symbol="T",
                                      units=r"^{\circ}C",
                                      tag="TC")
        return descriptor

    @staticmethod
    def displacement_descriptor() -> SensorDescriptor:
        """Creates a generic displacement sensor descriptor. Assumes units of mm
        and vector components of x,y,z.

        Returns
        -------
        SensorDescriptor
            The default displacement sensor descriptor.
        """
        descriptor = SensorDescriptor(name="Disp.",
                                      symbol="u",
                                      units=r"mm",
                                      tag="DS",
                                      components=("x","y","z"))
        return descriptor

    @staticmethod
    def strain_descriptor(spat_dims: int = 3) -> SensorDescriptor:
        """Creates a generic strain sensor descriptor. Assumes strain is
        unitless and that the components are xx,yy,xy for 2D and xx,yy,zz,xy,yz,
        xz for 3D.

        Parameters
        ----------
        spat_dims : int, optional
            Number of spatial dimensions used for setting the components of the
            tensor strain field, by default 3.

        Returns
        -------
        SensorDescriptor
            The default strain sensor descriptor.
        """
        descriptor = SensorDescriptor(name="Strain",
                                      symbol=r"\varepsilon",
                                      units=r"-",
                                      tag="SG")

        if spat_dims == 2:
            descriptor.components = ("xx","yy","xy")
        else:
            descriptor.components = ("xx","yy","zz","xy","yz","xz")

        return descriptor
