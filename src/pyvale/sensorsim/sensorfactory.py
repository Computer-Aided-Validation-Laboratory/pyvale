# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains helper functions that assemble common sensor array
configurations without the user needing to configure all the sub-components
themselves.
"""


import pyvale.mooseherder as mh
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.sensordescriptor import (DescriptorFactory,
                                               SensorDescriptor)
from pyvale.sensorsim.sensorspoint import SensorsPoint, SensorData
from pyvale.sensorsim.enums import EDim


class SensorFactory:
    @staticmethod
    def scalar_point(sim_data: mh.SimData,
                     sensor_data: SensorData,
                     comp_key: str,
                     spatial_dims: EDim,
                     descriptor: SensorDescriptor | None = None
                     ) -> SensorsPoint:
        """Helper function to assemble a scalar field point sensor array object
        based on the input simulation data, sensor data and specified physical
        field.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the physical field that the v
            virtual sensor array will sample.
        sensor_data : SensorData
            Sensor data object specifying the sensor array parameters such as
            the sensor positions and sampling times.
        comp_key : str
            String key to acces the physical field that the sensors will be
            applied to in the node_vars dictionary of the SimData object.
        spatial_dims : EDim
            Enumeration specifying the number of spatial dimensions the
            simulation uses as .TWOD or .THREED. Used to determine the element
            type for mesh-based data or the triangulation type for mesh free.
        descriptor : SensorDescriptor | None, optional
            Optional dataclass specifying the strings used to describe the
            sensor array such as the name of the field to be sensed and the
            units, by default None. If None then a default descriptor is
            created.

        Returns
        -------
        SensorArrayPoint
            The assembled point sensor array object.
        """
        if descriptor is None:
            descriptor = DescriptorFactory.scalar()

        s_field = FieldScalar(sim_data,comp_key,spatial_dims)

        sens_array = SensorsPoint(sensor_data,
                                      s_field,
                                      descriptor)
        return sens_array

    @staticmethod
    def vector_point(sim_data: mh.SimData,
                     sensor_data: SensorData,
                     comp_keys: tuple[str,...],
                     spatial_dims: EDim,
                     descriptor: SensorDescriptor | None = None,
                     ) -> SensorsPoint:
        """"Helper function to assemble a vector field point sensor array object
        based on the input simulation data, sensor data and specified physical
        field.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the physical field that the v
            virtual sensor array will sample.
        sensor_data : SensorData
            Sensor data object specifying the sensor array parameters such as
            the sensor positions and sampling times.
        comp_keys : tuple[str,...]
            Tuple of keys for the components of the vector field that will be
            sampled by the virtual sensors. For example: displacement fields in
            2D will have ("disp_x","disp_y").
        spatial_dims : EDim
            Enumeration specifying the number of spatial dimensions the
            simulation uses as .TWOD or .THREED. Used to determine the element
            type for mesh-based data or the triangulation type for mesh free.
        descriptor : SensorDescriptor | None, optional
            Optional dataclass specifying the strings used to describe the
            sensor array such as the name of the field to be sensed and the
            units, by default None. If None then a default descriptor is
            created.

        Returns
        -------
        SensorArrayPoint
            The assembled point sensor array object.
        """

        if descriptor is None:
            descriptor = DescriptorFactory.vector()

        disp_field = FieldVector(sim_data,
                                 comp_keys,
                                 spatial_dims)
        sens_array = SensorsPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array

    @staticmethod
    def tensor_point(sim_data: mh.SimData,
                     sensor_data: SensorData,
                     norm_comp_keys: tuple[str,...],
                     dev_comp_keys: tuple[str,...],
                     spatial_dims: EDim,
                     descriptor: SensorDescriptor | None = None,
                     ) -> SensorsPoint:
        """Helper function to assemble a tensor field point sensor array object
        based on the input simulation data, sensor data and specified physical
        field.

        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the physical field that the v
            virtual sensor array will sample.
        sensor_data : SensorData
            Sensor data object specifying the sensor array parameters such as
            the sensor positions and sampling times.
        norm_comp_keys : tuple[str,...]
            Tuple of string keys for the normal components of the tensor field
            in the node_vars dictionary of the SimData object. For example:
            strain fields in 2D will typically have ("strain_xx","strain_yy").
        dev_comp_keys : tuple[str,...]
            Tuple of string keys for the deviatoric components of the tensor
            field in the node_vars dictionary of the SimData object. For
            example: strain fields in 2D will typicall have ("strain_xy",).
        spatial_dims : EDim
            Enumeration specifying the number of spatial dimensions the
            simulation uses as .TWOD or .THREED. Used to determine the element
            type for mesh-based data or the triangulation type for mesh free.
        descriptor : SensorDescriptor | None, optional
            Optional dataclass specifying the strings used to describe the
            sensor array such as the name of the field to be sensed and the
            units, by default None. If None then a default descriptor is
            created.
        Returns
        -------
        SensorArrayPoint
            The assembled point sensor array object.
        """

        if descriptor is None:
            descriptor = DescriptorFactory.tensor(spatial_dims)

        strain_field = FieldTensor(sim_data,
                                   norm_comp_keys,
                                   dev_comp_keys,
                                   spatial_dims)
        sens_array = SensorsPoint(sensor_data,
                                      strain_field,
                                      descriptor)

        return sens_array



