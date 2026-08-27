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

import numpy as np
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.sensordescriptor import (
    DescriptorFactory,
    SensorDescriptor,
)
from pyvale.sensorsim.sensorspoint import SensorsPoint, SensorData
from pyvale.sensorsim.sensorsspatial import SensorsSpatial
from pyvale.sensorsim.enums import EDim, EIntegrationMode
from pyvale.sensorsim.spatialwindows import (
    SpatialWindowLine,
    SpatialWindowRectangle,
    SpatialWindowBox,
)
from pyvale.sensorsim.sensortools import orient_from_direction



class SensorFactory:
    @staticmethod
    def scalar_point(sim_data: SimData,
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
        sim_data : SimData
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
    def vector_point(sim_data: SimData,
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
        sim_data : SimData
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
    def tensor_point(sim_data: SimData,
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
        sim_data : SimData
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

        strain_field = FieldTensor(
            sim_data, norm_comp_keys, dev_comp_keys, spatial_dims
        )
        sens_array = SensorsPoint(
            sensor_data, strain_field, descriptor
        )

        return sens_array

    @staticmethod
    def line_from_endpoints(
        sim_data: SimData,
        point_start: tuple[float, float, float] | np.ndarray,
        point_end: tuple[float, float, float] | np.ndarray,
        comp_key: str,
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        integ_rule=None,
        kernel=None,
        descriptor: SensorDescriptor | None = None,
        integration_mode: EIntegrationMode = EIntegrationMode.AVERAGE,
    ) -> SensorsSpatial:
        """Assembles a 1D line sensor between two 3D endpoints."""
        p1 = np.array(point_start, dtype=float).ravel()
        p2 = np.array(point_end, dtype=float).ravel()

        diff = p2 - p1
        length = float(np.linalg.norm(diff))
        center = 0.5 * (p1 + p2)

        rot = orient_from_direction(diff)
        sensor_data = SensorData(
            positions=center.reshape(1, 3),
            sample_times=sample_times,
            angles=(rot,),
        )
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        window = SpatialWindowLine(
            length=length,
            axis=(1.0, 0.0, 0.0),
            integ_rule=integ_rule,
            kernel=kernel,
        )
        return SensorsSpatial(
            sensor_data=sensor_data,
            field=field,
            spatial_window=window,
            descriptor=descriptor,
            integration_mode=integration_mode,
        )

    @staticmethod
    def line(
        sim_data: SimData,
        sensor_data: SensorData,
        comp_key: str,
        length: float,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        spatial_dims: EDim = EDim.THREED,
        integ_rule=None,
        kernel=None,
        descriptor: SensorDescriptor | None = None,
        integration_mode: EIntegrationMode = EIntegrationMode.AVERAGE,
    ) -> SensorsSpatial:
        """Assembles a 1D line sensor array."""
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        window = SpatialWindowLine(
            length=length, axis=axis, integ_rule=integ_rule, kernel=kernel
        )
        return SensorsSpatial(
            sensor_data=sensor_data,
            field=field,
            spatial_window=window,
            descriptor=descriptor,
            integration_mode=integration_mode,
        )

    @staticmethod
    def area(
        sim_data: SimData,
        sensor_data: SensorData,
        comp_key: str,
        length_x: float,
        length_y: float,
        spatial_dims: EDim = EDim.THREED,
        integ_rule=None,
        kernel=None,
        descriptor: SensorDescriptor | None = None,
        integration_mode: EIntegrationMode = EIntegrationMode.AVERAGE,
    ) -> SensorsSpatial:
        """Assembles a 2D rectangular area sensor array."""
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        window = SpatialWindowRectangle(
            length_x=length_x,
            length_y=length_y,
            integ_rule=integ_rule,
            kernel=kernel,
        )
        return SensorsSpatial(
            sensor_data=sensor_data,
            field=field,
            spatial_window=window,
            descriptor=descriptor,
            integration_mode=integration_mode,
        )

    @staticmethod
    def volume(
        sim_data: SimData,
        sensor_data: SensorData,
        comp_key: str,
        length_x: float,
        length_y: float,
        length_z: float,
        spatial_dims: EDim = EDim.THREED,
        integ_rule=None,
        kernel=None,
        descriptor: SensorDescriptor | None = None,
        integration_mode: EIntegrationMode = EIntegrationMode.AVERAGE,
    ) -> SensorsSpatial:
        """Assembles a 3D volume sensor array."""
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        window = SpatialWindowBox(
            length_x=length_x,
            length_y=length_y,
            length_z=length_z,
            integ_rule=integ_rule,
            kernel=kernel,
        )
        return SensorsSpatial(
            sensor_data=sensor_data,
            field=field,
            spatial_window=window,
            descriptor=descriptor,
            integration_mode=integration_mode,
        )
