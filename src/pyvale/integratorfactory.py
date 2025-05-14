# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from pyvale.field import IField
from pyvale.sensordata import SensorData
from pyvale.integratorspatial import IIntegratorSpatial
from pyvale.integratortype import EIntSpatialType
from pyvale.integratorrectangle import Rectangle2D
from pyvale.integratorquadrature import (Quadrature2D,
                                              create_gauss_weights_2d_4pts,
                                              create_gauss_weights_2d_9pts)

class IntegratorSpatialFactory:
    """Namespace for static methods used to build 2D spatial integrators. These
    integrators are used to simulate spatial averaging for sensors.
    """

    @staticmethod
    def rect_2d_1pt(field: IField,
                    sensor_data: SensorData,
                    )-> Rectangle2D:
        """Builds and returns a 2D rectangular spatial integrator with a single
        integration point.

        Parameters
        ----------
        field : IField
            Interface specifying the physical field that the integrator will
            sample from.
        sensor_data : SensorData
            Sensor data specifying the location of the sensors that will be
            used for the spatial integration

        Returns
        -------
        Rectangle2D
            Rectangular spatial integrator that can sample the specified
            physical field at specified locations
        """

        int_pt_offsets = np.array([[0,0,0],])

        return Rectangle2D(field,
                           sensor_data,
                           int_pt_offsets)


    @staticmethod
    def rect_2d_4pt(field: IField,
                    sensor_data: SensorData,
                    )-> Rectangle2D:
        """Builds and returns a 2D rectangular spatial integrator with four
        equally spaced integration points in a grid pattern.

        Parameters
        ----------
        field : IField
            Interface specifying the physical field that the integrator will
            sample from.
        sensor_data : SensorData
            Sensor data specifying the location of the sensors that will be
            used for the spatial integration

        Returns
        -------
        Rectangle2D
            Rectangular spatial integrator that can sample the specified
            physical field at specified locations
        """
        int_pt_offsets = sensor_data.spatial_dims * np.array([[-0.5,-0.5,0],
                                                              [-0.5,0.5,0],
                                                              [0.5,-0.5,0],
                                                              [0.5,0.5,0],])

        return Rectangle2D(field,
                           sensor_data,
                           int_pt_offsets)



    @staticmethod
    def rect_2d_9pt(field: IField,
                    sensor_data: SensorData,
                    )-> Rectangle2D:
        """Builds and returns a 2D rectangular spatial integrator with nine
        equally spaced integration points in a grid pattern.

        Parameters
        ----------
        field : IField
            Interface specifying the physical field that the integrator will
            sample from.
        sensor_data : SensorData
            Sensor data specifying the location of the sensors that will be
            used for the spatial integration

        Returns
        -------
        Rectangle2D
            Rectangular spatial integrator that can sample the specified
            physical field at specified locations
        """
        int_pt_offsets = sensor_data.spatial_dims * np.array([[-1/3,-1/3,0],
                                                              [-1/3,0,0],
                                                              [-1/3,1/3,0],
                                                              [0,-1/3,0],
                                                              [0,0,0],
                                                              [0,1/3,0],
                                                              [1/3,-1/3,0],
                                                              [1/3,0,0],
                                                              [1/3,1/3,0]])

        return Rectangle2D(field,
                           sensor_data,
                           int_pt_offsets)


    @staticmethod
    def quad_2d_4pt(field: IField,
                    sensor_data: SensorData,
                    )-> Quadrature2D:
        """Builds and returns a Gaussian quadrature spatal integrator based on
        a rectangular area with four integration points.

        Parameters
        ----------
        field : IField
            Interface specifying the physical field that the integrator will
            sample from.
        sensor_data : SensorData
            Sensor data specifying the location of the sensors that will be
            used for the spatial integration

        Returns
        -------
        Quadrature2D
            Quadrature integrator that can be used to sample the physical field
            at specified locations.
        """
        gauss_pt_offsets = (sensor_data.spatial_dims * 1/np.sqrt(3)
                                            * np.array([[-1,-1,0],
                                                        [-1,1,0],
                                                        [1,-1,0],
                                                        [1,1,0]]))

        gauss_weight_func = create_gauss_weights_2d_4pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)


    @staticmethod
    def quad_2d_9pt(field: IField,
                    sensor_data: SensorData,
                    )-> Quadrature2D:
        """Builds and returns a Gaussian quadrature spatal integrator based on
        a rectangular area with nine integration points.

        Parameters
        ----------
        field : IField
            Interface specifying the physical field that the integrator will
            sample from.
        sensor_data : SensorData
            Sensor data specifying the location of the sensors that will be
            used for the spatial integration

        Returns
        -------
        Quadrature2D
            Quadrature integrator that can be used to sample the physical field
            at specified locations.
        """
        gauss_pt_offsets = (sensor_data.spatial_dims
                            * np.array([[-np.sqrt(0.6),-np.sqrt(0.6),0],
                                        [-np.sqrt(0.6),np.sqrt(0.6),0],
                                        [np.sqrt(0.6),-np.sqrt(0.6),0],
                                        [np.sqrt(0.6),np.sqrt(0.6),0],
                                        [-np.sqrt(0.6),0,0],
                                        [0,-np.sqrt(0.6),0],
                                        [0,np.sqrt(0.6),0],
                                        [np.sqrt(0.6),0,0],
                                        [0,0,0]]))

        gauss_weight_func = create_gauss_weights_2d_9pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)


def build_spatial_averager(field: IField, sensor_data: SensorData,
                          ) -> IIntegratorSpatial | None:
    """Helper function to build a spatial integrator based on the
    "EIntSpatialType" enumeration in the SensorData object. Separates the
    spatial integration object from the SensorData object.

    Parameters
    ----------
    field : IField
        Physical field that will be sampled by the spatial averager.
    sensor_data : SensorData
        Sensor data containing the type of spatial integrator to build in the
        `sensor_data.spatial_averager` as an `EIntSpatialType` enumeration.

    Returns
    -------
    IIntegratorSpatial | None
        The spatial averager that will sample the physical field at the
        specified sensor locations. If `sensor_data.spatial_averager` or
        `sensor_data.spatial_dims` are None then it is not possible to build a
        spatial integrator and None is returned.
    """
    if sensor_data.spatial_averager is None or sensor_data.spatial_dims is None:
        return None

    if sensor_data.spatial_averager == EIntSpatialType.RECT1PT:
        return IntegratorSpatialFactory.rect_2d_1pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == EIntSpatialType.RECT4PT:
        return IntegratorSpatialFactory.rect_2d_4pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == EIntSpatialType.RECT9PT:
        return IntegratorSpatialFactory.rect_2d_9pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == EIntSpatialType.QUAD4PT:
        return IntegratorSpatialFactory.quad_2d_4pt(field,
                                                    sensor_data)
    elif sensor_data.spatial_averager == EIntSpatialType.QUAD9PT:
        return IntegratorSpatialFactory.quad_2d_9pt(field,
                                                    sensor_data)
    else:
        return None