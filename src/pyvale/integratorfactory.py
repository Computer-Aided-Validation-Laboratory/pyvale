'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
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
    @staticmethod
    def rect_2d_1pt(field: IField,
                    sensor_data: SensorData,
                    )-> IIntegratorSpatial:

        int_pt_offsets = np.array([[0,0,0],])

        return Rectangle2D(field,
                           sensor_data,
                           int_pt_offsets)


    @staticmethod
    def rect_2d_4pt(field: IField,
                    sensor_data: SensorData,
                    )-> IIntegratorSpatial:

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
                    )-> IIntegratorSpatial:

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
                    )-> IIntegratorSpatial:

        gauss_pt_offsets = sensor_data.spatial_dims * 1/np.sqrt(3) * \
                                                np.array([[-1,-1,0],
                                                          [-1,1,0],
                                                          [1,-1,0],
                                                          [1,1,0]])

        gauss_weight_func = create_gauss_weights_2d_4pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)


    @staticmethod
    def quad_2d_9pt(field: IField,
                    sensor_data: SensorData,
                    )-> IIntegratorSpatial:

        gauss_pt_offsets = sensor_data.spatial_dims * \
                                np.array([[-np.sqrt(0.6),-np.sqrt(0.6),0],
                                          [-np.sqrt(0.6),np.sqrt(0.6),0],
                                          [np.sqrt(0.6),-np.sqrt(0.6),0],
                                          [np.sqrt(0.6),np.sqrt(0.6),0],
                                          [-np.sqrt(0.6),0,0],
                                          [0,-np.sqrt(0.6),0],
                                          [0,np.sqrt(0.6),0],
                                          [np.sqrt(0.6),0,0],
                                          [0,0,0]])

        gauss_weight_func = create_gauss_weights_2d_9pts

        return Quadrature2D(field,
                            sensor_data,
                            gauss_pt_offsets,
                            gauss_weight_func)


def build_spatial_averager(field: IField, sensor_data: SensorData,
                          ) -> IIntegratorSpatial | None:
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