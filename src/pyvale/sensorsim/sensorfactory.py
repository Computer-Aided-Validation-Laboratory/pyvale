# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains methods that assemble common sensor array configurations
without the user needing to configure all the sub-components themselves.
"""

import numpy as np

import pyvale.mooseherder as mh

from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.sensordescriptor import (DescriptorFactory,
                                               SensorDescriptor) 
from pyvale.sensorsim.sensorarraypoint import SensorArrayPoint, SensorData
from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.errorsimulator import IErrSimulator
from pyvale.sensorsim.errorsysindep import ErrSysUnifPercent
from pyvale.sensorsim.errorrand import ErrRandNormPercent
from pyvale.sensorsim.errorsysdep import (ErrSysDigitisation,
                                          ErrSysSaturation)
from pyvale.sensorsim.enums import EDim

# TODO:
# - docstrings

class SensorFactory:
    @staticmethod
    def scalar(sim_data: mh.SimData,
               sensor_data: SensorData,
               comp_key: str,
               spatial_dims: EDim,
               descriptor: SensorDescriptor | None = None 
               ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.scalar()
                       
        s_field = FieldScalar(sim_data,comp_key,spatial_dims)
                       
        sens_array = SensorArrayPoint(sensor_data,
                                      s_field,
                                      descriptor)
        return sens_array

    @staticmethod
    def vector(sim_data: mh.SimData,
               sensor_data: SensorData,
               comp_keys: tuple[str,...],
               spatial_dims: EDim,
               descriptor: SensorDescriptor | None = None,
               ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.vector()

        disp_field = FieldVector(sim_data,
                                 comp_keys,
                                 spatial_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array

    @staticmethod
    def tensor(sim_data: mh.SimData,
               sensor_data: SensorData,
               norm_comp_keys: tuple[str,...],
               dev_comp_keys: tuple[str,...],
               spatial_dims: EDim,
               descriptor: SensorDescriptor | None = None,
               ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.tensor(spatial_dims)

        strain_field = FieldTensor(sim_data,
                                   norm_comp_keys,
                                   dev_comp_keys,
                                   spatial_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      strain_field,
                                      descriptor)

        return sens_array
          
    # @staticmethod
    # def scalar_basic_errs(sim_data: mh.SimData,
    #                       sensor_data: SensorData,
    #                       comp_key: str,
    #                       spatial_dims: EDim,
    #                       descriptor: SensorDescriptor | None = None,
    #                       sys_err_pc: float = 1.0,
    #                       rand_err_pc: float = 1.0, 
    #                       ) -> SensorArrayPoint:
    #                       
    #     sens_array = SensorFactory.scalar(sim_data,
    #                                               sensor_data,
    #                                               comp_key,
    #                                               spatial_dims,
    #                                               descriptor)
    #                                               
    #     err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
    #                                 rand_err_pc=rand_err_pc)
    #                             
    #     sens_array.set_error_chain(err_chain)
    #     return sens_array

#     @staticmethod
#     def vector_basic_errs(sim_data: mh.SimData,
#                           sensor_data: SensorData,
#                           comp_keys: tuple[str,...],
#                           spatial_dims: EDim,
#                           descriptor: SensorDescriptor | None = None,
#                           sys_err_pc: float = 1.0,
#                           rand_err_pc: float = 1.0,
#                           ) -> SensorArrayPoint:
# 
#         sens_array = SensorFactory.vector(sim_data,
#                                                   sensor_data,
#                                                   comp_keys,
#                                                   spatial_dims,
#                                                   descriptor)
#         err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
#                                     rand_err_pc=rand_err_pc)
#         sens_array.set_error_chain(err_chain)
#         return sens_array

#     @staticmethod
#     def tensor_basic_errs(sim_data: mh.SimData,
#                           sensor_data: SensorData,
#                           norm_comp_keys: tuple[str,...],
#                           dev_comp_keys: tuple[str,...],
#                           spatial_dims: EDim,
#                           descriptor: SensorDescriptor | None = None,
#                           ) -> SensorArrayPoint:
# 
#         sens_array = SensorFactory.tensor(sim_data,
#                                                   sensor_data,
#                                                   norm_comp_keys,
#                                                   dev_comp_keys,
#                                                   spatial_dims,
#                                                   descriptor)
#         err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
#                                     rand_err_pc=rand_err_pc)
#         sens_array.set_error_chain(err_chain)
# 
#         return sens_array



def basic_err_chain(sys_err_pc: float = 1.0,
                    rand_err_pc: float = 1.0) -> list[IErrSimulator]:
    """Builds a basic error chain with uniform percentage systematic error
    calculator and a percentage normal random error calculator.

    Parameters
    ----------
    sys_err_pc : float, optional
        Percentage systematic error, by default 1.0.
    rand_err_pc : float, optional
        Percentage random error, by default 1.0.

    Returns
    -------
    list[IErrSimulator]
        A basic error chain with a uniform percentage systematic error and
        a normal percentage random error.
    """
    err_chain = []
    err_chain.append(ErrSysUnifPercent(-sys_err_pc,sys_err_pc))
    err_chain.append(ErrRandNormPercent(rand_err_pc))
    return err_chain
