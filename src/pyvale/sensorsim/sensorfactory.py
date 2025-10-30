# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np

import pyvale.mooseherder as mh

from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.sensordescriptor import (DescriptorFactory,
                                               SensorDescriptor) 
from pyvale.sensorsim.sensorarraypoint import SensorArrayPoint, SensorData
from pyvale.sensorsim.errorintegrator import ErrIntegrator
from pyvale.sensorsim.errorcalculator import IErrCalculator
from pyvale.sensorsim.errorsysindep import ErrSysUnifPercent
from pyvale.sensorsim.errorrand import ErrRandNormPercent
from pyvale.sensorsim.errorsysdep import (ErrSysDigitisation,
                                          ErrSysSaturation)

# TODO:
# - docstrings
# - more sensor models

class SensorFactory:
    """Namespace for static methods used to build common types of sensor arrays
    simplifying sensor array creation for users.
    """

    @staticmethod
    def scalar_no_errs(sim_data: mh.SimData,
                       sensor_data: SensorData,
                       elem_dims: int,
                       field_name: str,
                       descriptor: SensorDescriptor | None = None 
                       ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.scalar_descriptor()
                       
        s_field = FieldScalar(sim_data,field_name,elem_dims)
                       
        sens_array = SensorArrayPoint(sensor_data,
                                      s_field,
                                      descriptor)

        return sens_array
      
    @staticmethod
    def scalar_basic_errs(sim_data: mh.SimData,
                          sensor_data: SensorData,
                          elem_dims: int,
                          field_name: str,
                          descriptor: SensorDescriptor | None = None,
                          sys_err_pc: float = 1.0,
                          rand_err_pc: float = 1.0, 
                          ) -> SensorArrayPoint:
                          
        sens_array = SensorFactory.scalar_no_errs(sim_data,
                                                  sensor_data,
                                                  elem_dims,
                                                  field_name,
                                                  descriptor)
                                                  
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
                                
        sens_array.set_error_chain(err_chain)
        return sens_array

    @staticmethod
    def thermocouples_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              elem_dims: int,
                              field_name: str = "temperature", 
                              ) -> SensorArrayPoint:

        descriptor = DescriptorFactory.temperature_descriptor()
        t_field = FieldScalar(sim_data,field_name,elem_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      t_field,
                                      descriptor)

        return sens_array

    @staticmethod
    def thermocouples_basic_errs(sim_data: mh.SimData,
                                 sensor_data: SensorData,
                                 elem_dims: int,
                                 field_name: str = "temperature",
                                 sys_err_pc: float = 1.0,
                                 rand_err_pc: float = 1.0,                                
                                 ) -> SensorArrayPoint:

        sens_array = SensorFactory.thermocouples_no_errs(sim_data,
                                                         sensor_data,
                                                         elem_dims,
                                                         field_name)
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
        # Normal thermcouple amp = 5mV / K
        # err_chain.append(ErrSysDigitisation(bits_per_unit=2**16/1000))
        # err_chain.append(ErrSysSaturation(meas_min=0.0,meas_max=1000.0))

        sens_array.set_error_chain(err_chain)
        return sens_array


    @staticmethod
    def vector_no_errs(sim_data: mh.SimData,
                       sensor_data: SensorData,
                       elem_dims: int,
                       field_name: str,
                       field_comps: tuple[str,...],
                       descriptor: SensorDescriptor | None = None,
                       ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.vector_descriptor()
    
        disp_field = FieldVector(sim_data,
                                 field_name,
                                 field_comps,
                                 elem_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array

    @staticmethod
    def vector_basic_errs(sim_data: mh.SimData,
                          sensor_data: SensorData,
                          elem_dims: int,
                          field_name: str,
                          field_comps: tuple[str,...],
                          descriptor: SensorDescriptor | None = None,
                          sys_err_pc: float = 1.0,
                          rand_err_pc: float = 1.0,
                          ) -> SensorArrayPoint:

        sens_array = SensorFactory.vector_no_errs(sim_data,
                                                  sensor_data,
                                                  elem_dims,
                                                  field_name,
                                                  field_comps,
                                                  descriptor)
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
        sens_array.set_error_chain(err_chain)
        return sens_array
        
    @staticmethod
    def disp_no_errs(sim_data: mh.SimData,
                     sensor_data: SensorData,
                     elem_dims: int,
                     field_name: str,
                     field_comps: tuple[str,...],
                     ) -> SensorArrayPoint:

        descriptor = DescriptorFactory.displacement_descriptor()
        disp_field = FieldVector(sim_data,
                                 field_name,
                                 field_comps,
                                 elem_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array


    @staticmethod
    def disp_basic_errs(sim_data: mh.SimData,
                        sensor_data: SensorData,
                        elem_dims: int,
                        field_name: str,
                        field_comps: tuple[str,...],
                        sys_err_pc: float = 1.0,
                        rand_err_pc: float = 1.0,
                        ) -> SensorArrayPoint:

        sens_array = SensorFactory.disp_sensors_no_errs(sim_data,
                                                        sensor_data,
                                                        elem_dims,
                                                        field_name,
                                                        field_comps)
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
        sens_array.set_error_chain(err_chain)
        return sens_array


    @staticmethod
    def tensor_no_errs(sim_data: mh.SimData,
                       sensor_data: SensorData,
                       elem_dims: int,
                       field_name: str,
                       norm_comps: tuple[str,...],
                       dev_comps: tuple[str,...],
                       descriptor: SensorDescriptor | None = None,
                       ) -> SensorArrayPoint:

        if descriptor is None:
            descriptor = DescriptorFactory.tensor_descriptor(elem_dims)

        strain_field = FieldTensor(sim_data,
                                   field_name,
                                   norm_comps,
                                   dev_comps,
                                   elem_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      strain_field,
                                      descriptor)

        return sens_array

    @staticmethod
    def tensor_basic_errs(sim_data: mh.SimData,
                          sensor_data: SensorData,
                          elem_dims: int,
                          field_name: str,
                          norm_comps: tuple[str,...],
                          dev_comps: tuple[str,...],
                          descriptor: SensorDescriptor | None = None,
                          ) -> SensorArrayPoint:

        sens_array = SensorFactory.tensor_no_errs(sim_data,
                                                  sensor_data,
                                                  elem_dims,
                                                  field_name,
                                                  norm_comps,
                                                  dev_comps,
                                                  descriptor)
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
        sens_array.set_error_chain(err_chain)

        return sens_array



    @staticmethod
    def strain_no_errs(sim_data: mh.SimData,
                       sensor_data: SensorData,
                       elem_dims: int,
                       field_name: str,
                       norm_comps: tuple[str,...],
                       dev_comps: tuple[str,...]
                       ) -> SensorArrayPoint:

        descriptor = DescriptorFactory.strain_descriptor(elem_dims)
        strain_field = FieldTensor(sim_data,
                                   field_name,
                                   norm_comps,
                                   dev_comps,
                                   elem_dims)
        sens_array = SensorArrayPoint(sensor_data,
                                      strain_field,
                                      descriptor)

        return sens_array


    @staticmethod
    def strain_basic_errs(sim_data: mh.SimData,
                          sensor_data: SensorData,
                          elem_dims: int,
                          field_name: str,
                          norm_comps: tuple[str,...],
                          dev_comps: tuple[str,...],
                          sys_err_pc: float = 1.0,
                          rand_err_pc: float = 1.0,
                          ) -> SensorArrayPoint:

        sens_array = SensorFactory.strain_no_errs(sim_data,
                                                  sensor_data,
                                                  elem_dims,
                                                  field_name,
                                                  norm_comps,
                                                  dev_comps)
        err_chain = basic_err_chain(sys_err_pc=sys_err_pc,
                                    rand_err_pc=rand_err_pc)
        sens_array.set_error_chain(err_chain)

        return sens_array


def basic_err_chain(sys_err_pc: float = 1.0,
                    rand_err_pc: float = 1.0) -> list[IErrCalculator]:
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
    list[IErrCalculator]
        A basic error chain with a uniform percentage systematic error and
        a normal percentage random error.
    """
    err_chain = []
    err_chain.append(ErrSysUnifPercent(-sys_err_pc,sys_err_pc))
    err_chain.append(ErrRandNormPercent(rand_err_pc))
    return err_chain
