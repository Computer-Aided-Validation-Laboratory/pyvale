# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np

import mooseherder as mh

from pyvale.fieldscalar import FieldScalar
from pyvale.fieldvector import FieldVector
from pyvale.fieldtensor import FieldTensor
from pyvale.sensordescriptor import SensorDescriptorFactory
from pyvale.sensorarraypoint import SensorArrayPoint, SensorData
from pyvale.errorintegrator import ErrIntegrator
from pyvale.errorsysindep import ErrSysUnifPercent
from pyvale.errorrand import ErrRandNormPercent
from pyvale.errorsysdep import (ErrSysDigitisation,
                                     ErrSysSaturation)

# TODO:
# - docstrings
# - more sensor models

class SensorArrayFactory:
    """Namespace for static methods used to build common types of sensor arrays
    simplifying sensor array creation for users.
    """

    @staticmethod
    def thermocouples_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              elem_dims: int,
                              field_name: str = "temperature",
                              ) -> SensorArrayPoint:

        descriptor = SensorDescriptorFactory.temperature_descriptor()

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
                                 errs_pc: float = 1.0
                                 ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.thermocouples_no_errs(sim_data,
                                                              sensor_data,
                                                              elem_dims,
                                                              field_name)

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       sys_err_pc=errs_pc,
                                       rand_err_pc=errs_pc)

        # Normal thermcouple amp = 5mV / K
        err_int._err_chain.append(ErrSysDigitisation(bits_per_unit=2**16/1000))
        err_int._err_chain.append(ErrSysSaturation(meas_min=0.0,meas_max=1000.0))

        sens_array.set_error_integrator(err_int)
        return sens_array



    @staticmethod
    def disp_sensors_no_errs(sim_data: mh.SimData,
                             sensor_data: SensorData,
                             elem_dims: int,
                             field_name: str,
                             field_comps: tuple[str,...],
                             ) -> SensorArrayPoint:

        descriptor = SensorDescriptorFactory.displacement_descriptor()

        disp_field = FieldVector(sim_data,
                                 field_name,
                                 field_comps,
                                 elem_dims)

        sens_array = SensorArrayPoint(sensor_data,
                                      disp_field,
                                      descriptor)
        return sens_array


    @staticmethod
    def disp_sensors_basic_errs(sim_data: mh.SimData,
                                sensor_data: SensorData,
                                elem_dims: int,
                                field_name: str,
                                field_comps: tuple[str,...],
                                errs_pc: float = 1.0,
                                ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.disp_sensors_no_errs(sim_data,
                                                             sensor_data,
                                                             elem_dims,
                                                             field_name,
                                                             field_comps)
        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       sys_err_pc=errs_pc,
                                       rand_err_pc=errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array

    @staticmethod
    def strain_gauges_no_errs(sim_data: mh.SimData,
                              sensor_data: SensorData,
                              elem_dims: int,
                              field_name: str,
                              norm_comps: tuple[str,...],
                              dev_comps: tuple[str,...]
                              ) -> SensorArrayPoint:
        descriptor = SensorDescriptorFactory.strain_descriptor(elem_dims)

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
    def strain_gauges_basic_errs(sim_data: mh.SimData,
                                sensor_data: SensorData,
                                elem_dims: int,
                                field_name: str,
                                norm_comps: tuple[str,...],
                                dev_comps: tuple[str,...],
                                errs_pc: float = 1.0
                                ) -> SensorArrayPoint:

        sens_array = SensorArrayFactory.strain_gauges_no_errs(sim_data,
                                                              sensor_data,
                                                              elem_dims,
                                                              field_name,
                                                              norm_comps,
                                                              dev_comps)

        err_int = basic_err_integrator(sens_array.get_measurement_shape(),
                                       sensor_data,
                                       sys_err_pc=errs_pc,
                                       rand_err_pc=errs_pc)
        sens_array.set_error_integrator(err_int)

        return sens_array


def basic_err_integrator(meas_shape: np.ndarray,
                         sensor_data: SensorData,
                         sys_err_pc: float = 1.0,
                         rand_err_pc: float = 1.0) -> ErrIntegrator:
    """Builds a basic error integrator with uniform percentage systematic error
    calculator and a percentage normal random error calculator.

    Parameters
    ----------
    meas_shape : np.ndarray
        Shape of the measurement array which is (num_sensors,
        num_field_components,num_time_steps)
    sensor_data : SensorData
        Sensor array parameters for feeding through the error chain.
    sys_err_pc : float, optional
        Percentage systematic error, by default 1.0.
    rand_err_pc : float, optional
        Percentage random error, by default 1.0.

    Returns
    -------
    ErrIntegrator
        A basic error integrator with a uniform percentage systematic error and
        a normal percentage random error.
    """
    err_chain = []
    err_chain.append(ErrSysUnifPercent(-sys_err_pc,sys_err_pc))
    err_chain.append(ErrRandNormPercent(rand_err_pc))
    err_int = ErrIntegrator(err_chain,sensor_data,meas_shape)
    return err_int