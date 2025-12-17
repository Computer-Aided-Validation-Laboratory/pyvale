# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import enum
from abc import ABC, abstractmethod
import numpy as np
from pyvale.sensorsim.sensordata import SensorData


class EErrType(enum.Enum):
    """Enumeration defining the error type for separation of error types for
    later analysis.

    EErrType.SYSTEMATIC:
        Also known as an epistemic error and is due to a lack of
        knowledge. Common examples include spatial or temporal averaging,
        digitisation / round off error and calibration errors.

    EErrType.RANDOM:
        Also known as aleatory error and is generally a result of sensor
        noise.
    """
    SYSTEMATIC = enum.auto()
    RANDOM = enum.auto()


class EErrDep(enum.Enum):
    """Enumeration defining error dependence.

    EErrDep.INDEPENDENT:
        Errors are calculated based on the ground truth sensor values
        interpolated from the input simulation.

    EErrDep.DEPENDENT:
        Errors are calculated based on the accumulated sensor reading due
        to all preceeding errors in the chain.
    """
    INDEPENDENT = enum.auto()
    DEPENDENT = enum.auto()


class IErrSimulator(ABC):
    """Interface (abstract base class) for sensor error simulation allowing a
    list of simulated errors (i.e. an error chain) to be constructed and
    executed in order.
    """

    @abstractmethod
    def get_error_type(self) -> EErrType:
        """Gets the error type enumeration as either random or systematic. 
        Random errors sample at every time step and systematic errors typically
        apply a bias that is constant over time. 

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """

    @abstractmethod
    def get_error_dep(self) -> EErrDep:
        """Gets the error dependence enumeration value. Independent errors are
        calculated based on the ground truth and ignore other errors in the 
        error chain. Dependent errors are calculated based on the accumulated
        measurement value at their place in the error chain.

        Returns
        -------
        EErrDep
            Enumeration definining INDEPENDENT or DEPENDENT error dependence.
        """

    @abstractmethod
    def set_error_dep(self, dependence: EErrDep) -> None:
        """Sets the error dependence for errors that support changing the 
        dependence. Independent errors are calculated based on the ground truth 
        and ignore other errors in the error chain. Dependent errors are 
        calculated based on the accumulated measurement value at their place in 
        the error chain.

        Parameters
        ----------
        dependence : EErrDep
            Enumeration definining INDEPENDENT or DEPENDENT error dependence.
        """

    @abstractmethod
    def reseed(self,seed: int | None = None) -> None:
        """Reseeds the random generators of the error simulator. Mainly used for
        multi-processed simulations which inherit the same seed as the main 
        process so need to be reseeded. If the error simulator does not have any
        random generators then this function implementation will be empty.

        Parameters
        ----------
        seed : int | None, optional
            Integer seed for the random number generator, by default None. If 
            None then the seed is generated using OS entropy (see numpy docs).
        """


    @abstractmethod
    def sim_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        """Creates the simulated error array based on the input error basis 
        array. The output error array will be the same shape as the input 
        error basis array.

        Parameters
        ----------
        err_basis : np.ndarray
            Used as the base array for calculating the returned error. If the 
            error is independent this will be the 'truth' array and if the error
            is dependent this will be the accumulated sensor measurement array 
            at this point in the error chain.
        sens_data : SensorData
            Sensor data object holding the current sensor state before applying
            this error calculation.

        Returns
        -------
        tuple[np.ndarray, SensorData]
            Tuple containing the error array from this calculator and a
            SensorData object with the current accumulated sensor state starting
            from the nominal state up to and including this error calculator in
            the error chain. Note that many errors do not modify the sensor data
            so the sensor data class is passed through this function unchanged.
        """




