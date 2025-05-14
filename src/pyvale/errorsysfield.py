# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import copy
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.field import IField
from pyvale.fieldsampler import sample_field_with_sensor_data
from pyvale.sensordata import SensorData
from pyvale.integratortype import EIntSpatialType
from pyvale.errorcalculator import (IErrCalculator,
                                    EErrType,
                                    EErrDep)
from pyvale.errordriftcalc import IDriftCalculator
from pyvale.generatorsrandom import IGenRandom

# TODO:
# - Implement different perturbed sampling times for each sensor or allow all
#   to lock to the same time step as it works now.
# - Need to check that we perform field rotations correctly for sensor angles.
#   - This needs to be updated to take rotation objects for offsets and to build
#     and compose rotations


@dataclass(slots=True)
class ErrFieldData:
    """Dataclass for controlling sensor parameter perturbations for field based
    systematic errors (i.e. errors that require interpolation of the physical
    field).
    """

    pos_offset_xyz: np.ndarray | None = None
    """Array of offsets to apply to the sensor positions for error calculation.
    shape=(num_sensors,3) where the columns represent the X, Y and Z offsets in
    simulation world coordinates. If None then no position offset is applied.
    """

    ang_offset_zyx: np.ndarray | None = None
    """Array of offsets to apply to the sensor angles for error calculation.
    shape=(num_sensors,3) where the columns represent rotations about offsets
    about the Z, Y and X axis of the sensor in sensor local coordinates. If None
    then no angular offsets are applied.
    """

    time_offset: np.ndarray | None = None
    """Array of offsets to apply to the sampling times for all sensors. shape=(
    num_time_steps,). If None then no time offset is applied.
    """

    pos_rand_xyz: tuple[IGenRandom | None,
                        IGenRandom | None,
                        IGenRandom | None] = (None,None,None)
    """Tuple of random generators (implementations of `IGenRandom`
    interface) for perturbing the sensor positions. The generators perturb the
    X, Y and Z coordinates in order. If None then that axis is not randomly
    perturbed from the nominal sensor position. Note that the random generators
    should return position perturbations consistent with the simulation units.
    """

    ang_rand_zyx: tuple[IGenRandom | None,
                        IGenRandom | None,
                        IGenRandom | None] = (None,None,None)
    """Tuple of random generators (implementations of `IGenRandom`
    interface) for perturbing the sensor angles. The generators perturb
    rotations about the the Z, Y and X axis  in order. If None then that axis is
    not randomly perturbed from the nominal sensor position.
    """

    time_rand: IGenRandom | None = None
    """Random generator for perturbing sensor array sampling times for the
    purpose of calculating field based errors. If None then sensor sampling
    times will not be perturbed from the nominal times.
    """

    spatial_averager: EIntSpatialType | None = None
    """Type of spatial averaging to use for this sensor array for the purpose of
    calculating field based errors. If None then no spatial averaging is
    performed.
    """

    spatial_dims: np.ndarray | None = None
    """The spatial dimension of the sensor in its local X,Y,Z coordinates for
    the purpose of calculating field errors. Only used if spatial averager is
    specified above. shape=(3,)
    """

    # DEV FEATURE: locks the coordinate even if offsets and random generators
    # are specified. These allow individual sensors to be locked when we only
    # specify a random generator for each axis not each sensor.
    pos_lock_xyz: np.ndarray | None = None
    ang_lock_zyx: np.ndarray | None = None

    # TODO: implement drift for other dimensions, pos/angle
    time_drift: IDriftCalculator | None = None
    """Temporal drift calculation
    """


class ErrSysField(IErrCalculator):
    """Class for calculating field based systematic errors. Field based errors
    are errors that require interpolation or sampling of the simulated physical
    field such as perturbations of the sensor position or sampling time.

    All perturbations to the sensor parameters (positions, sample times, angles
    area averaging) are calculated first before performing a single
    interpolation with the perturbed sensor state.

    Implements the `IErrCalculator` interface.
    """
    __slots__ = ("_field","_sensor_data_perturbed","_field_err_data","_err_dep")

    def __init__(self,
                field: IField,
                field_err_data: ErrFieldData,
                err_dep: EErrDep = EErrDep.INDEPENDENT) -> None:
        """
        Parameters
        ----------
        field : IField
            The physical field to interpolate which will be an implementation of
            the `IField` interface. This will be a `FieldScalar`, `FieldVector`
            or `FieldTensor` object.
        field_err_data : ErrFieldData
            Dataclass specifying which sensor array parameters will be perturbed
            and how they will be perturbed. See the `ErrFieldData` class for
            more detail
        err_dep : EErrDep, optional
            Error calculation dependence, by default EErrDep.DEPENDENT.
        """
        self._field = field
        self._field_err_data = field_err_data
        self._err_dep = err_dep
        self._sensor_data_perturbed = SensorData()

    def get_error_dep(self) -> EErrDep:
        """Gets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Returns
        -------
        EErrDep
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        return self._err_dep

    def set_error_dep(self, dependence: EErrDep) -> None:
        """Sets the error dependence state for this error calculator. An
        independent error is calculated based on the input truth values as the
        error basis. A dependent error is calculated based on the accumulated
        sensor reading from all preceeding errors in the chain.

        Parameters
        ----------
        dependence : EErrDep
            Enumeration defining INDEPENDENT or DEPENDENT behaviour.
        """
        self._err_dep = dependence

    def get_error_type(self) -> EErrType:
        """Gets the error type.

        Returns
        -------
        EErrType
            Enumeration definining RANDOM or SYSTEMATIC error types.
        """
        return EErrType.SYSTEMATIC

    def get_perturbed_sensor_data(self) -> SensorData:

        return self._sensor_data_perturbed

    def calc_errs(self,
                  err_basis: np.ndarray,
                  sens_data: SensorData,
                  ) -> tuple[np.ndarray, SensorData]:
        """Calculates the error array based on the size of the input. First
        calculates the combined perturbed sensor state from all perturbations
        specified in the `ErrFieldData` object and then performs a single
        interpolation of the field to obtain the error array.

        Parameters
        ----------
        err_basis : np.ndarray
            Array of values with the same dimensions as the sensor measurement
            matrix.
        sens_data : SensorData
            The accumulated sensor state data for all errors prior to this one.

        Returns
        -------
        tuple[np.ndarray, SensorData]
            Tuple containing the calculated error array and pass through of the
            sensor data object as it is not modified by this class. The returned
            error array has the same shape as the input error basis.
        """
        self._sensor_data_perturbed = copy.deepcopy(sens_data)
        self._sensor_data_perturbed.spatial_averager = \
            self._field_err_data.spatial_averager
        self._sensor_data_perturbed.spatial_dims = \
            self._field_err_data.spatial_dims

        self._sensor_data_perturbed.positions = _perturb_sensor_positions(
            self._sensor_data_perturbed.positions,
            self._field_err_data.pos_offset_xyz,
            self._field_err_data.pos_rand_xyz,
            self._field_err_data.pos_lock_xyz,
        )

        self._sensor_data_perturbed.sample_times = _perturb_sample_times(
            self._field.get_time_steps(),
            self._sensor_data_perturbed.sample_times,
            self._field_err_data.time_offset,
            self._field_err_data.time_rand,
            self._field_err_data.time_drift,
        )

        self._sensor_data_perturbed.angles = _perturb_sensor_angles(
            sens_data.positions.shape[0],
            self._sensor_data_perturbed.angles,
            self._field_err_data.ang_offset_zyx,
            self._field_err_data.ang_rand_zyx,
            self._field_err_data.ang_lock_zyx,
        )

        sys_errs = sample_field_with_sensor_data(
            self._field,
            self._sensor_data_perturbed
        ) - err_basis

        return (sys_errs,self._sensor_data_perturbed)


def _perturb_sensor_positions(sens_pos_nominal: np.ndarray,
                             pos_offset_xyz: np.ndarray | None,
                             pos_rand_xyz: tuple[IGenRandom | None,
                                                 IGenRandom | None,
                                                 IGenRandom | None] | None,
                             pos_loc_xyz: np.ndarray | None,
                            ) -> np.ndarray:
    """Helper function for perturbing the sensor positions from their nominal
    positions based on the user specified offset and random generators for each
    axis.

    Parameters
    ----------
    sens_pos_nominal : np.ndarray
        Nominal sensor positions as an array with shape=(num_sensors,3) where
        the columns represent the position in the X, Y and Z axes.
    pos_offset_xyz : np.ndarray | None
        Offsets to apply to the sensor positions as an array with shape=
        (num_sensors,3) wherethe columns represent the position in the X, Y and
        Z axes. If None then no offset is applied.
    pos_rand_xyz : tuple[IGenRandom  |  None,
                         IGenRandom  |  None,
                         IGenRandom  |  None] | None
        Random generators for sensor position perturbations along the the X, Y
        and Z axes. If None then no perturbation is applied.
    pos_loc_xyz : np.ndarray | None
        Boolean mask with shape=(num_sensors,3), where the mask is true the
        coordinate is locked and will not perturb based on offset or rand above.

    Returns
    -------
    np.ndarray
        Array of perturbed sensors positions with shape=(num_sensors,3) where
        the columns represent the position in the X, Y and Z axes.
    """
    sens_pos_perturbed = np.copy(sens_pos_nominal)

    if pos_offset_xyz is not None:
        sens_pos_perturbed = sens_pos_perturbed + pos_offset_xyz

    if pos_rand_xyz is not None:
        for ii,rng in enumerate(pos_rand_xyz):
            if rng is not None:
                sens_pos_perturbed[:,ii] = sens_pos_perturbed[:,ii] + \
                    rng.generate(shape=sens_pos_perturbed.shape[0])

    if pos_loc_xyz is not None:
        sens_pos_perturbed[pos_loc_xyz] = sens_pos_nominal[pos_loc_xyz]

    return sens_pos_perturbed


def _perturb_sample_times(sim_time: np.ndarray,
                         time_nominal: np.ndarray | None,
                         time_offset: np.ndarray | None,
                         time_rand: IGenRandom | None,
                         time_drift: IDriftCalculator | None
                         ) -> np.ndarray | None:
    """Helper function for calculating perturbed sensor sampling times for the
    purpose of calculating field based systematic errors.

    Parameters
    ----------
    sim_time : np.ndarray
        Simulation time steps for the underlying physical field.
    time_nominal : np.ndarray | None
        Nominal sensor sampling times. If None then the simulation time steps
        are assumed to be the sampling times.
    time_offset : np.ndarray | None
        Array of time offsets to apply to all sensors. If None then no offsets
        are applied.
    time_rand : IGenRandom | None
        Random generator for perturbing the sampling times of all sensors. If
        None then no random perturbation of sampling times occurs.
    time_drift : IDriftCalculator | None
        Drift function for calculating temporal sampling drift. If None then no
        temporal drift is applied.

    Returns
    -------
    np.ndarray | None
        Array of perturbed sample times
    """
    if time_nominal is None:
        if (time_offset is not None
            or time_rand is not None
            or time_drift is not None):
            time_nominal = sim_time
        else:
            return None

    time_perturbed = np.copy(time_nominal)

    if time_offset is not None:
        time_perturbed = time_perturbed + time_offset
    if time_rand is not None:
        time_perturbed = time_perturbed + time_rand.generate(
            shape=time_nominal.shape)
    if time_drift is not None:
        time_perturbed = time_perturbed + time_drift.calc_drift(time_nominal)

    return time_perturbed


def _perturb_sensor_angles(n_sensors: int,
                          angles_nominal: tuple[Rotation,...] | None,
                          angle_offsets_zyx: np.ndarray | None,
                          rand_ang_zyx: tuple[IGenRandom | None,
                                              IGenRandom | None,
                                              IGenRandom | None] | None,
                          angle_loc_zyx: np.ndarray | None,
                          ) -> tuple[Rotation,...] | None:
    """Helper function for perturbing sensor angles for the purpose of
    calculating field based systematic errors.

    Parameters
    ----------
    n_sensors : int
        Number of sensors in the sensor array.
    angles_nominal : tuple[Rotation,...] | None
        The nominal angles of the sensors as a tuple of scipy Rotation objects.
        This tuple should have length equal to the number of sensors. If None
        then an initial orienation of [0,0,0] is assumed.
    angle_offsets_zyx : np.ndarray | None
        Angle offsets to apply to the sensor array as an array with shape=(
        num_sensors,3) where the columns are the rotations about Z, Y and X in
        degrees. If None then no offsets are applied.
    rand_ang_zyx : tuple[IGenRandom  |  None,
                         IGenRandom  |  None,
                         IGenRandom  |  None] | None
        Random generators for perturbing sensor angles about the Z, Y and X axis
        respectively. If None then no random perturbation to the sensor angle
        occurs.
    angle_loc_zyx : np.ndarray | None
        Boolean mask with shape=(num_sensors,3), where the mask is true the
        angle is locked and the sensor will not rotate about that axis despite
        the offset of rand generators above,

    Returns
    -------
    tuple[Rotation,...] | None
        Rotation object giving each sensors perturbed angle. If None then the
        no sensors have had their angles perturbed.
    """
    if angles_nominal is None:
        if angle_offsets_zyx is not None or rand_ang_zyx is not None:
            angles_nominal = n_sensors * \
                (Rotation.from_euler("zyx",[0,0,0], degrees=True),)
        else:
            return None

    angles_perturbed = [Rotation.from_euler("zyx",[0,0,0], degrees=True)] * \
        len(angles_nominal)
    for ii,rot_nom in enumerate(angles_nominal): # loop over sensors
        # NOTE: adding angles here might not be correct
        sensor_rot_angs = np.zeros((3,))

        if angle_offsets_zyx is not None:
            sensor_rot_angs = sensor_rot_angs + angle_offsets_zyx[ii,:]

        if rand_ang_zyx is not None:
            for jj,rand_ang in enumerate(rand_ang_zyx): # loop over components
                if rand_ang is not None:
                    sensor_rot_angs[jj] = sensor_rot_angs[jj] + \
                        rand_ang.generate(shape=1)

        if angle_loc_zyx is not None:
            # No rotation about locked axes using mask
            sensor_rot_angs[angle_loc_zyx[ii,:]] = 0.0

        sensor_rot = Rotation.from_euler("zyx",sensor_rot_angs, degrees=True)
        angles_perturbed[ii] = sensor_rot*rot_nom

    return tuple(angles_perturbed)