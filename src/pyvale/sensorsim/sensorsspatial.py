# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unified spatio-temporal integration sensor array implementing ISensorArray.
"""

import numpy as np
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.enums import EIntegrationMode
from pyvale.sensorsim.errorintegrator import ErrIntegrator, ErrIntOpts
from pyvale.sensorsim.errorgraph import ErrGraph
from pyvale.sensorsim.errorsimulator import IErrSimulator
from pyvale.sensorsim.spatialwindows import (
    ISpatialWindow,
    SpatialWindowPoint,
)
from pyvale.sensorsim.temporalwindows import (
    ITemporalWindow,
    TemporalWindowInstant,
)


class SensorsSpatial(ISensorArray):
    """Unified sensor array measuring physical fields with finite spatial
    support windows (0D point, 1D line, 2D area, 3D volume) and finite temporal
    response windows (exposure times, response lag).

    Directly implements the `ISensorArray` interface.
    """

    __slots__ = (
        "_field",
        "_descriptor",
        "_sensor_data",
        "_spatial_window",
        "_temporal_window",
        "_integration_mode",
        "_truth",
        "_measurements",
        "_error_integrator",
    )

    def __init__(
        self,
        sensor_data: SensorData,
        field: IField,
        spatial_window: ISpatialWindow | None = None,
        temporal_window: ITemporalWindow | None = None,
        descriptor: SensorDescriptor | None = None,
        integration_mode: EIntegrationMode = EIntegrationMode.AVERAGE,
    ) -> None:
        """
        Parameters
        ----------
        sensor_data : SensorData
            Sensor anchor positions, sample times, and 3D rotations.
        field : IField
            Physical simulated field (FieldScalar, FieldVector, or FieldTensor).
        spatial_window : ISpatialWindow | None, optional
            Spatial support geometry and sensitivity kernel, by default
            SpatialWindowPoint().
        temporal_window : ITemporalWindow | None, optional
            Temporal integration window and kernel, by default
            TemporalWindowInstant().
        descriptor : SensorDescriptor | None, optional
            Descriptor for visualisations and metadata, by default None.
        integration_mode : EIntegrationMode, optional
            Integration mode (AVERAGE for intensive variables, ACCUMULATE
            for extensive variables), by default AVERAGE.
        """
        self._sensor_data = sensor_data
        self._field = field

        if spatial_window is None:
            spatial_window = SpatialWindowPoint()
        self._spatial_window = spatial_window

        if temporal_window is None:
            temporal_window = TemporalWindowInstant()
        self._temporal_window = temporal_window

        if descriptor is None:
            descriptor = SensorDescriptor()
        self._descriptor = descriptor

        self._integration_mode = integration_mode
        self._error_integrator = None
        self._truth = None
        self._measurements = None

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def get_spatial_window(self) -> ISpatialWindow:
        return self._spatial_window

    def get_temporal_window(self) -> ITemporalWindow:
        return self._temporal_window

    def get_integration_mode(self) -> EIntegrationMode:
        return self._integration_mode

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data

    def get_field(self) -> IField:
        return self._field

    def get_sample_times(self) -> np.ndarray:
        if self._sensor_data.sample_times is None:
            return self._field.get_time_steps()
        return self._sensor_data.sample_times

    def get_measurement_shape(self) -> tuple[int, int, int]:
        n_sensors = self._sensor_data.positions.shape[0]
        n_comps = len(self._field.get_all_components())
        n_times = self.get_sample_times().shape[0]
        return (n_sensors, n_comps, n_times)

    def calc_truth(self) -> np.ndarray:
        """Calculates ground truth sensor values by integrating the physical
        field over the spatio-temporal window.
        """
        local_pts, spat_wts = (
            self._spatial_window.get_local_points_and_weights(
                mode=self._integration_mode
            )
        )
        tau_offsets, temp_wts = (
            self._temporal_window.get_sample_offsets_and_weights(
                mode=self._integration_mode
            )
        )

        n_sensors = self._sensor_data.positions.shape[0]
        n_spat_pts = local_pts.shape[0]
        n_comps = len(self._field.get_all_components())
        nominal_times = self.get_sample_times()
        n_nom_times = nominal_times.shape[0]
        n_tau = tau_offsets.shape[0]

        # 1. Transform spatial points to global simulation coordinates
        glob_pts = self._spatial_window.to_global_points(
            self._sensor_data.positions, self._sensor_data.angles
        )
        flat_pts = glob_pts.reshape(-1, 3)

        # 2. Build temporal evaluation points
        # Grid of (n_nom_times, n_tau) -> flat array
        time_grid = nominal_times[:, np.newaxis] + tau_offsets[np.newaxis, :]
        flat_times = time_grid.ravel()

        # 3. Handle rotation angles for vector and tensor fields
        angles = None
        if self._sensor_data.angles is not None:
            if len(self._sensor_data.angles) == 1:
                angles = self._sensor_data.angles
            elif len(self._sensor_data.angles) == n_sensors:
                # Replicate each sensor rotation for spatial quadrature points
                expanded_angles = []
                for rot_s in self._sensor_data.angles:
                    for _ in range(n_spat_pts):
                        expanded_angles.append(rot_s)
                angles = tuple(expanded_angles)

        # 4. Sample the field at all points and times
        sampled_raw = self._field.sample_field(
            points=flat_pts, times=flat_times, angles=angles
        )

        # 5. Reshape and contract over spatio-temporal weights
        reshaped = sampled_raw.reshape(
            n_sensors, n_spat_pts, n_comps, n_nom_times, n_tau
        )

        spat_contracted = np.einsum("sqctm,q->sctm", reshaped, spat_wts)
        self._truth = np.einsum("sctm,m->sct", spat_contracted, temp_wts)
        return self._truth

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        return self._truth

    def get_error_integrator(self) -> ErrIntegrator | ErrGraph | None:
        return self._error_integrator

    def get_sensor_data_perturbed(self) -> SensorData | None:
        if self._error_integrator is not None:
            if hasattr(self._error_integrator, "get_sensor_data_perturbed"):
                return self._error_integrator.get_sensor_data_perturbed()
        return None

    def set_error_graph(self, error_graph: ErrGraph | None) -> None:
        self._error_integrator = error_graph

    def set_error_chain(
        self,
        err_chain: (
            IErrSimulator
            | list[IErrSimulator]
            | tuple[IErrSimulator, ...]
            | ErrGraph
            | None
        ),
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        """Sets the error chain or graph that will be used to calculate sensor
        measurement errors when `sim_measurements()` is called.

        Parameters
        ----------
        err_chain : IErrSimulator | list | tuple | ErrGraph | None
            Single error model, chain of user defined error models, or ErrGraph.
            Set to None to remove error calculation.
        err_int_opts : ErrIntOpts | None, optional
            Options for virtual sensor error integration, by default None.
        """
        if err_chain is None:
            self._error_integrator = None
            return None

        if isinstance(err_chain, ErrGraph):
            self._error_integrator = err_chain
            return None

        if not isinstance(err_chain, (list, tuple)):
            err_chain = [err_chain]

        if err_int_opts is None:
            err_int_opts = ErrIntOpts()

        self._error_integrator = ErrIntegrator(
            err_chain,
            self._sensor_data,
            self.get_measurement_shape(),
            err_int_opts,
        )

    def set_error_model(
        self,
        err_model: (
            IErrSimulator
            | list[IErrSimulator]
            | tuple[IErrSimulator, ...]
            | ErrGraph
            | None
        ),
        err_int_opts: ErrIntOpts | None = None,
    ) -> None:
        """Convenience method to set an error model, chain, or graph."""
        self.set_error_chain(err_model, err_int_opts=err_int_opts)

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_systematic()

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_random()

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_total()

    def calc_errors(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        if isinstance(self._error_integrator, ErrGraph):
            return self._error_integrator.calc_errors_from_graph(
                self.get_truth()
            )
        return self._error_integrator.calc_errors_from_chain(
            self.get_truth()
        )

    def sim_measurements(self) -> np.ndarray:
        truth = self.get_truth()
        if self._error_integrator is None:
            self._measurements = truth
            return self._measurements

        tot_err = self.calc_errors()
        if tot_err is not None:
            self._measurements = truth + tot_err
        else:
            self._measurements = truth
        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()
        return self._measurements


# Backward compatibility aliases
SensorsPoint = SensorsSpatial
SensorsLine = SensorsSpatial
SensorsArea = SensorsSpatial
SensorsVolume = SensorsSpatial
