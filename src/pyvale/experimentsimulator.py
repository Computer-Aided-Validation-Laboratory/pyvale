# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module is used for performing Monte-Carlo virtual experiments over a series
of input simulation cases and sensor arrays.
"""

from dataclasses import dataclass
import numpy as np
import mooseherder as mh
from pyvale.sensorarray import ISensorArray


@dataclass(slots=True)
class ExperimentStats:
    """Dataclass holding summary statistics for a series of simulated
    experiments produced using the experiment simulator. All summary statistics
    are calculated over the 'experiments' dimension of the measurements array so
    the arrays of statistics have the shape=(n_sims,n_sensors,n_field_comps,
    n_time_steps). Note that the n_sims dimension refers to the number of input
    multi-physics simulations (i.e. SimData objects) that the virtual
    experiments were performed over.
    """

    mean: np.ndarray | None = None
    """Mean of each sensors measurement for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps).
    """

    std: np.ndarray | None = None
    """Standard deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """

    max: np.ndarray | None = None
    """Maximum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    min: np.ndarray | None = None
    """Minmum of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    med: np.ndarray | None = None
    """Median  of the sensor measurements for the given field component and time
    step as an array with shape=(n_sims,n_sensors,n_field_comps,n_time_steps)
    """

    q25: np.ndarray | None = None
    """Lower 25% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """

    q75: np.ndarray | None = None
    """Upper 75% quantile of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    _field_comps, n_time_steps)
    """

    mad: np.ndarray | None = None
    """Median absolute deviation of the sensor measurements for the given field
    component and time step as an array with shape=(n_sims,n_sensors,
    n_field_comps, n_time_steps)
    """


class ExperimentSimulator:
    """An experiment simulator for running monte-carlo analysis by applying a
    list of sensor arrays to a list of simulations over a given number of user
    defined experiments. Calculates summary statistics for each sensor array
    applied to each simulation.
    """
    __slots__ = ("_sim_list","_sensor_arrays","_num_exp_per_sim","_exp_data",
                 "_exp_stats")

    def __init__(self,
                 sim_list: list[mh.SimData],
                 sensor_arrays: list[ISensorArray],
                 num_exp_per_sim: int
                 ) -> None:
        """
        Parameters
        ----------
        sim_list : list[mh.SimData]
            List of simulation data objects over which the virtual experiments
            will be performed.
        sensor_arrays : list[ISensorArray]
            The sensor arrays that will be applied to each simulation to
            generate the virtual experiment data.
        num_exp_per_sim : int
            Number of virtual experiments to perform for each simulation and
            sensor array.
        """
        self._sim_list = sim_list
        self._sensor_arrays = sensor_arrays
        self._num_exp_per_sim = num_exp_per_sim
        self._exp_data = None
        self._exp_stats = None

    def run_experiments(self) -> list[np.ndarray]:
        """Runs the specified number of virtual experiments over the number of
        input simulation cases and virtual sensor arrays.

        Returns
        -------
        list[np.ndarray]
            List of virtual experimental data arrays where the list index
            corresponds to the virtual sensor array and the data is an array
            with shape=(n_sims,n_exps,n_sens,n_comps,n_time_steps).
        """

        n_sims = len(self._sim_list)
        # shape=list[n_sens_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        self._exp_data = [None]*len(self._sensor_arrays)

        for ii,aa in enumerate(self._sensor_arrays):
            meas_array = np.zeros((n_sims,self._num_exp_per_sim)+
                                aa.get_measurement_shape())

            for jj,ss in enumerate(self._sim_list):
                aa.get_field().set_sim_data(ss)

                for ee in range(self._num_exp_per_sim):
                    meas_array[jj,ee,:,:,:] = aa.calc_measurements()

            self._exp_data[ii] = meas_array

        # shape=list[n_sens_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        return self._exp_data


    def calc_stats(self) -> list[ExperimentStats]:
        """Calculates summary statistics over the number of virtual experiments
        specified. If `run_experiments()` has not been called then it is called
        to generate the virtual experimental data to perform the statistical
        calculations.

        Returns
        -------
        list[ExperimentStats]
            List of summary statistics data classes for the virtual experiments.
            The list index correponds to the virtual sensor array.
        """
        if self._exp_data is None:
            self._exp_data = self.run_experiments()

        # shape=list[n_sens_arrays](n_sims,n_sens,n_comps,n_time_steps)
        self._exp_stats = [None]*len(self._sensor_arrays)
        for ii,_ in enumerate(self._sensor_arrays):
            array_stats = ExperimentStats()
            array_stats.max = np.max(self._exp_data[ii],axis=1)
            array_stats.min = np.min(self._exp_data[ii],axis=1)
            array_stats.mean = np.mean(self._exp_data[ii],axis=1)
            array_stats.std = np.std(self._exp_data[ii],axis=1)
            array_stats.med = np.median(self._exp_data[ii],axis=1)
            array_stats.q25 = np.quantile(self._exp_data[ii],0.25,axis=1)
            array_stats.q75 = np.quantile(self._exp_data[ii],0.75,axis=1)
            array_stats.mad = np.median(np.abs(self._exp_data[ii] -
                np.median(self._exp_data[ii],axis=1,keepdims=True)),axis=1)
            self._exp_stats[ii] = array_stats

        # shape=list[n_sens_arrays](n_sims,n_sens,n_comps,n_time_steps)
        return self._exp_stats





