'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Digital Validation Team
================================================================================
'''
from typing import Callable
from dataclasses import dataclass
import numpy as np
from pyvale.sensors.pointsensorarray import PointSensorArray
import mooseherder as mh


@dataclass
class ExperimentStats:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    cov: np.ndarray | None = None
    max: np.ndarray | None = None
    min: np.ndarray | None = None
    med: np.ndarray | None = None
    q25: np.ndarray | None = None
    q75: np.ndarray | None = None
    mad: np.ndarray | None = None


class ExperimentSimulator:
    def __init__(self,
                 sim_list: list[mh.SimData],
                 sensor_arrays: list[PointSensorArray],
                 num_exp_per_sim: int
                 ) -> None:

        self.sim_list = sim_list
        self.sensor_arrays = sensor_arrays
        self.num_exp_per_sim = num_exp_per_sim
        self._exp_data = [None]*len(self.sensor_arrays)
        self._exp_stats = [None]*len(self.sensor_arrays)

    def get_data(self) -> list[np.ndarray | None]:
        return self._exp_data

    def get_stats(self) -> list[np.ndarray | None]:
        return self._exp_stats

    def run_experiments(self) -> list[np.ndarray]:
        n_sims = len(self.sim_list)
        # shape=list[n_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        self._exp_data = [None]*len(self.sensor_arrays)

        for ii,aa in enumerate(self.sensor_arrays):
            meas_array = np.zeros((n_sims,self.num_exp_per_sim)+
                                aa.get_measurement_shape())

            for jj,ss in enumerate(self.sim_list):
                aa.field.set_sim_data(ss)

                for ee in range(self.num_exp_per_sim):
                    meas_array[jj,ee,:,:,:] = aa.calc_measurements()

            self._exp_data[ii] = meas_array

        return self._exp_data


    def calc_stats(self) -> list[np.ndarray]:
        # shape=list[n_arrays](n_sims,n_exps,n_sens,n_comps,n_time_steps)
        self._exp_stats = [None]*len(self.sensor_arrays)
        for ii,_ in enumerate(self.sensor_arrays):
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

        return self._exp_stats

    '''
    def apply_over_experiment(func: Callable[[np.ndarray],np.ndarray]
                              ) -> list[np.ndarray]:
    '''



