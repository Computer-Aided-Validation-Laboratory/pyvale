# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from pyvale.sensorsim.fieldinterp import (FieldInterp,
                                          interp_to_sample_time)

class FieldInterpPoints(FieldInterp):

    def __init__(self,
                 coords: np.ndarray,
                 sim_time_steps: np.ndarray,
                 components: tuple[str,...],
                 ) -> None:
        pass
        # self._coords = coords
        # self._sim_time_steps = sim_time_steps
        # self._components = components

        # self._triangulation = Delaunay()
        # self._interp_funcs = []

        # for ii in range(self._sim_time_steps.shape[0]):
        #     interp = LinearNDInterpolator(coords,)
        #     self._interp_funcs.append(interp)


    def interp_field(self,
                    points: np.ndarray,
                    sample_times: np.ndarray | None = None,
                    ) -> np.ndarray:
        pass
        # n_points = points.shape[0]
        # n_comps = len(self._components)
        # n_sim_time = self._sim_time_steps.shape[0]
        # sample_at_sim_time = np.empty((n_points,n_comps,n_sim_time),
        #                               dtype=np.float64)

        # for ii in range(self._sim_time_steps.shape[0]):

        #     sample_at_sim_time[:,]



        # if sample_times is None:
        #     return sample_at_sim_time

        # return interp_to_sample_time(sample_at_sim_time,
        #                             sim_time_steps,
        #                             sample_times)