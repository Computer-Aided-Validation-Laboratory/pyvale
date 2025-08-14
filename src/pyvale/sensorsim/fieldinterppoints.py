# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import pyvale.mooseherder as mh
from pyvale.sensorsim.simtools import (coords_to_2D)
from pyvale.sensorsim.fieldinterp import (FieldInterp,
                                          interp_to_sample_time)



class FieldInterpPoints(FieldInterp):
    __slots__ = ("_sim_time_steps", "_components","_elem_dims","_interp_funcs")

    def __init__(self,
                 sim_data: mh.SimData,
                 components: tuple[str,...],
                 elem_dims: int,
                 ) -> None:

        self._sim_time_steps = sim_data.time
        self._components = components
        self._elem_dims = elem_dims

        # Collapse problem to 2D
        coords = sim_data.coords
        if self._elem_dims == 2:
            coords = coords_to_2D(coords)

        # We do this once instead of inside the loop to save a lot of time as
        # the coordinates don't change between frames
        triang = Delaunay(coords)

        self._interp_funcs = {}
        for cc in self._components:
            interp_frames = []
            for tt in range(self._sim_time_steps.shape[0]):
                interp = LinearNDInterpolator(triang,
                                              sim_data.node_vars[cc][:,tt])
                interp_frames.append(interp)

            self._interp_funcs[cc] = interp_frames


    def interp_field(self,
                    points: np.ndarray,
                    sample_times: np.ndarray | None = None,
                    ) -> np.ndarray:

        if self._elem_dims == 2:
            points = coords_to_2D(points)

        n_points = points.shape[0]
        n_comps = len(self._components)
        n_sim_time = self._sim_time_steps.shape[0]
        sample_at_sim_time = np.empty((n_points,n_comps,n_sim_time),
                                      dtype=np.float64)

        for ii,cc in enumerate(self._components):
            for tt in range(self._sim_time_steps.shape[0]):
                interp_func = self._interp_funcs[cc][tt]
                sample_at_sim_time[:,ii,tt] = interp_func(points)

        if sample_times is None:
            return sample_at_sim_time

        return interp_to_sample_time(sample_at_sim_time,
                                     self._sim_time_steps,
                                     sample_times)

