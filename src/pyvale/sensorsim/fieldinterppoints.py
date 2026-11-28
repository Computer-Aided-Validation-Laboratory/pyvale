# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldinterp import interp_to_sample_time
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler
from pyvale.sensorsim.simtools import coords_to_2D


class FieldInterpPoints(IFieldSampler):
    """Class for interpolating mesh-free simulation fields to the virtual
    sensor locations and sample times.

    Implements the `IFieldSampler` interface.
    """
    __slots__ = (
        "_comp_keys",
        "_coords",
        "_interp_funcs",
        "_sim_data",
        "_sim_time_steps",
        "_spatial_dims",
    )

    def __init__(
        self,
        sim_data: SimData,
        comp_keys: tuple[str, ...],
        spatial_dims: EDim,
    ) -> None:
        """
        Parameters
        ----------
        sim_data : SimData
            Simulation data object containing the physical field(s) that the
            virtual sensors will sample.
        comp_keys : tuple[str,...]
            Tuple of string keys for the components of the field(s) to be
            interpolated.
        spatial_dims : EDim
            Enumeration used to determine the number of spatial dimensions of
            the simulation to determine the underlying element types in the
            mesh.
        """
        if (
            sim_data.time is None
            or sim_data.coords is None
            or sim_data.node_vars is None
        ):
            raise ValueError(
                "Point field sampling requires coordinates, times and nodal fields."
            )
        self._sim_data = sim_data
        self._sim_time_steps = sim_data.time
        self._comp_keys = comp_keys
        self._spatial_dims = spatial_dims

        # Collapse problem to 2D
        self._coords = sim_data.coords
        if self._spatial_dims == EDim.TWOD:
            self._coords = coords_to_2D(self._coords)

        # We do this once instead of inside the loop to save a lot of time as
        # the coordinates don't change between frames
        triang = Delaunay(self._coords)

        self._interp_funcs = {}
        for cc in self._comp_keys:
            interp_frames = []
            for tt in range(self._sim_time_steps.shape[0]):
                interp = LinearNDInterpolator(triang, sim_data.node_vars[cc][:, tt])
                interp_frames.append(interp)

            self._interp_funcs[cc] = interp_frames

    def get_time_steps(self) -> np.ndarray:
        return self._sim_time_steps

    def get_num_components(self) -> int:
        return len(self._comp_keys)

    def get_component_keys(self) -> tuple[str, ...]:
        return self._comp_keys

    def get_sim_data(self) -> SimData:
        return self._sim_data

    def sample(self, points: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
        if self._spatial_dims == EDim.TWOD:
            points = coords_to_2D(points)

        n_points = points.shape[0]
        n_comps = len(self._comp_keys)
        n_sim_time = self._sim_time_steps.shape[0]
        sample_at_sim_time = np.empty((n_points, n_comps, n_sim_time), dtype=np.float64)

        for ii, cc in enumerate(self._comp_keys):
            for tt in range(self._sim_time_steps.shape[0]):
                interp_func = self._interp_funcs[cc][tt]
                sample_at_sim_time[:, ii, tt] = interp_func(points)

        if times is None:
            return sample_at_sim_time

        return interp_to_sample_time(sample_at_sim_time, self._sim_time_steps, times)
