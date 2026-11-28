# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.field import IField, _bind_field_source
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler


class FieldScalar(IField):
    """Class for sampling (interpolating) scalar fields from simulations to
    provide sensor values at specified locations and times. Supports 
    interpolation of mesh-based data (with a connectivity table) and point 
    clouds.

    Implements the `IField` interface.
    """
    __slots__ = (
        "_comp_key",
        "_sampler",
        "_sim_data",
        "_spatial_dims",
        "_visualiser",
    )

    def __init__(
        self, field_source: SimData | IFieldSampler, comp_key: str, spatial_dims: EDim
    ) -> None:
        """
        Parameters
        ----------
        field_source : SimData or IFieldSampler
            Simulation data or a sampler supplying the world-frame field.
        comp_key : str
            String key for the scalar field component in the `SimData` nodal
            variables dictionary.
        spatial_dims : EDim
            Number of spatial dimensions (TWOD or THREED) used for identifying
            element types. If point cloud data then set to the number of
            dimensions of the problem as 2D triangulation is much faster than
            3D.
        """

        self._comp_key = comp_key
        self._spatial_dims = spatial_dims

        self._set_field_source(field_source)

    def _set_field_source(self, source: SimData | IFieldSampler) -> None:
        sampler, sim_data = _bind_field_source(
            source, self.get_all_components(), self._spatial_dims
        )
        self._sampler = sampler
        self._sim_data = sim_data
        self._visualiser: pv.UnstructuredGrid | pv.PolyData | None = None

    def set_field_sampler(self, sampler: IFieldSampler) -> None:
        self._set_field_source(sampler)

    def set_sim_data(self, sim_data: SimData) -> None:
        self._set_field_source(sim_data)

    def get_sim_data(self) -> SimData:
        if self._sim_data is None:
            raise ValueError("This field sampler has no simulation data.")
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return self._sampler.get_time_steps()

    def get_visualiser(self) -> pv.UnstructuredGrid | pv.PolyData:
        if self._visualiser is None:
            self._visualiser = simdata_to_pyvista_vis(
                self.get_sim_data(), self._spatial_dims
            )
        return self._visualiser

    def get_all_components(self) -> tuple[str, ...]:
        return (self._comp_key,)

    def get_component_index(self, comp_key: str) -> int:
        return 0  # scalar fields only have one component!

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return self._sampler.sample(points, times)
