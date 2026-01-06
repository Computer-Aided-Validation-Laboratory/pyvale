# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
import pyvale.mooseherder as mh

from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.fieldinterpmesh import FieldInterpMesh
from pyvale.sensorsim.fieldinterppoints import FieldInterpPoints
from pyvale.sensorsim.enums import EDim

class FieldScalar(IField):
    """Class for sampling (interpolating) scalar fields from simulations to
    provide sensor values at specified locations and times. Supports 
    interpolation of mesh-based data (with a connectivity table) and point 
    clouds.

    Implements the `IField` interface.
    """
    __slots__ = ("_comp_key","_spatial_dims","_sim_data","_interpolator",
                 "_visualiser")

    def __init__(self,
                 sim_data: mh.SimData,
                 comp_key: str,
                 spatial_dims: EDim) -> None:
        """
        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the mesh and field to interpolate.
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

        # NOTE: these get set in the function call to `set_sim_data` - this is
        # separated out to allow inserting a new simdata object
        self._sim_data =  None
        self._visualiser = None
        self._interpolator = None
        self.set_sim_data(sim_data)


    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._sim_data = sim_data

        self._visualiser = simdata_to_pyvista_vis(sim_data,
                                                  self._spatial_dims)
        if self._sim_data.connect is None:
            self._interpolator = FieldInterpPoints(self._sim_data,
                                                   (self._comp_key,),
                                                   self._spatial_dims)
        else:
            self._interpolator = FieldInterpMesh(self._sim_data,
                                                 (self._comp_key,),
                                                 self._spatial_dims)

    def get_sim_data(self) -> mh.SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return self._sim_data.time

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._visualiser

    def get_all_components(self) -> tuple[str, ...]:
        return (self._comp_key,)

    def get_component_index(self, comp_key: str) -> int:
        return 0 # scalar fields only have one component!

    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        return self._interpolator.interp_field(points,times)

