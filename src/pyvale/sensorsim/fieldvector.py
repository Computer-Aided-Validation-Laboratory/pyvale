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
from pyvale.sensorsim.fieldconverter import (simdata_to_pyvista_interp,
                                   simdata_to_pyvista_vis)
from pyvale.sensorsim.fieldinterpmesh import FieldInterpMesh
from pyvale.sensorsim.fieldinterppoints import FieldInterpPoints
from pyvale.sensorsim.fieldtransform import (transform_vector_2d,
                                   transform_vector_2d_batch,
                                   transform_vector_3d,
                                   transform_vector_3d_batch)
from pyvale.sensorsim.enums import EDim

class FieldVector(IField):
    """Class for sampling (interpolating) vector fields from simulations to
    provide sensor values at specified locations and times. Supports 
    interpolation of mesh-based data (with a connectivity table) and point 
    clouds.

    Implements the `IField` interface.
    """
    __slots__ = ("_comp_keys","_spatial_dims","_sim_data","_interpolator",
                 "_visualiser")

    def __init__(self,
                 sim_data: mh.SimData,
                 comp_keys: tuple[str,...],
                 spatial_dims: EDim) -> None:
        """
        Parameters
        ----------
        sim_data : mh.SimData
            Simulation data object containing the mesh and field to interpolate.
        comp_keys : tuple[str,...]
            String keys to the vector field component in the `SimData` nodal 
            variables dictionary. For displacement in 2D: ('disp_x','disp_y') 
            and ('disp_x','disp_y','disp_z').
        spatial_dims : EDim
            Number of spatial dimensions (TWOD or THREED) used for identifying 
            element types. If point cloud data then set to the number of 
            dimensions of the problem as 2D triangulation is much faster than 
            3D.
        """
        self._comp_keys = comp_keys
        self._spatial_dims = spatial_dims

        # NOTE: these get set in the function call to `set_sim_data` - this is
        # separated out to allow inserting a new simdata object
        self._sim_data = None
        self._interpolator = None
        self._visualiser = None
        self.set_sim_data(sim_data)

    def set_sim_data(self, sim_data: mh.SimData) -> None:
        self._sim_data = sim_data

        self._visualiser = simdata_to_pyvista_vis(sim_data,
                                                  self._spatial_dims)
        if self._sim_data.connect is None:
            self._interpolator = FieldInterpPoints(self._sim_data,
                                                   self._comp_keys,
                                                   self._spatial_dims)
        else:
            self._interpolator = FieldInterpMesh(self._sim_data,
                                                 self._comp_keys,
                                                 self._spatial_dims)

    def get_sim_data(self) -> mh.SimData:
        return self._sim_data

    def get_time_steps(self) -> np.ndarray:
        return self._sim_data.time

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._visualiser

    def get_all_components(self) -> tuple[str, ...]:
        return self._comp_keys

    def get_component_index(self, comp_key: str) -> int:
        return self._comp_keys.index(comp_key)

    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        field_data = self._interpolator.interp_field(points,times)

        if angles is None:
            return field_data

        # NOTE:
        # ROTATION= object rotates with coords fixed
        # For Z rotation: sin negative in row 1.
        # TRANSFORMATION= coords rotate with object fixed
        # For Z transformation: sin negative in row 2, transpose scipy mat.

        # If we only have one angle we assume all sensors have the same angle
        # and we can batch process the rotations
        if len(angles) == 1:
            rmat = angles[0].as_matrix().T

            #TODO: assumes 2D in the x-y plane
            if self._spatial_dims == EDim.TWOD:
                rmat = rmat[:2,:2]
                field_data = transform_vector_2d_batch(rmat,field_data)
            else:
                field_data = transform_vector_3d_batch(rmat,field_data)

        else: # Need to rotate each sensor using individual rotation = loop :(
            #TODO: assumes 2D in the x-y plane
            if self._spatial_dims == EDim.TWOD:
                for ii,rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    rmat = rmat[:2,:2]
                    field_data[ii,:,:] = transform_vector_2d(rmat,
                                                             field_data[ii,:,:])

            else:
                for ii,rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    field_data[ii,:,:] = transform_vector_3d(rmat,
                                                             field_data[ii,:,:])

        return field_data

