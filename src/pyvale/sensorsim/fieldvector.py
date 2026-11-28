# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from typing import Self

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.field import IField, _bind_field_source, _get_sampler_sim_data
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler
from pyvale.sensorsim.fieldtransform import (
    transform_vector_2d,
    transform_vector_2d_batch,
    transform_vector_3d,
    transform_vector_3d_batch,
)


class FieldVector(IField):
    """Class for sampling (interpolating) vector fields from simulations to
    provide sensor values at specified locations and times. Supports
    interpolation of mesh-based data (with a connectivity table) and point
    clouds.

    Implements the `IField` interface.
    """

    __slots__ = (
        "_comp_keys",
        "_sampler",
        "_spatial_dims",
        "_visualiser",
    )

    def __init__(
        self,
        field_source: SimData | IFieldSampler,
        comp_keys: tuple[str, ...],
        spatial_dims: EDim,
    ) -> None:
        """
        Parameters
        ----------
        field_source : SimData or IFieldSampler
            Simulation data or a sampler supplying the world-frame field.
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

        self._sampler = _bind_field_source(
            field_source, self.get_all_components(), spatial_dims
        )
        self._visualiser: pv.UnstructuredGrid | pv.PolyData | None = None

    def with_sampler(self, sampler: IFieldSampler) -> Self:
        return type(self)(
            field_source=sampler,
            comp_keys=self._comp_keys,
            spatial_dims=self._spatial_dims,
        )

    def get_spatial_dims(self) -> EDim:
        return self._spatial_dims

    def get_sim_data(self) -> SimData:
        return _get_sampler_sim_data(self._sampler)

    def get_time_steps(self) -> np.ndarray:
        return self._sampler.get_time_steps()

    def get_visualiser(self) -> pv.UnstructuredGrid | pv.PolyData:
        if self._visualiser is None:
            self._visualiser = simdata_to_pyvista_vis(
                self.get_sim_data(), self._spatial_dims
            )
        return self._visualiser

    def get_all_components(self) -> tuple[str, ...]:
        return self._comp_keys

    def get_component_index(self, comp_key: str) -> int:
        return self._comp_keys.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        field_data = self._sampler.sample(points, times)

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

            # TODO: assumes 2D in the x-y plane
            if self._spatial_dims == EDim.TWOD:
                rmat = rmat[:2, :2]
                field_data = transform_vector_2d_batch(rmat, field_data)
            else:
                field_data = transform_vector_3d_batch(rmat, field_data)

        else:  # Need to rotate each sensor using individual rotation = loop :(
            field_data = field_data.copy()
            # TODO: assumes 2D in the x-y plane
            if self._spatial_dims == EDim.TWOD:
                for ii, rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    rmat = rmat[:2, :2]
                    field_data[ii, :, :] = transform_vector_2d(
                        rmat, field_data[ii, :, :]
                    )

            else:
                for ii, rr in enumerate(angles):
                    rmat = rr.as_matrix().T
                    field_data[ii, :, :] = transform_vector_3d(
                        rmat, field_data[ii, :, :]
                    )

        return field_data
