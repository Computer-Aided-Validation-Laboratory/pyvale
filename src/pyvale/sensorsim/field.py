# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldinterpmesh import FieldInterpMesh
from pyvale.sensorsim.fieldinterppoints import FieldInterpPoints
from pyvale.sensorsim.fieldsamplerinterface import IFieldSampler


class IField(ABC):
    """Interface (abstract base class) for sampling (interpolating) physical
    fields from simulations to provide sensor values at specified locations and
    times.
    """

    @abstractmethod
    def set_field_sampler(self, sampler: IFieldSampler) -> None:
        """Replace the source while retaining field components and coordinates."""

    @abstractmethod
    def set_sim_data(self,sim_data: SimData) -> None:
        """Sets the SimData object that will be interpolated to obtain sensor 
        values. The purpose of this is to be able to apply the same sensor array 
        to an array of different simulations.

        Parameters
        ----------
        sim_data : SimData
            Mooseherder SimData object. Contains a mesh and a simulated
            physical field.
        """

    @abstractmethod
    def get_sim_data(self) -> SimData:
        """Gets the simulation data object associated with this field. Used by 
        pyvale visualisation tools to display simulation data with simulated 
        sensor values.

        Returns
        -------
        SimData
            Mooseherder SimData object. Contains a mesh and a simulated
            physical field.
        """

    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        """Gets the sampler's default 1D array of time steps.

        Returns
        -------
        np.ndarray
            1D array of simulation time steps. shape=(num_time_steps,)
        """

    @abstractmethod
    def get_visualiser(self) -> pv.UnstructuredGrid | pv.PolyData:
        """Gets the display geometry when the sampler is backed by SimData.

        Returns
        -------
        pv.UnstructuredGrid or pv.PolyData
            Pyvista unstructured grid object containing only a mesh without any
            physical field data attached.
        """

    @abstractmethod
    def get_all_components(self) -> tuple[str,...]:
        """Gets the string keys for the component of the physical field. For
        example: a scalar field might just have ('temperature',) whereas a
        vector field might have ('disp_x','disp_y','disp_z').

        Returns
        -------
        tuple[str,...]
            Tuple containing the string keys for all components of the physical
            field.
        """

    @abstractmethod
    def get_component_index(self,comp_key: str) -> int:
        """Gets the index for a component of the physical field. Used for
        getting the index of a component in the sensor measurement array based 
        on its string key.

        Parameters
        ----------
        comp_key : str
            String key for the field component (e.g. 'temperature' or 'disp_x').

        Returns
        -------
        int
            Index for the selected field component.
        """

    @abstractmethod
    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        """Samples (interpolates) the simulation field at the specified
        positions, times, and angles.

        Parameters
        ----------
        points : np.ndarray
            Spatial points to be sampled with the rows indicating the point
            number of the columns indicating the X,Y and Z coordinates.
        times : np.ndarray | None, optional
            Times to sample the underlying field. If None the sampler's
            default times are used.
        angles : tuple[Rotation,...] | None, optional
            Angles to rotate the sampled values into with rotations specified
            with respect to the simulation world coordinates. If a single
            rotation is specified then all points are assumed to have the same
            angle and are batch processed for speed. If None then no rotation is
            performed, by default None.

        Returns
        -------
        np.ndarray
            An array of sampled (interpolated) values with the following
            dimensions: shape=(num_points,num_components,num_time_steps).
        """


def _bind_field_source(
    source: SimData | IFieldSampler, comp_keys: tuple[str, ...], spatial_dims: EDim
) -> tuple[IFieldSampler, SimData | None]:
    """Build or validate a sampler against the field's component layout."""
    if isinstance(source, SimData):
        if source.connect is None:
            source = FieldInterpPoints(source, comp_keys, spatial_dims)
        else:
            source = FieldInterpMesh(source, comp_keys, spatial_dims)
    if source.get_num_components() != len(comp_keys):
        raise ValueError(
            "Sampler component count must match the field's component keys."
        )
    if isinstance(source, (FieldInterpMesh, FieldInterpPoints)):
        if source.get_component_keys() != comp_keys:
            raise ValueError("Sampler component keys and order must match the field.")
        return source, source.get_sim_data()
    return source, None
