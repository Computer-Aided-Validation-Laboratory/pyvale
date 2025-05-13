# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
import mooseherder as mh


class IField(ABC):
    """Interface (abstract base class) for sampling (interpolating) physical
    fields from simulations to provide sensor values at specified locations and
    times.
    """

    @abstractmethod
    def set_sim_data(self,sim_data: mh.SimData) -> None:
        """Abstract method. Sets the SimData object that will be interpolated to
        obtain sensor values. The purpose of this is to be able to apply the
        same sensor array to an array of different simulations.

        Parameters
        ----------
        sim_data : mh.SimData
            Mooseherder SimData object. Contains a mesh and a simulated
            physical field.
        """

    @abstractmethod
    def get_sim_data(self) -> mh.SimData:
        """Abstract method. Gets the simulation data object associated with this
        field. Used by pyvale visualisation tools to display simulation data
        with simulated sensor values.

        Returns
        -------
        mh.SimData
            Mooseherder SimData object. Contains a mesh and a simulated
            physical field.
        """

    @abstractmethod
    def get_time_steps(self) -> np.ndarray:
        """Abstract method. Gets a 1D array of time steps from the simulation
        data.

        Returns
        -------
        np.ndarray
            1D array of simulation time steps. shape=(num_time_steps,)
        """

    @abstractmethod
    def get_visualiser(self) -> pv.UnstructuredGrid:
        """Abstract method. Gets a pyvista unstructured grid object for
        visualisation purposes.

        Returns
        -------
        pv.UnstructuredGrid
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
    def get_component_index(self,component: str) -> int:
        """Gets the index for a component of the physical field. Used for
        getting the index of a component in the sensor measurement array.

        Parameters
        ----------
        component : str
            String key for the field component (e.g. 'temperature' or 'disp_x').

        Returns
        -------
        int
            Index for the selected field component
        """

    @abstractmethod
    def sample_field(self,
                    points: np.ndarray,
                    times: np.ndarray | None = None,
                    angles: tuple[Rotation,...] | None = None,
                    ) -> np.ndarray:
        """Samples (interpolates) the simulation field at the specified
        positions, times, and angles.

        Parameters
        ----------
        points : np.ndarray
            Spatial points to be sampled with the rows indicating the point
            number of the columns indicating the X,Y and Z coordinates.
        times : np.ndarray | None, optional
            Times to sample the underlying simulation. If None then the
            simulation time steps are used and no temporal interpolation is
            performed, by default None.
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
