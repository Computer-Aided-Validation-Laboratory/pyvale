# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================


import numpy as np
import pyvista as pv
import pyvale.mooseherder as mh
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_interp
from pyvale.sensorsim.fieldinterp import (FieldInterp,
                                          interp_to_sample_time)

class FieldInterpMesh(FieldInterp):
    """TODO
    """

    __slots__ = ("_sim_time_steps","_components","_pyvista_interp")

    def __init__(self,
                 sim_data: mh.SimData,
                 components: tuple[str,...],
                 elem_dims: int,
                 ) -> None:
        """
        Parameters
        ----------
        sim_data : mh.SimData
            _description_
        components : tuple[str,...]
            _description_
        elem_dims : int
            _description_
        """
        self._sim_time_steps = sim_data.time
        self._components = components
        self._pyvista_interp = simdata_to_pyvista_interp(sim_data,
                                                         self._components,
                                                         elem_dims=elem_dims)

    def interp_field(self,
                     points: np.ndarray,
                     sample_times: np.ndarray | None = None,
                     ) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        points : np.ndarray
            _description_
        times : np.ndarray | None, optional
            _description_, by default None

        Returns
        -------
        np.ndarray
            _description_
        """
        return sample_pyvista_grid(self._components,
                                   self._pyvista_interp,
                                   self._sim_time_steps,
                                   points,
                                   sample_times)


def sample_pyvista_grid(components: tuple[str,...],
                        pyvista_interp: pv.UnstructuredGrid,
                        sim_time_steps: np.ndarray,
                        points: np.ndarray,
                        sample_times: np.ndarray | None = None
                        ) -> np.ndarray:
    """Function for sampling (interpolating) a pyvista grid object containing
    simulated field data. The pyvista sample method uses VTK to perform the
    spatial interpolation using the element shape functions. If the sampling
    time steps are not the same as the simulation time then a linear
    interpolation over time is performed using numpy.

    NOTE: sampling outside the mesh bounds of the sample returns a value of 0.

    Parameters
    ----------
    components : tuple[str,...]
        String keys for the components to be sampled in the pyvista grid object.
        Useful for only interpolating the field components of interest for speed
        and memory reduction.
    pyvista_interp : pv.UnstructuredGrid
        Pyvista grid object containing the simulation mesh and the components of
        the physical field that will be sampled.
    sim_time_steps : np.ndarray
        Simulation time steps corresponding to the fields in the pyvista grid
        object.
    points : np.ndarray
        Coordinates of the points at which to sample the pyvista grid object.
        shape=(num_points,3) where the columns are the X, Y and Z coordinates of
        the sample points in simulation world coordintes.
    sample_times : np.ndarray | None, optional
        Array of time steps at which to sample the pyvista grid. If None then no
        temporal interpolation is performed and the sample times are assumed to
        be the simulation time steps.

    Returns
    -------
    np.ndarray
        Array of sampled sensor measurements with shape=(num_sensors,
        num_field_components,num_time_steps).
    """
    pv_points = pv.PolyData(points)
    sample_data = pv_points.sample(pyvista_interp)

    n_comps = len(components)
    (n_sensors,n_time_steps) = np.array(sample_data[components[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(components):
        sample_at_sim_time[:,ii,:] = np.array(sample_data[cc])

    if sample_times is None:
        return sample_at_sim_time

    return interp_to_sample_time(sample_at_sim_time,
                                 sim_time_steps,
                                 sample_times)

