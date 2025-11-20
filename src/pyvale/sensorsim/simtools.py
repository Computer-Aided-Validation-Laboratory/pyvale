# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""This module contains helper functions that are useful for inspecting and
manipulating simulation data.
"""

from typing import Any
import dataclasses
import numpy as np
import pyvale.mooseherder as mh
from pyvale.sensorsim.rendermesh import RenderMesh
from pyvale.sensorsim.exceptions import Collapse2Dto3DError


def print_dataclass_fields(in_data: Any) -> None:
    """Diagnostic function to print all fields of a dataclass.

    Parameters
    ----------
    in_data : Any
        A data class to print the type and fields for as well as the type of
        each of the fields.
    """

    print(f"Data class fields for: {type(in_data)}")
    for field in dataclasses.fields(in_data):
        if not field.name.startswith('__'):
            print(f"    {field.name}: {field.type}")
    print()


def print_sim_data(sim_data: mh.SimData) -> None:
    """Diagnostic function for inspecting a sim data object to work out shapes
    of time, coordinates, connectivity tables, node vars, elem vars as well as
    the associated keys in the dicttionaries for the connectivity,
    node/elem/glob vars.

    Parameters
    ----------
    sim_data : mh.SimData
        SimData to print shapes of numpy arrays.
    """
    print()
    if sim_data.time is not None:
        print(f"{sim_data.time.shape=}")
    print()

    if sim_data.coords is not None:
        print(f"{sim_data.coords.shape=}")
    print()

    def print_dict(in_dict: dict | None) -> None:
        if in_dict is None:
            print("    None\n")
            return

        print(f"keys={in_dict.keys()}")
        for kk in in_dict:
            print(f"    {kk}.shape={in_dict[kk].shape}")

        print()

    print("sim_data.connect")
    print_dict(sim_data.connect)
    print("sim_data.node_vars")
    print_dict(sim_data.node_vars)
    print("sim_data.elem_vars")
    print_dict(sim_data.elem_vars)
    print("sim_data.glob_vars")
    print_dict(sim_data.glob_vars)


def print_dimensions(sim_data: mh.SimData) -> None:
    """Diagnostic function for quickly finding the coordinate limits for from a
    given simulation.

    Parameters
    ----------
    sim_data : mh.SimData
        Simulation data object containing the nodal coordinates.
    """
    print(80*"-")
    print("SimData Dimensions:")
    print(f"x [min,max] = [{np.min(sim_data.coords[:,0])}," + \
        f"{np.max(sim_data.coords[:,0])}]")
    print(f"y [min,max] = [{np.min(sim_data.coords[:,1])}," + \
        f"{np.max(sim_data.coords[:,1])}]")
    print(f"z [min,max] = [{np.min(sim_data.coords[:,2])}," + \
        f"{np.max(sim_data.coords[:,2])}]")
    print(f"t [min,max] = [{np.min(sim_data.time)},{np.max(sim_data.time)}]")
    print(80*"-")


def get_sim_dims(sim_data: mh.SimData) -> dict[str,tuple[float,float]]:
    """Diagnostic function for extracting the dimensional limits in space and
    time from a SimData object. Useful for finding the spatial dimensions over
    which simulated sensors can be placed as well as the times over which they
    can sampled the underlying field.

    Parameters
    ----------
    sim_data : mh.SimData
        Simulation data object containing the coordinates and time steps.

    Returns
    -------
    dict[str,tuple[float,float]]
        Dictionary of space and time coordinate limits keyed as 'x','y','z' for
        the spatial dimensions and 't' for time. The dictionary will return a
        tuple with the (min,max) of the given dimension.
    """
    sim_dims = {}
    sim_dims["x"] = (np.min(sim_data.coords[:,0]),np.max(sim_data.coords[:,0]))
    sim_dims["y"] = (np.min(sim_data.coords[:,1]),np.max(sim_data.coords[:,1]))
    sim_dims["z"] = (np.min(sim_data.coords[:,2]),np.max(sim_data.coords[:,2]))
    sim_dims["t"] = (np.min(sim_data.time),np.max(sim_data.time))
    return sim_dims


def centre_mesh_nodes(nodes: np.ndarray, spat_dim: int) -> np.ndarray:
    """A method to centre the nodes of a mesh around the origin.

    Parameters
    ----------
    nodes : np.ndarray
        An array containing the node locations of the mesh.
    spat_dim : int
        The spatial dimension of the mesh.

    Returns
    -------
    np.ndarray
        An array containing the mesh node locations, but centred around
        the origin.
    """
    max = np.max(nodes, axis=0)
    min = np.min(nodes, axis=0)
    middle = max - ((max - min) / 2)
    if spat_dim == 3:
        middle[2] = 0
    centred = np.subtract(nodes, middle)
    return centred


def get_deformed_nodes(timestep: int,
                        render_mesh: RenderMesh) -> np.ndarray | None:
    """A method to obtain the deformed locations of all the nodes at a given
        timestep.

    Parameters
    ----------
    timestep : int
        The timestep at which to find the deformed nodes.
    render_mesh: RenderMeshData
        A dataclass containing the skinned mesh and simulation results.

    Returns
    -------
    np.ndarray | None
        An array containing the deformed values of all the components at
        each node location. Returns None if there are no deformation values.
    """
    if render_mesh.fields_disp is None:
        return None

    added_disp = render_mesh.fields_disp[:, timestep]
    if added_disp.shape[1] == 2:
        added_disp = np.hstack((added_disp,np.zeros([added_disp.shape[0],1])))
    coords = np.delete(render_mesh.coords, 3, axis=1)
    deformed_nodes = coords + added_disp
    return deformed_nodes


def scale_length_units(scale: float,
                       sim_data: mh.SimData,
                       disp_keys: tuple[str,...] | None = None,
                       ) -> mh.SimData:
    """Used to scale the length units of a simulation. Commonly used to convert
    SI units to mm for use with visualisation tools and rendering algorithms.

    Parameters
    ----------
    scale : float
        Scale multiplier used to scale the coordinates and displacement fields
        if specified.
    sim_data : mh.SimData
        Simulation dataclass that will be scaled.
    disp_keys : tuple[str,...] | None, optional
        Tuple of string keys for the displacement keys to be scaled, by default
        None. If None then the displacements are not scaled.

    Returns
    -------
    mh.SimData
        Simulation dataclass with scaled length units.
    """
    sim_data.coords = sim_data.coords*scale

    if disp_keys is not None:
        for cc in disp_keys:
            sim_data.node_vars[cc] = sim_data.node_vars[cc]*scale

    return sim_data



def coords_to_2D(coords_3d: np.ndarray) -> np.ndarray:
    """Collapses and input coordinate array with 3 spatial dimensions to have
    only 2 spatial dimensions. Useful for removing the axis that is zero for
    Delaunay triangulation.

    Parameters
    ----------
    coords_3d : np.ndarray
        Array of coordinates with shape=(num_points,coord[X,Y,Z]). Note that
        this is the same format as in a `SimData` object

    Returns
    -------
    np.ndarray
        A coordinate array for the 2D simulation with the zero axis removed.
        The array has shape (num_points,2) where the second axis represents the
        2D coords.

    Raises
    ------
    Collapse2Dto3DError
        Problem is either 3D or 1D, coordinates must have exactly one axis which
        is all zeros.
    """
    zero_axs = get_sim_zero_axs(coords_3d)
    num_zero_ax = np.sum(zero_axs)

    if num_zero_ax == 0:
        raise Collapse2Dto3DError("No coordinate axis is close to zero, unable" \
            "to collapse problem to 2D. Check coords in SimData object.")

    if num_zero_ax > 1:
        raise Collapse2Dto3DError("Two coordinate axes are close to zero," \
            " problem is 1D cannot collapse problem to 2D. Check coords in " \
            "SimData object.")

    ax_to_zero = np.argmax(zero_axs != 0)
    coords_2d = np.delete(coords_3d,ax_to_zero,axis=1)
    return coords_2d


def get_sim_zero_axs(coords_3d: np.ndarray) -> np.ndarray:
    """Helper function to extract which (if any) axis is all zeros in a
    coordinate array.

    Parameters
    ----------
    coords_3d : np.ndarray
        Array of coordinates with shape=(num_points,coord[X,Y,Z]). Note that
        this is the same format as in a `SimData` object

    Returns
    -------
    np.ndarray
        A 3 element array with '1' where the axis is zero and '0' everywhere
        else. The elements of the array nominally represent (X,Y,Z).
    """
    zero_axs = np.zeros((3,),dtype=np.uintp)
    for ii in range(coords_3d.shape[1]):
        if np.allclose(coords_3d[:,ii],0):
            zero_axs[ii] = 1

    return zero_axs


def is_sim_2D(coords_3d: np.ndarray) -> bool:
    """Helper function that inspects a numpy array of coordinates and determines
    if one of the spatial axes is all zero to infer that the simulation is 2D.

    Parameters
    ----------
    coords_3d : np.ndarray
        Array of coordinates with shape=(num_points,coord[X,Y,Z]). Note that
        this is the same format as in a `SimData` object

    Returns
    -------
    bool
        True if the simulation has exactly one coordinate axis as all zero and
        False otherwise.
    """
    zero_axs = get_sim_zero_axs(coords_3d)

    if np.sum(zero_axs) == 1:
        return True

    return False



