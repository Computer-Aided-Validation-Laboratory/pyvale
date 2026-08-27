# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module provides functions for manipulating simulation data objects to be
compatible with the underlying machinery of pyvale.
"""

import numpy as np
import pyvista as pv
from pyvista import CellType
from pyvale.dataio.simdata import SimData
from pyvale.dataio.meshconv import enforce_mesh_convention
from pyvale.dataio.meshconv import extract_surf_mesh as _extract_surf_mesh
from pyvale.sensorsim.enums import EDim


def simdata_to_pyvista_interp(sim_data: SimData,
                              components: tuple[str,...] | None,
                              spatial_dims: EDim,
                              ) -> pv.UnstructuredGrid:
    """Converts the mesh and field data in a `SimData` object into a pyvista
    UnstructuredGrid for interpolating the data.

    Parameters
    ----------
    sim_data : SimData
        Object containing a mesh and associated field data from a simulation.
    components : tuple[str,...] | None
        String keys for the components of the field to extract from the
        simulation data.
    elem_dim : EDim
        Number of spatial dimensions in the simulation (TWOD or THREED).  For 
        mesh-based data this is used to determine the element type and 
        distinguish between 4 node quads in 2D and 4 node tets in 3D. For point
        cloud data this determines if 2D or 3D Delaunay triangulation is used.

    Returns
    -------
    pv.UnstructuredGrid
        As pyvista grid with attached field data to allow for interpolation on
        the mesh using the element shape functions.
    """

    sim_data = enforce_mesh_convention(sim_data)
    pv_grid = _gen_pyvista_grid(sim_data,spatial_dims)

    if components is not None and sim_data.node_vars is not None:
        for cc in components:
            pv_grid[cc] = sim_data.node_vars[cc]

    return pv_grid


def simdata_to_pyvista_vis(sim_data: SimData,
                           spatial_dims: EDim,
                           ) -> pv.UnstructuredGrid | pv.PolyData:
    """Converts the mesh and field data in a `SimData` object into a pyvista
    UnstructuredGrid or PolyData object for visualisation.

    Parameters
    ----------
    sim_data : SimData
        Object containing a mesh and associated field data from a simulation.
    elem_dim : EDim
        Number of spatial dimensions in the simulation (TWOD or THREED).  For 
        mesh-based data this is used to determine the element type and 
        distinguish between 4 node quads in 2D and 4 node tets in 3D. For point
        cloud data this determines if 2D or 3D Delaunay triangulation is used.

    Returns
    -------
    pv.UnstructuredGrid | pv.PolyData
        A pyvista unstructured grid or poly data object that has no field data
        attached for visualisation purposes.
    """
    if sim_data.connect is None:
        return pv.PolyData(sim_data.coords)

    sim_data = enforce_mesh_convention(sim_data)
    return _gen_pyvista_grid(sim_data,spatial_dims)



def _gen_pyvista_grid(sim_data: SimData,
                      spatial_dims: int) -> pv.UnstructuredGrid:
    """Helper function for generating a blank pyvista unstructure grid mesh from
    a SimData object.

    Parameters
    ----------
    sim_data : SimData
        Object containing a mesh and associated field data from a simulation.
    elem_dim : EDim
        Number of spatial dimensions in the simulation (TWOD or THREED).  For 
        mesh-based data this is used to determine the element type and 
        distinguish between 4 node quads in 2D and 4 node tets in 3D. For point
        cloud data this determines if 2D or 3D Delaunay triangulation is used.

    Returns
    -------
    pv.UnstructuredGrid
        A pyvista unstructured grid that has no field data attached.
    """
    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    for cc in sim_data.connect:
        this_connect = np.ascontiguousarray(
            sim_data.connect[cc], dtype=np.int64
        )
        (n_elems,nodes_per_elem) = this_connect.shape

        this_cell_type = _get_pyvista_cell_type(nodes_per_elem,spatial_dims)
        assert this_cell_type is not None, ("Cell type with dimension " +
            f"{spatial_dims} and {nodes_per_elem} nodes per element not " +
            "recognised.")

        this_connect = this_connect.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)

        this_connect = np.insert(this_connect,idxs,nodes_per_elem)

        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,this_connect),dtype=np.int64)

    cells = flat_connect

    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)
    return pv_grid


def extract_surf_mesh(
    sim_data: SimData,
    enforce_convention: bool = True,
) -> SimData:
    """Compatibility wrapper for the shared surface-extraction helper."""

    return _extract_surf_mesh(sim_data, enforce_convention=enforce_convention)

def _get_pyvista_cell_type(
    nodes_per_elem: int,
    spat_dim: EDim | int,
) -> CellType | None:
    """Helper function to identify the pyvista element type in the mesh.

    Parameters
    ----------
    nodes_per_elem : int
        Number of nodes per element.
    spat_dim : EDim | int
        Number of spatial dimensions in the simulation (TWOD or THREED).

    Returns
    -------
    CellType | None
        Enumeration describing the element type in pyvista.
    """
    cell_type = None

    if spat_dim == EDim.TWOD or spat_dim == 2:
        if nodes_per_elem == 4:
            cell_type = CellType.QUAD
        elif nodes_per_elem == 3:
            cell_type = CellType.TRIANGLE
        elif nodes_per_elem == 6:
            cell_type = CellType.QUADRATIC_TRIANGLE
        elif nodes_per_elem == 7:
            cell_type = CellType.BIQUADRATIC_TRIANGLE
        elif nodes_per_elem == 8:
            cell_type = CellType.QUADRATIC_QUAD
        elif nodes_per_elem == 9:
            cell_type = CellType.BIQUADRATIC_QUAD
    elif spat_dim == EDim.THREED or spat_dim == 3:
        if nodes_per_elem == 8:
            cell_type = CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        elif nodes_per_elem == 6:
            cell_type = CellType.WEDGE
        elif nodes_per_elem == 5:
            cell_type = CellType.PYRAMID
        elif nodes_per_elem == 10:
            cell_type = CellType.QUADRATIC_TETRA
        elif nodes_per_elem == 15:
            cell_type = CellType.QUADRATIC_WEDGE
        elif nodes_per_elem == 13:
            cell_type = CellType.QUADRATIC_PYRAMID
        elif nodes_per_elem == 20:
            cell_type = CellType.QUADRATIC_HEXAHEDRON
        elif nodes_per_elem == 27:
            cell_type = CellType.TRIQUADRATIC_HEXAHEDRON

    return cell_type
