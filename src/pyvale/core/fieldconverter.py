"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""
import warnings
import numpy as np
import pyvista as pv
from pyvista import CellType
import mooseherder as mh


def simdata_to_pyvista(sim_data: mh.SimData,
                            components: tuple[str,...] | None,
                            spat_dim: int
                            ) -> tuple[pv.UnstructuredGrid,pv.UnstructuredGrid]:
    """Converts the mesh and field data in a `SimData` object into a pyvista
    UnstructuredGrid for sampling (interpolating) the data and visualisation.

    Parameters
    ----------
    sim_data : mh.SimData
        Object containing a mesh and associated field data from a simulation.
    components : tuple[str,...] | None
        String keys for the components of the field to extract from the
        simulation data.
    spat_dim : int
        Number of spatial dimensions (2 or 3) used to determine the element
        types in the mesh from the number of nodes per element.

    Returns
    -------
    tuple[pv.UnstructuredGrid,pv.UnstructuredGrid]
        The first UnstructuredGrid has the field components attached as dataset
        arrays. The second has no field data attached for visualisation.
    """
    flat_connect = np.array([],dtype=np.int64)
    cell_types = np.array([],dtype=np.int64)

    for cc in sim_data.connect:
        # NOTE: need the -1 here to make element numbers 0 indexed!
        temp_connect = sim_data.connect[cc]-1
        (nodes_per_elem,n_elems) = temp_connect.shape

        temp_connect = temp_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)

        this_cell_type = _get_pyvista_cell_type(nodes_per_elem,spat_dim)
        temp_connect = np.insert(temp_connect,idxs,nodes_per_elem)
        #temp_connect = np.insert(temp_connect,idxs,this_cell_type)

        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,temp_connect),dtype=np.int64)

    cells = flat_connect

    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)
    pv_grid_vis = pv.UnstructuredGrid(cells, cell_types, points)

    if components is not None and sim_data.node_vars is not None:
        for cc in components:
            pv_grid[cc] = sim_data.node_vars[cc]

    return (pv_grid,pv_grid_vis)


def _get_pyvista_cell_type(nodes_per_elem: int, spat_dim: int) -> CellType:
    """Helper function to identify the pyvista element type in the mesh.

    Parameters
    ----------
    nodes_per_elem : int
        Number of nodes per element.
    spat_dim : int
        Number of spatial dimensions in the mesh (2 or 3).

    Returns
    -------
    CellType
        Enumeration describing the element type in pyvista.
    """
    cell_type = 0

    if spat_dim == 2:
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
        else:
            warnings.warn(f"Cell type 2D with {nodes_per_elem} "
                          + "nodes not recognised. Defaulting to 4 node QUAD")
            cell_type = CellType.QUAD
    else:
        if nodes_per_elem == 8:
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        elif nodes_per_elem == 10:
            cell_type = CellType.QUADRATIC_TETRA
        elif nodes_per_elem == 20:
            cell_type = CellType.QUADRATIC_HEXAHEDRON
        elif nodes_per_elem == 27:
            cell_type = CellType.TRIQUADRATIC_HEXAHEDRON
        else:
            warnings.warn(f"Cell type 3D with {nodes_per_elem} "
                + "nodes not recognised. Defaulting to 8 node HEX")
            cell_type = CellType.HEXAHEDRON

    print(80*"=")
    print(f"{nodes_per_elem=}")
    print(f"{cell_type=}")
    print(80*"=")

    return cell_type


def scale_length_units(sim_data: mh.SimData,
                       disp_comps: tuple[str,...],
                       scale: float) -> mh.SimData:

    sim_data.coords = sim_data.coords*scale
    for cc in disp_comps:
        sim_data.node_vars[cc] = sim_data.node_vars[cc]*scale

    return sim_data


