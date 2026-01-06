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
import pyvale.mooseherder as mh
from pyvale.sensorsim.enums import EDim


def simdata_to_pyvista_interp(sim_data: mh.SimData,
                              components: tuple[str,...] | None,
                              spatial_dims: EDim,
                              ) -> pv.UnstructuredGrid:
    """Converts the mesh and field data in a `SimData` object into a pyvista
    UnstructuredGrid for interpolating the data.

    Parameters
    ----------
    sim_data : mh.SimData
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

    pv_grid = _gen_pyvista_grid(sim_data,spatial_dims)

    if components is not None and sim_data.node_vars is not None:
        for cc in components:
            pv_grid[cc] = sim_data.node_vars[cc]

    return pv_grid


def simdata_to_pyvista_vis(sim_data: mh.SimData,
                           spatial_dims: EDim,
                           ) -> pv.UnstructuredGrid | pv.PolyData:
    """Converts the mesh and field data in a `SimData` object into a pyvista
    UnstructuredGrid or PolyData object for visualisation.

    Parameters
    ----------
    sim_data : mh.SimData
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

    return _gen_pyvista_grid(sim_data,spatial_dims)



def _gen_pyvista_grid(sim_data: mh.SimData,
                      spatial_dims: int) -> pv.UnstructuredGrid:
    """Helper function for generating a blank pyvista unstructure grid mesh from
    a SimData object.

    Parameters
    ----------
    sim_data : mh.SimData
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
        # NOTE: need the -1 here to make element numbers 0 indexed!
        this_connect = np.copy(sim_data.connect[cc])-1
        (nodes_per_elem,n_elems) = this_connect.shape

        this_cell_type = _get_pyvista_cell_type(nodes_per_elem,spatial_dims)
        assert this_cell_type is not None, ("Cell type with dimension " +
            f"{spatial_dims} and {nodes_per_elem} nodes per element not " +
            "recognised.")

        # VTK and exodus have different winding for 3D higher order quads
        this_connect = _exodus_to_pyvista_connect(this_cell_type,this_connect)

        this_connect = this_connect.T.flatten()
        idxs = np.arange(0,n_elems*nodes_per_elem,nodes_per_elem,dtype=np.int64)

        this_connect = np.insert(this_connect,idxs,nodes_per_elem)

        cell_types = np.hstack((cell_types,np.full(n_elems,this_cell_type)))
        flat_connect = np.hstack((flat_connect,this_connect),dtype=np.int64)

    cells = flat_connect

    points = sim_data.coords
    pv_grid = pv.UnstructuredGrid(cells, cell_types, points)
    return pv_grid


# TODO: make this work for sim_data with multiple connectivity
def extract_surf_mesh(sim_data: mh.SimData) -> mh.SimData:
    """Extracts a surface mesh from a 3D simulation dataclass. Useful for
    limiting the memory required for analysing sensors that only measure surface
    fields. This function currently supports:
        - A single connectivity table
        - Higher order retrahedral and hexahedral elements (but not wedges or
          pyramids)

    NOTE: this function returns the surface mesh with element nodal winding
    consistent with th exodus output format.

    Parameters
    ----------
    sim_data : mh.SimData
        Simulation dataclass containing the 3D mesh from which the surface mesh
        is to be extracted.

    Returns
    -------
    mh.SimData
        Simulation data class containing the data for the surface mesh.
    """

    # NOTE: need to fix exodus 1 indexing for now and put it back at the end
    # shape=(nodes_per_elem,num_elems)
    connect = np.copy(sim_data.connect["connect1"])-1
    num_elems = connect.shape[1]

    assert "connect2" not in sim_data.connect, \
        "Multiple connectivity tables not supported yet."

    # Mapping of node numbers to faces for each element face
    face_map = _get_surf_map(nodes_per_elem=connect.shape[0])
    faces_per_elem = face_map.shape[0]
    nodes_per_face = face_map.shape[1]

    # shape=(faces_per_elem,nodes_per_face,num_elems)
    faces_wound = connect[face_map,:]
    # shape=(num_elems,faces_per_elem,nodes_per_face)
    faces_wound = faces_wound.transpose((2,0,1))

    # Create an array of all faces with shape=(total_faces,nodes_per_face)
    faces_total = faces_per_elem*num_elems
    faces_flat_wound = faces_wound.reshape((faces_total,nodes_per_face))
    # Sort the rows so nodes are in the same order when comparing them
    faces_flat_sorted = np.copy(np.sort(faces_flat_wound,axis=1))

    # Count each unique face in the list of faces, faces that appear only once
    # must be external faces
    (_,
     faces_unique_inds,
     faces_unique_counts) = np.unique(faces_flat_sorted,
                                      axis=0,
                                      return_counts=True,
                                      return_index=True)

    # Indices of the external faces in faces_flat
    faces_ext_inds_in_unique = np.where(faces_unique_counts==1)[0]

    # shape=(num_ext_faces,nodes_per_face)
    faces_ext_inds = faces_unique_inds[faces_ext_inds_in_unique]

    faces_ext_wound = faces_flat_wound[faces_ext_inds]

    faces_coord_inds = np.unique(faces_ext_wound.flatten())
    faces_coords = np.copy(sim_data.coords[faces_coord_inds])

    faces_shape = faces_ext_wound.shape
    faces_ext_wound_flat = faces_ext_wound.flatten()
    faces_ext_remap_flat = np.copy(faces_ext_wound_flat)

    # Remap coordinates in the connectivity to match the trimmed list of coords
    # that belong to the external faces
    for mm,cc in enumerate(faces_coord_inds):
        if mm == cc:
            continue

        ind_to_map = np.where(faces_ext_wound_flat == cc)[0]
        faces_ext_remap_flat[ind_to_map] = mm

    faces_ext_remap = faces_ext_remap_flat.reshape(faces_shape)
    faces_ext_remap = faces_ext_remap + 1 # back to exodus 1 index

    # Now we build the SimData object and slice out the node and element
    # variables using the coordinate indexing.
    face_data = mh.SimData(coords=faces_coords,
                           connect={"connect1":faces_ext_remap.T},
                           time=sim_data.time)

    if sim_data.node_vars is not None:
        face_data.node_vars = {}
        for nn in sim_data.node_vars:
            face_data.node_vars[nn] = sim_data.node_vars[nn][faces_coord_inds,:]

    if sim_data.elem_vars is not None:
        face_data.elem_vars = {}
        for ee in sim_data.node_vars:
            face_data.elem_vars[ee] = sim_data.elem_vars[ee][faces_coord_inds,:]

    return face_data

#TODO: make this support triangular prisms in 3D.
def _get_pyvista_cell_type(nodes_per_elem: int, 
                           spat_dim: EDim) -> CellType | None:
    """Helper function to identify the pyvista element type in the mesh.

    Parameters
    ----------
    nodes_per_elem : int
        Number of nodes per element.
    spat_dim : EDim
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
            cell_type =  CellType.HEXAHEDRON
        elif nodes_per_elem == 4:
            cell_type = CellType.TETRA
        elif nodes_per_elem == 10:
            cell_type = CellType.QUADRATIC_TETRA
        elif nodes_per_elem == 20:
            cell_type = CellType.QUADRATIC_HEXAHEDRON
        elif nodes_per_elem == 27:
            cell_type = CellType.TRIQUADRATIC_HEXAHEDRON

    return cell_type

#TODO: make this support triangular prisms in 3D.
def _exodus_to_pyvista_connect(cell_type: CellType,
                               connect: np.ndarray) -> np.ndarray:
    """Helper function that specifies the nodal winding map for higher order
    tet and hex elements between the exodus output format and pyvista (VTK).

    Parameters
    ----------
    cell_type : CellType
        pyvista (VTK) cell type enumeration.
    connect : np.ndarray
        Input connectivity table in exodus winding format.
        shape=(nodes_per_elem,num_elems)

    Returns
    -------
    np.ndarray
        Output connectivity table in pyvista (VTK) format.
        shape=(nodes_per_elem,num_elems)
    """
    copy_connect = np.copy(connect)

    # NOTE: it looks like VTK does not support TET14
    # VTK and exodus have different winding for 3D higher order quads
    if cell_type == CellType.QUADRATIC_HEXAHEDRON:
        connect[12:16,:] = copy_connect[16:20,:]
        connect[16:20,:] = copy_connect[12:16,:]
    elif cell_type == CellType.TRIQUADRATIC_HEXAHEDRON:
        connect[12:16,:] = copy_connect[16:20,:]
        connect[16:20,:] = copy_connect[12:16,:]
        connect[20:24,:] = copy_connect[23:27,:]
        connect[24,:] = copy_connect[21,:]
        connect[25,:] = copy_connect[22,:]
        connect[26,:] = copy_connect[20,:]

    return connect

#TODO: make this support triangular prisms in 3D.
def _get_surf_map(nodes_per_elem: int) -> np.ndarray:
    """Helper function specifying the mapping from 3D tet and hex elements to
    the individual faces consistent with the exodus output format.

    Parameters
    ----------
    nodes_per_elem : int
        Number of nodes per element.

    Returns
    -------
    np.ndarray
        Array of indices mapping the nodes to faces with shape=(num_faces,n
        odes_per_face)

    Raises
    ------
    ValueError
        Element type is not supported.
    """
    if nodes_per_elem == 4: # TET4
       return np.array(((0,1,2),
                        (0,3,1),
                        (0,2,3),
                        (1,3,2)))

    if nodes_per_elem == 8: # HEX8
        return np.array(((0,1,2,3),
                         (0,3,7,4),
                         (4,7,6,5),
                         (1,5,6,2),
                         (0,4,5,1),
                         (2,6,7,3)))

    if nodes_per_elem == 10: # TET10
       return np.array(((0,1,2,4,5,6),
                        (0,3,1,4,8,7),
                        (0,2,3,6,9,7),
                        (1,3,2,8,9,5)))

    if nodes_per_elem == 20: # HEX20
        return np.array(((0,1,2,3,8,9,10,11),
                         (0,3,7,4,11,15,19,12),
                         (4,7,6,5,19,18,17,16),
                         (1,5,6,2,13,17,14,9),
                         (0,4,5,1,12,16,13,8),
                         (2,6,7,3,14,18,15,10)))

    if nodes_per_elem == 27: # HEX27
        return np.array(((0,1,2,3,8,9,10,11,21),
                         (0,3,7,4,11,15,19,12,23),
                         (4,7,6,5,19,18,17,16,22),
                         (1,5,6,2,13,17,14,9,24),
                         (0,4,5,1,12,16,13,8,25),
                         (2,6,7,3,14,18,15,10,26)))

    raise ValueError("Number of nodes does not match a 3D element type for " \
        "surface extraction.")
