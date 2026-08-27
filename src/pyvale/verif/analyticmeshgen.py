#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
Analytic mesh creation tools for testing pyvale sensor simulation and
uncertainty quantification functionality with a known analytic function for the
scalar/vector/tensor field of interest.
"""
import numpy as np


def rectangle_mesh_2d(leng_x: float,
                      leng_y: float,
                      n_elem_x: int,
                      n_elem_y: int) -> tuple[np.ndarray,np.ndarray]:
    """Creates the nodal coordinates and element connectivity table for a simple
    2D quad mesh for a rectangular plate.

    Parameters
    ----------
    leng_x : float
        Length of the plate in the x direction.
    leng_y : float
        Length of the plate in the y direction.
    n_elem_x : int
        Number of elements along the x axis
    n_elem_y : int
        Number of elements along the y axis

    Returns
    -------
        tuple[np.ndarray,np.ndarray]
        The coordinates and connectivity table as numpy arrays. The coordinates
        have shape=(n_nodes,coord[x,y,z]). The connectivity table has shape=
        (num_elems,nodes_per_elem).
    """
    n_elems = n_elem_x*n_elem_y
    n_node_x = n_elem_x+1
    n_node_y = n_elem_y+1
    nodes_per_elem = 4

    coord_x = np.linspace(0,leng_x,n_node_x)
    coord_y = np.linspace(0,leng_y,n_node_y)
    (coord_grid_x,coord_grid_y) = np.meshgrid(coord_x,coord_y)

    coord_x = np.atleast_2d(coord_grid_x.flatten()).T
    coord_y = np.atleast_2d(coord_grid_y.flatten()).T
    coord_z = np.zeros_like(coord_x)
    coords = np.hstack((coord_x,coord_y,coord_z))

    connect = np.zeros((n_elems,nodes_per_elem)).astype(np.int64)
    row = 1
    nn = 0
    for ee in range(n_elems):
        nn += 1
        if nn >= row*n_node_x:
            row += 1
            nn += 1

        connect[ee,:] = np.array([nn-1,nn,nn+n_node_x,nn+n_node_x-1])

    return (coords,connect)


def fill_dims_2d(coord_x: np.ndarray,
              coord_y: np.ndarray,
              time: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Helper function to generate 2D filled arrays tih consistent dimensions
    for array based maths operations. Takes 1D input vectors for the x, y and
    time dimensions and returns 2D arrays with shape=(num_coords,num_timesteps).
    Useful for evaluating analytical functions in space and time.

    Parameters
    ----------
    coord_x : np.ndarray
        1D flattened coordinate list for the x axis.
    coord_y : np.ndarray
        1D flattened coordinate list for the y axis.
    time : np.ndarray
        1D array of time steps.


    Returns
    -------
    tuple[np.ndarray,np.ndarray,np.ndarray]
        Filled 2D arrays with shape=(num_coords,num_timesteps) for the x, y and
        time parameters respectively.
    """
    full_x = np.repeat(np.atleast_2d(coord_x).T,
                       time.shape[0],
                       axis=1)
    full_y = np.repeat(np.atleast_2d(coord_y).T,
                       time.shape[0],
                       axis=1)
    full_time = np.repeat(np.atleast_2d(time),
                          coord_x.shape[0],
                          axis=0)
    return (full_x, full_y, full_time)


def box_mesh_3d(
    leng_x: float,
    leng_y: float,
    leng_z: float,
    n_elem_x: int,
    n_elem_y: int,
    n_elem_z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates nodal coordinates and element connectivity table for a structured
    3D 8-node hexahedral (HEX8) mesh of a rectangular cuboid (box).

    Parameters
    ----------
    leng_x : float
        Length of the box in the x direction.
    leng_y : float
        Length of the box in the y direction.
    leng_z : float
        Length of the box in the z direction.
    n_elem_x : int
        Number of elements along the x axis.
    n_elem_y : int
        Number of elements along the y axis.
    n_elem_z : int
        Number of elements along the z axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The coordinates and connectivity table as numpy arrays. The coordinates
        have shape=(n_nodes, 3). The connectivity table has shape=(n_elems, 8).
    """
    n_node_x = n_elem_x + 1
    n_node_y = n_elem_y + 1
    n_node_z = n_elem_z + 1
    n_elems = n_elem_x * n_elem_y * n_elem_z
    nodes_per_elem = 8

    coord_x = np.linspace(0.0, leng_x, n_node_x)
    coord_y = np.linspace(0.0, leng_y, n_node_y)
    coord_z = np.linspace(0.0, leng_z, n_node_z)

    # Use indexing='ij' so that index 0 is x, index 1 is y, index 2 is z
    grid_x, grid_y, grid_z = np.meshgrid(
        coord_x, coord_y, coord_z, indexing="ij"
    )

    # Flatten order='F' or explicit index mapping
    # Standard node ordering:
    # node(i, j, k) = k * (n_node_x * n_node_y) + j * n_node_x + i
    # When indexing='ij', meshgrid shapes are (n_node_x, n_node_y, n_node_z)
    # Permuting to (z, y, x) or direct flatten:
    # Let's create coords directly matching the index formula:
    coords = np.zeros((n_node_x * n_node_y * n_node_z, 3), dtype=np.float64)
    for kk in range(n_node_z):
        for jj in range(n_node_y):
            for ii in range(n_node_x):
                idx = kk * (n_node_x * n_node_y) + jj * n_node_x + ii
                coords[idx, 0] = coord_x[ii]
                coords[idx, 1] = coord_y[jj]
                coords[idx, 2] = coord_z[kk]

    connect = np.zeros((n_elems, nodes_per_elem), dtype=np.int64)
    elem_idx = 0
    slice_size = n_node_x * n_node_y

    for kk in range(n_elem_z):
        for jj in range(n_elem_y):
            for ii in range(n_elem_x):
                # Bottom face (z = k)
                n0 = kk * slice_size + jj * n_node_x + ii
                n1 = kk * slice_size + jj * n_node_x + (ii + 1)
                n2 = kk * slice_size + (jj + 1) * n_node_x + (ii + 1)
                n3 = kk * slice_size + (jj + 1) * n_node_x + ii
                # Top face (z = k + 1)
                n4 = (kk + 1) * slice_size + jj * n_node_x + ii
                n5 = (kk + 1) * slice_size + jj * n_node_x + (ii + 1)
                n6 = (kk + 1) * slice_size + (jj + 1) * n_node_x + (ii + 1)
                n7 = (kk + 1) * slice_size + (jj + 1) * n_node_x + ii

                connect[elem_idx, :] = [n0, n1, n2, n3, n4, n5, n6, n7]
                elem_idx += 1

    return (coords, connect)


def fill_dims_3d(
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    coord_z: np.ndarray,
    time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to generate 3D filled arrays with consistent dimensions
    for vectorized operations. Takes 1D input vectors for x, y, z and time
    dimensions and returns 2D arrays with shape=(num_coords, num_timesteps).

    Parameters
    ----------
    coord_x : np.ndarray
        1D flattened coordinate array for x axis.
    coord_y : np.ndarray
        1D flattened coordinate array for y axis.
    coord_z : np.ndarray
        1D flattened coordinate array for z axis.
    time : np.ndarray
        1D array of time steps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Filled 2D arrays with shape=(num_coords, num_timesteps) for x, y, z,
        and time parameters respectively.
    """
    n_times = time.shape[0]
    n_coords = coord_x.shape[0]

    full_x = np.repeat(np.atleast_2d(coord_x).T, n_times, axis=1)
    full_y = np.repeat(np.atleast_2d(coord_y).T, n_times, axis=1)
    full_z = np.repeat(np.atleast_2d(coord_z).T, n_times, axis=1)
    full_time = np.repeat(np.atleast_2d(time), n_coords, axis=0)

    return (full_x, full_y, full_z, full_time)

