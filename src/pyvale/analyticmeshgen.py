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
        have shape=(n_nodes,coord[x,y,z]). The connectivity tables has shape=
        (nodes_per_elem,num_elems).
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

        connect[ee,:] = np.array([nn,nn+1,nn+n_node_x+1,nn+n_node_x])
    connect = connect.T

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
    return (full_x,full_y,full_time)