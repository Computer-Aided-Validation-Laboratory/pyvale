# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
This module contains a set of functions for transforming vector and tensor
fields based on an input transformation matrix.
"""

import numpy as np

def transform_vector_2d(trans_mat: np.ndarray, vector: np.ndarray
                        ) -> np.ndarray:
    """Transforms a 2D vector field based on the input transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(2,2)
    vector : np.ndarray
        Vector field with shape = (2,num_points), where the first row are the X
        components of the field and the second row are the Y components.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape (2,num_points).
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy) = (0,1)

    vector_trans[xx,:] = (trans_mat[0,0]*vector[xx,:]
                        + trans_mat[0,1]*vector[yy,:])
    vector_trans[yy,:] = (trans_mat[0,1]*vector[xx,:]
                        + trans_mat[1,1]*vector[yy,:])
    return vector_trans


def transform_vector_3d(trans_mat: np.ndarray, vector: np.ndarray
                       ) -> np.ndarray:
    """Transforms a 3D vector field based on the input transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(3,3).
    vector : np.ndarray
        Vector field with shape = (3,num_points), where the rows are the X, Y
        and Z components of the vector field.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape=(3,num_points).
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy,zz) = (0,1,2)

    vector_trans[xx,:] = (trans_mat[0,0]*vector[xx,:]
                          + trans_mat[0,1]*vector[yy,:]
                          + trans_mat[0,2]*vector[zz,:])
    vector_trans[yy,:] = (trans_mat[0,1]*vector[xx,:]
                          + trans_mat[1,1]*vector[yy,:]
                          + trans_mat[1,2]*vector[zz,:])
    vector_trans[zz,:] = (trans_mat[0,2]*vector[xx,:]
                          + trans_mat[1,2]*vector[yy,:]
                          + trans_mat[2,2]*vector[zz,:])

    return vector_trans

def transform_vector_2d_batch(trans_mat: np.ndarray, vector: np.ndarray
                        ) -> np.ndarray:
    """Performs a batched 2D vector transformation for a series of sensors
    assuming all sensors have the same transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(2,2).
    vector : np.ndarray
        Input vector field to transform with shape=(num_sensors,2,num_time_steps
        ) where the second dimension is the X and Y components of the vector
        field.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape=(num_sensors,2,num_time_steps),
        where the second dimension is the X and Y components of the
        transformed vector field.
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy) = (0,1)

    vector_trans[:,xx,:] = (trans_mat[0,0]*vector[:,xx,:]
                            + trans_mat[0,1]*vector[:,yy,:])
    vector_trans[:,yy,:] = (trans_mat[0,1]*vector[:,xx,:]
                            + trans_mat[1,1]*vector[:,yy,:])

    return vector_trans


def transform_vector_3d_batch(trans_mat: np.ndarray, vector: np.ndarray
                             ) -> np.ndarray:
    """Performs a batched 3D vector transformation for a series of sensors
    assuming all sensors have the same transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(3,3).
    vector : np.ndarray
        Input vector field to transform with shape=(num_sensors,3,num_time_steps
        ) where the second dimension is the X, Y and Z components of the vector
        field.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape=(num_sensors,2,num_time_steps),
        where the second dimension is the X, Y and Z components of the
        transformed vector field.
    """
    vector_trans = np.zeros_like(vector)
    (xx,yy,zz) = (0,1,2)

    vector_trans[:,xx,:] = (trans_mat[0,0]*vector[:,xx,:]
                            + trans_mat[0,1]*vector[:,yy,:]
                            + trans_mat[0,2]*vector[:,zz,:])
    vector_trans[:,yy,:] = (trans_mat[0,1]*vector[:,xx,:]
                            + trans_mat[1,1]*vector[:,yy,:]
                            + trans_mat[1,2]*vector[:,zz,:])
    vector_trans[:,zz,:] = (trans_mat[0,2]*vector[:,xx,:]
                            + trans_mat[1,2]*vector[:,yy,:]
                            + trans_mat[2,2]*vector[:,zz,:])

    return vector_trans

def transform_tensor_2d(trans_mat: np.ndarray, tensor: np.ndarray
                        ) -> np.ndarray:
    """Transforms a 2D tensor field assuming the shear terms are symmetric.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(2,2)
    tensor : np.ndarray
        Tensor field with shape=(3,num_points) where the rows are the XX, YY and
        XY components of the tensor field

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape=(3,num_points) where the rows are
        the XX, YY and XY components of the tensor field.
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,xy) = (0,1,2)

    tensor_trans[xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:])
                        + trans_mat[0,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]))

    tensor_trans[yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[xx,:]
                                          + trans_mat[1,1]*tensor[xy,:])
                        + trans_mat[1,1]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]))

    tensor_trans[xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:])
                        + trans_mat[1,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]))

    return tensor_trans


def transform_tensor_3d(trans_mat: np.ndarray, tensor: np.ndarray
                       ) -> np.ndarray:
    """Transforms a 3D tensor field assuming all the shear terms are symmetric.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(3,3).
    tensor : np.ndarray
        Tensor field with shape=(6,num_points), where the rows are the XX, YY,
        ZZ, XY, XZ and YZ components of the field.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape=(6,num_points), where the rows are
        the XX, YY, ZZ, XY, XZ and YZ components of the field.
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,zz,xy,xz,yz) = (0,1,2,3,4,5)

    tensor_trans[xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[0,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[0,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[xx,:]
                                          + trans_mat[1,1]*tensor[xy,:]
                                          + trans_mat[1,2]*tensor[xz,:])
                        + trans_mat[1,1]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]
                                          + trans_mat[1,2]*tensor[yz,:])
                        + trans_mat[1,2]*(trans_mat[0,1]*tensor[xz,:]
                                          + trans_mat[1,1]*tensor[yz,:]
                                          + trans_mat[1,2]*tensor[zz,:]))

    tensor_trans[zz,:] = (trans_mat[0,2]*(trans_mat[0,2]*tensor[xx,:]
                                          + trans_mat[1,2]*tensor[xy,:]
                                          + trans_mat[2,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,2]*tensor[xy,:]
                                          + trans_mat[1,2]*tensor[yy,:]
                                          + trans_mat[2,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,2]*tensor[xz,:]
                                          + trans_mat[1,2]*tensor[yz,:]
                                          + trans_mat[2,2]*tensor[zz,:]))

    tensor_trans[xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[1,1]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[1,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[xz,:] = (trans_mat[0,2]*(trans_mat[0,0]*tensor[xx,:]
                                          + trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[0,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,0]*tensor[xy,:]
                                          + trans_mat[0,1]*tensor[yy,:]
                                          + trans_mat[0,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,0]*tensor[xz,:]
                                          + trans_mat[0,1]*tensor[yz,:]
                                          + trans_mat[0,2]*tensor[zz,:]))

    tensor_trans[yz,:] = (trans_mat[0,2]*(trans_mat[0,1]*tensor[xx,:]
                                         + trans_mat[1,1]*tensor[xy,:]
                                         + trans_mat[1,2]*tensor[xz,:])
                        + trans_mat[1,2]*(trans_mat[0,1]*tensor[xy,:]
                                          + trans_mat[1,1]*tensor[yy,:]
                                          + trans_mat[1,2]*tensor[yz,:])
                        + trans_mat[2,2]*(trans_mat[0,1]*tensor[xz,:]
                                          + trans_mat[1,1]*tensor[yz,:]
                                          + trans_mat[1,2]*tensor[zz,:]))

    return tensor_trans


def transform_tensor_2d_batch(trans_mat: np.ndarray, tensor: np.ndarray
                             ) -> np.ndarray:
    """Performs a batched transformation of a 2D tensor field assuming the shear
    terms are symmetric. Assumes the same transformation is applied to all
    sensors in the array so they can be processed together for speed.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(2,2)
    tensor : np.ndarray
        Tensor field with shape=(num_sensors,3,num_points) where the rows are
        the XX, YY and XY components of the tensor field

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape=(num_sensors,3,num_time_steps) where
        the rows are the XX, YY and XY components of the tensor field.
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,xy) = (0,1,2)

    tensor_trans[:,xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:])
                          + trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]))

    tensor_trans[:,yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:])
                          + trans_mat[1,1]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]))

    tensor_trans[:,xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:])
                          + trans_mat[1,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]))

    return tensor_trans


def transform_tensor_3d_batch(trans_mat: np.ndarray, tensor: np.ndarray
                              ) -> np.ndarray:
    """Performs a batched transformation a 3D tensor field assuming all the
    shear terms are symmetric. Assumes all sensors have the same transformation
    applied so they can be processed together for speed.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape=(3,3).
    tensor : np.ndarray
        Tensor field with shape=(num_sensors,6,num_points), where the rows are
        the XX, YY, ZZ, XY, XZ and YZ components of the field.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape=(num_sensors,6,num_points), where
        the rows are the XX, YY, ZZ, XY, XZ and YZ components of the field.
    """
    tensor_trans = np.zeros_like(tensor)
    (xx,yy,zz,xy,xz,yz) = (0,1,2,3,4,5)

    tensor_trans[:,xx,:] = (trans_mat[0,0]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[0,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,yy,:] = (trans_mat[0,1]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,xz,:])
                          + trans_mat[1,1]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]
                                            + trans_mat[1,2]*tensor[:,yz,:])
                          + trans_mat[1,2]*(trans_mat[0,1]*tensor[:,xz,:]
                                            + trans_mat[1,1]*tensor[:,yz,:]
                                            + trans_mat[1,2]*tensor[:,zz,:]))

    tensor_trans[:,zz,:] = (trans_mat[0,2]*(trans_mat[0,2]*tensor[:,xx,:]
                                            + trans_mat[1,2]*tensor[:,xy,:]
                                            + trans_mat[2,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,2]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,yy,:]
                                            + trans_mat[2,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,2]*tensor[:,xz,:]
                                            + trans_mat[1,2]*tensor[:,yz,:]
                                            + trans_mat[2,2]*tensor[:,zz,:]))

    tensor_trans[:,xy,:] = (trans_mat[0,1]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[1,1]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[1,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,xz,:] = (trans_mat[0,2]*(trans_mat[0,0]*tensor[:,xx,:]
                                            + trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[0,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,0]*tensor[:,xy,:]
                                            + trans_mat[0,1]*tensor[:,yy,:]
                                            + trans_mat[0,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,0]*tensor[:,xz,:]
                                            + trans_mat[0,1]*tensor[:,yz,:]
                                            + trans_mat[0,2]*tensor[:,zz,:]))

    tensor_trans[:,yz,:] = (trans_mat[0,2]*(trans_mat[0,1]*tensor[:,xx,:]
                                            + trans_mat[1,1]*tensor[:,xy,:]
                                            + trans_mat[1,2]*tensor[:,xz,:])
                          + trans_mat[1,2]*(trans_mat[0,1]*tensor[:,xy,:]
                                            + trans_mat[1,1]*tensor[:,yy,:]
                                            + trans_mat[1,2]*tensor[:,yz,:])
                          + trans_mat[2,2]*(trans_mat[0,1]*tensor[:,xz,:]
                                            + trans_mat[1,1]*tensor[:,yz,:]
                                            + trans_mat[1,2]*tensor[:,zz,:]))

    return tensor_trans