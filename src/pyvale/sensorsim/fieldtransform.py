# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Functions for transforming vector and tensor fields based on an input
transformation matrix.
"""

import numpy as np


def transform_vector_2d(
    trans_mat: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Transforms a 2D vector field based on the input transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (2, 2).
    vector : np.ndarray
        Vector field with shape (2, num_points), where row 0 is the X component
        and row 1 is the Y component.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape (2, num_points).
    """
    vector_trans = np.zeros_like(vector)
    (xx, yy) = (0, 1)

    vector_trans[xx, :] = (
        trans_mat[0, 0] * vector[xx, :]
        + trans_mat[0, 1] * vector[yy, :]
    )
    vector_trans[yy, :] = (
        trans_mat[1, 0] * vector[xx, :]
        + trans_mat[1, 1] * vector[yy, :]
    )
    return vector_trans


def transform_vector_3d(
    trans_mat: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Transforms a 3D vector field based on the input transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (3, 3).
    vector : np.ndarray
        Vector field with shape (3, num_points), where rows are X, Y, and Z
        components of the vector field.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape (3, num_points).
    """
    vector_trans = np.zeros_like(vector)
    (xx, yy, zz) = (0, 1, 2)

    vector_trans[xx, :] = (
        trans_mat[0, 0] * vector[xx, :]
        + trans_mat[0, 1] * vector[yy, :]
        + trans_mat[0, 2] * vector[zz, :]
    )
    vector_trans[yy, :] = (
        trans_mat[1, 0] * vector[xx, :]
        + trans_mat[1, 1] * vector[yy, :]
        + trans_mat[1, 2] * vector[zz, :]
    )
    vector_trans[zz, :] = (
        trans_mat[2, 0] * vector[xx, :]
        + trans_mat[2, 1] * vector[yy, :]
        + trans_mat[2, 2] * vector[zz, :]
    )

    return vector_trans


def transform_vector_2d_batch(
    trans_mat: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Performs a batched 2D vector transformation for an array of sensors.

    Assumes all sensors share the same transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (2, 2).
    vector : np.ndarray
        Input vector field with shape (num_sensors, 2, num_time_steps) where
        dimension 1 holds the X and Y components.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape (num_sensors, 2, num_time_steps).
    """
    vector_trans = np.zeros_like(vector)
    (xx, yy) = (0, 1)

    vector_trans[:, xx, :] = (
        trans_mat[0, 0] * vector[:, xx, :]
        + trans_mat[0, 1] * vector[:, yy, :]
    )
    vector_trans[:, yy, :] = (
        trans_mat[1, 0] * vector[:, xx, :]
        + trans_mat[1, 1] * vector[:, yy, :]
    )

    return vector_trans


def transform_vector_3d_batch(
    trans_mat: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Performs a batched 3D vector transformation for an array of sensors.

    Assumes all sensors share the same transformation matrix.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (3, 3).
    vector : np.ndarray
        Input vector field with shape (num_sensors, 3, num_time_steps) where
        dimension 1 holds X, Y, and Z components.

    Returns
    -------
    np.ndarray
        Transformed vector field with shape (num_sensors, 3, num_time_steps).
    """
    vector_trans = np.zeros_like(vector)
    (xx, yy, zz) = (0, 1, 2)

    vector_trans[:, xx, :] = (
        trans_mat[0, 0] * vector[:, xx, :]
        + trans_mat[0, 1] * vector[:, yy, :]
        + trans_mat[0, 2] * vector[:, zz, :]
    )
    vector_trans[:, yy, :] = (
        trans_mat[1, 0] * vector[:, xx, :]
        + trans_mat[1, 1] * vector[:, yy, :]
        + trans_mat[1, 2] * vector[:, zz, :]
    )
    vector_trans[:, zz, :] = (
        trans_mat[2, 0] * vector[:, xx, :]
        + trans_mat[2, 1] * vector[:, yy, :]
        + trans_mat[2, 2] * vector[:, zz, :]
    )

    return vector_trans


def transform_tensor_2d(
    trans_mat: np.ndarray,
    tensor: np.ndarray,
) -> np.ndarray:
    """Transforms a 2D symmetric tensor field (XX, YY, XY).

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (2, 2).
    tensor : np.ndarray
        Tensor field with shape (3, num_points) where rows are the XX, YY, and
        XY components.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape (3, num_points).
    """
    tensor_trans = np.zeros_like(tensor)
    (xx, yy, xy) = (0, 1, 2)

    tensor_trans[xx, :] = (
        trans_mat[0, 0] * trans_mat[0, 0] * tensor[xx, :]
        + 2.0 * trans_mat[0, 0] * trans_mat[0, 1] * tensor[xy, :]
        + trans_mat[0, 1] * trans_mat[0, 1] * tensor[yy, :]
    )

    tensor_trans[yy, :] = (
        trans_mat[1, 0] * trans_mat[1, 0] * tensor[xx, :]
        + 2.0 * trans_mat[1, 0] * trans_mat[1, 1] * tensor[xy, :]
        + trans_mat[1, 1] * trans_mat[1, 1] * tensor[yy, :]
    )

    tensor_trans[xy, :] = (
        trans_mat[0, 0] * trans_mat[1, 0] * tensor[xx, :]
        + (
            trans_mat[0, 0] * trans_mat[1, 1]
            + trans_mat[0, 1] * trans_mat[1, 0]
        )
        * tensor[xy, :]
        + trans_mat[0, 1] * trans_mat[1, 1] * tensor[yy, :]
    )

    return tensor_trans


def transform_tensor_3d(
    trans_mat: np.ndarray,
    tensor: np.ndarray,
) -> np.ndarray:
    """Transforms a 3D symmetric tensor field (XX, YY, ZZ, XY, XZ, YZ).

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (3, 3).
    tensor : np.ndarray
        Tensor field with shape (6, num_points) where rows are the XX, YY, ZZ,
        XY, XZ, and YZ components.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape (6, num_points).
    """
    tensor_trans = np.zeros_like(tensor)
    (xx, yy, zz, xy, xz, yz) = (0, 1, 2, 3, 4, 5)

    t00, t01, t02 = trans_mat[0, 0], trans_mat[0, 1], trans_mat[0, 2]
    t10, t11, t12 = trans_mat[1, 0], trans_mat[1, 1], trans_mat[1, 2]
    t20, t21, t22 = trans_mat[2, 0], trans_mat[2, 1], trans_mat[2, 2]

    sxx, syy, szz = tensor[xx, :], tensor[yy, :], tensor[zz, :]
    sxy, sxz, syz = tensor[xy, :], tensor[xz, :], tensor[yz, :]

    ts00 = t00 * sxx + t01 * sxy + t02 * sxz
    ts01 = t00 * sxy + t01 * syy + t02 * syz
    ts02 = t00 * sxz + t01 * syz + t02 * szz

    ts10 = t10 * sxx + t11 * sxy + t12 * sxz
    ts11 = t10 * sxy + t11 * syy + t12 * syz
    ts12 = t10 * sxz + t11 * syz + t12 * szz

    ts20 = t20 * sxx + t21 * sxy + t22 * sxz
    ts21 = t20 * sxy + t21 * syy + t22 * syz
    ts22 = t20 * sxz + t21 * syz + t22 * szz

    tensor_trans[xx, :] = ts00 * t00 + ts01 * t01 + ts02 * t02
    tensor_trans[yy, :] = ts10 * t10 + ts11 * t11 + ts12 * t12
    tensor_trans[zz, :] = ts20 * t20 + ts21 * t21 + ts22 * t22
    tensor_trans[xy, :] = ts00 * t10 + ts01 * t11 + ts02 * t12
    tensor_trans[xz, :] = ts00 * t20 + ts01 * t21 + ts02 * t22
    tensor_trans[yz, :] = ts10 * t20 + ts11 * t21 + ts12 * t22

    return tensor_trans


def transform_tensor_2d_batch(
    trans_mat: np.ndarray,
    tensor: np.ndarray,
) -> np.ndarray:
    """Performs a batched transformation of a 2D symmetric tensor field.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (2, 2).
    tensor : np.ndarray
        Tensor field with shape (num_sensors, 3, num_time_steps) where dimension
        1 holds the XX, YY, and XY components.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape (num_sensors, 3, num_time_steps).
    """
    tensor_trans = np.zeros_like(tensor)
    (xx, yy, xy) = (0, 1, 2)

    tensor_trans[:, xx, :] = (
        trans_mat[0, 0] * trans_mat[0, 0] * tensor[:, xx, :]
        + 2.0 * trans_mat[0, 0] * trans_mat[0, 1] * tensor[:, xy, :]
        + trans_mat[0, 1] * trans_mat[0, 1] * tensor[:, yy, :]
    )

    tensor_trans[:, yy, :] = (
        trans_mat[1, 0] * trans_mat[1, 0] * tensor[:, xx, :]
        + 2.0 * trans_mat[1, 0] * trans_mat[1, 1] * tensor[:, xy, :]
        + trans_mat[1, 1] * trans_mat[1, 1] * tensor[:, yy, :]
    )

    tensor_trans[:, xy, :] = (
        trans_mat[0, 0] * trans_mat[1, 0] * tensor[:, xx, :]
        + (
            trans_mat[0, 0] * trans_mat[1, 1]
            + trans_mat[0, 1] * trans_mat[1, 0]
        )
        * tensor[:, xy, :]
        + trans_mat[0, 1] * trans_mat[1, 1] * tensor[:, yy, :]
    )

    return tensor_trans


def transform_tensor_3d_batch(
    trans_mat: np.ndarray,
    tensor: np.ndarray,
) -> np.ndarray:
    """Performs a batched transformation of a 3D symmetric tensor field.

    Parameters
    ----------
    trans_mat : np.ndarray
        Transformation matrix with shape (3, 3).
    tensor : np.ndarray
        Tensor field with shape (num_sensors, 6, num_time_steps) where dimension
        1 holds the XX, YY, ZZ, XY, XZ, and YZ components.

    Returns
    -------
    np.ndarray
        Transformed tensor field with shape (num_sensors, 6, num_time_steps).
    """
    tensor_trans = np.zeros_like(tensor)
    (xx, yy, zz, xy, xz, yz) = (0, 1, 2, 3, 4, 5)

    t00, t01, t02 = trans_mat[0, 0], trans_mat[0, 1], trans_mat[0, 2]
    t10, t11, t12 = trans_mat[1, 0], trans_mat[1, 1], trans_mat[1, 2]
    t20, t21, t22 = trans_mat[2, 0], trans_mat[2, 1], trans_mat[2, 2]

    sxx, syy, szz = tensor[:, xx, :], tensor[:, yy, :], tensor[:, zz, :]
    sxy, sxz, syz = tensor[:, xy, :], tensor[:, xz, :], tensor[:, yz, :]

    ts00 = t00 * sxx + t01 * sxy + t02 * sxz
    ts01 = t00 * sxy + t01 * syy + t02 * syz
    ts02 = t00 * sxz + t01 * syz + t02 * szz

    ts10 = t10 * sxx + t11 * sxy + t12 * sxz
    ts11 = t10 * sxy + t11 * syy + t12 * syz
    ts12 = t10 * sxz + t11 * syz + t12 * szz

    ts20 = t20 * sxx + t21 * sxy + t22 * sxz
    ts21 = t20 * sxy + t21 * syy + t22 * syz
    ts22 = t20 * sxz + t21 * syz + t22 * szz

    tensor_trans[:, xx, :] = ts00 * t00 + ts01 * t01 + ts02 * t02
    tensor_trans[:, yy, :] = ts10 * t10 + ts11 * t11 + ts12 * t12
    tensor_trans[:, zz, :] = ts20 * t20 + ts21 * t21 + ts22 * t22
    tensor_trans[:, xy, :] = ts00 * t10 + ts01 * t11 + ts02 * t12
    tensor_trans[:, xz, :] = ts00 * t20 + ts01 * t21 + ts02 * t22
    tensor_trans[:, yz, :] = ts10 * t20 + ts11 * t21 + ts12 * t22

    return tensor_trans


def validate_rotation_planar_2d(
    rmat: np.ndarray,
    tol: float = 1e-6,
) -> None:
    """Validates that a 3D rotation matrix represents a planar rotation about Z.

    Parameters
    ----------
    rmat : np.ndarray
        Transformation matrix with shape (3, 3) or larger.
    tol : float, optional
        Tolerance for checking out-of-plane elements (default 1e-6).

    Raises
    ------
    ValueError
        If any out-of-plane rotation component exceeds tolerance.
    """
    if rmat.shape[0] < 3 or rmat.shape[1] < 3:
        return

    if (
        abs(rmat[2, 2] - 1.0) > tol
        or abs(rmat[0, 2]) > tol
        or abs(rmat[1, 2]) > tol
        or abs(rmat[2, 0]) > tol
        or abs(rmat[2, 1]) > tol
    ):
        raise ValueError(
            "Out of plane rotation detected for 2D field. Rotations for 2D "
            "fields must be planar around the Z axis."
        )
