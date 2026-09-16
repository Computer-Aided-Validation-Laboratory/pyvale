# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Unit tests for vector and tensor coordinate field transformations."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pyvale.sensorsim.fieldtransform import (
    transform_tensor_2d,
    transform_tensor_2d_batch,
    transform_tensor_3d,
    transform_tensor_3d_batch,
    transform_vector_2d,
    transform_vector_2d_batch,
    transform_vector_3d,
    transform_vector_3d_batch,
    validate_rotation_planar_2d,
)


def test_vector_transform_2d_matches_matrix_product() -> None:
    """2D vector helpers match matrix multiplication with non-symmetric T."""
    trans_mat = np.array([[1.7, -2.4], [0.8, 3.1]], dtype=np.float64)
    vector = np.array([[1.0, 4.0, -2.0], [-3.0, 5.0, 6.0]], dtype=np.float64)

    expected = trans_mat @ vector
    result = transform_vector_2d(trans_mat, vector)
    assert np.allclose(result, expected)

    vector_batch = np.stack([vector, 2.5 * vector, -1.2 * vector], axis=0)
    expected_batch = np.einsum("ij,njt->nit", trans_mat, vector_batch)
    result_batch = transform_vector_2d_batch(trans_mat, vector_batch)
    assert np.allclose(result_batch, expected_batch)


def test_vector_transform_3d_matches_matrix_product() -> None:
    """3D vector helpers match matrix multiplication with 3D rotations."""
    rot = Rotation.from_euler("xyz", (17.0, -31.0, 48.0), degrees=True)
    matrix = rot.as_matrix().T
    vector = np.array(
        [[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]],
        dtype=np.float64,
    )
    expected = matrix @ vector
    result = transform_vector_3d(matrix, vector)
    assert np.allclose(result, expected)

    vector_batch = np.stack([vector, 2.0 * vector], axis=0)
    expected_batch = np.einsum("ij,njt->nit", matrix, vector_batch)
    result_batch = transform_vector_3d_batch(matrix, vector_batch)
    assert np.allclose(result_batch, expected_batch)


def test_tensor_transform_2d_matches_matrix_product() -> None:
    """2D tensor helpers match T @ S @ T.T for symmetric 2D tensors."""
    rot = Rotation.from_euler("z", 33.0, degrees=True)
    matrix_3d = rot.as_matrix().T
    matrix_2d = matrix_3d[:2, :2]

    tensor_mat = np.array([[12.0, -4.5], [-4.5, 7.2]], dtype=np.float64)
    tensor_vec = np.array(
        [[tensor_mat[0, 0]], [tensor_mat[1, 1]], [tensor_mat[0, 1]]],
        dtype=np.float64,
    )

    expected_mat = matrix_2d @ tensor_mat @ matrix_2d.T
    expected_vec = np.array(
        [[expected_mat[0, 0]], [expected_mat[1, 1]], [expected_mat[0, 1]]],
        dtype=np.float64,
    )

    result_vec = transform_tensor_2d(matrix_2d, tensor_vec)
    assert np.allclose(result_vec, expected_vec)

    tensor_batch = np.stack([tensor_vec, 1.8 * tensor_vec], axis=0)
    expected_batch = np.stack([expected_vec, 1.8 * expected_vec], axis=0)
    result_batch = transform_tensor_2d_batch(matrix_2d, tensor_batch)
    assert np.allclose(result_batch, expected_batch)


def test_tensor_transform_3d_matches_matrix_product() -> None:
    """3D tensor helpers match T @ S @ T.T using Voigt components."""
    rot = Rotation.from_euler("xyz", (17.0, -31.0, 48.0), degrees=True)
    matrix = rot.as_matrix().T
    tensor_mat = np.array(
        [
            [11.0, 2.0, 3.0],
            [2.0, 13.0, 5.0],
            [3.0, 5.0, 17.0],
        ],
        dtype=np.float64,
    )
    # Voigt order: xx, yy, zz, xy, xz, yz
    tensor = np.array(
        [
            [tensor_mat[0, 0]],
            [tensor_mat[1, 1]],
            [tensor_mat[2, 2]],
            [tensor_mat[0, 1]],
            [tensor_mat[0, 2]],
            [tensor_mat[1, 2]],
        ],
        dtype=np.float64,
    )
    expected_mat = matrix @ tensor_mat @ matrix.T
    expected = np.array(
        [
            [expected_mat[0, 0]],
            [expected_mat[1, 1]],
            [expected_mat[2, 2]],
            [expected_mat[0, 1]],
            [expected_mat[0, 2]],
            [expected_mat[1, 2]],
        ],
        dtype=np.float64,
    )

    result = transform_tensor_3d(matrix, tensor)
    assert np.allclose(result, expected)

    tensor_batch = np.stack([tensor, 2.0 * tensor], axis=0)
    expected_batch = np.stack([expected, 2.0 * expected], axis=0)
    result_batch = transform_tensor_3d_batch(matrix, tensor_batch)
    assert np.allclose(result_batch, expected_batch)


def test_validate_rotation_planar_2d() -> None:
    """Planar 2D rotation validator accepts Z rotations and rejects others."""
    # Pure Z rotation is planar
    rot_z = Rotation.from_euler("z", 45.0, degrees=True)
    validate_rotation_planar_2d(rot_z.as_matrix())

    # Out-of-plane X or Y rotations must raise ValueError
    rot_x = Rotation.from_euler("x", 15.0, degrees=True)
    with pytest.raises(ValueError, match="Out of plane rotation"):
        validate_rotation_planar_2d(rot_x.as_matrix())

    rot_y = Rotation.from_euler("y", 15.0, degrees=True)
    with pytest.raises(ValueError, match="Out of plane rotation"):
        validate_rotation_planar_2d(rot_y.as_matrix())
