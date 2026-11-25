# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic small-strain invariants and explicit shear conventions."""

import numpy as np
import pytest

import pyvale.dataio as io


def test_hydrostatic_and_isochoric_uniaxial_strains() -> None:
    tensors = np.array([np.eye(3) * 0.02, np.diag([0.02, -0.01, -0.01])])
    np.testing.assert_allclose(
        io.calc_principal_strains(tensors), [[0.02, 0.02, 0.02], [0.02, -0.01, -0.01]]
    )
    np.testing.assert_allclose(
        io.calc_equivalent_strain(tensors), [0.0, 0.02], atol=1e-17
    )


def test_engineering_shear_and_rotation_invariance() -> None:
    normal = np.zeros((1, 3))
    engineering_shear = np.array([[0.06, 0.0, 0.0]])
    tensor = io.assemble_strain_tensor(
        normal, engineering_shear, shear_convention=io.EStrainShear.ENGINEERING
    )
    tensor_shear = io.assemble_strain_tensor(
        normal, engineering_shear / 2.0, shear_convention=io.EStrainShear.TENSOR
    )
    np.testing.assert_array_equal(tensor, tensor_shear)
    np.testing.assert_array_equal(engineering_shear, [[0.06, 0.0, 0.0]])
    np.testing.assert_allclose(io.calc_principal_strains(tensor), [[0.03, 0.0, -0.03]])
    np.testing.assert_allclose(io.calc_equivalent_strain(tensor), [0.06 / np.sqrt(3.0)])
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = rotation @ tensor @ rotation.T
    rotated = 0.5 * (rotated + rotated.swapaxes(-1, -2))
    np.testing.assert_allclose(
        io.calc_principal_strains(rotated),
        io.calc_principal_strains(tensor),
        atol=1e-17,
    )
    np.testing.assert_allclose(
        io.calc_equivalent_strain(rotated),
        io.calc_equivalent_strain(tensor),
        atol=1e-17,
    )


@pytest.mark.parametrize(
    "tensor",
    [
        np.ones((2, 2)),
        np.full((3, 3), np.nan),
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_invalid_tensors_fail_explicitly(tensor: np.ndarray) -> None:
    with pytest.raises(ValueError):
        io.calc_principal_strains(tensor)
    with pytest.raises(ValueError):
        io.calc_equivalent_strain(tensor)
