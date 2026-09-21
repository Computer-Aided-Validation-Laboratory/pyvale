from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pyvale.vfm.vfmesh import (
    MeshNodalCoordinates,
    _generate_vf_mesh_nodal_coord,
)


def _structured_mesh(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> MeshNodalCoordinates:
    x, y = np.meshgrid(x_axis, y_axis)
    return MeshNodalCoordinates(nodal_coord_x=x, nodal_coord_y=y)


def test_vf_mesh_snapping_supports_descending_coordinate_axes() -> None:
    descending = np.array([3.0, 2.0, 1.0, 0.0])

    snapped = _generate_vf_mesh_nodal_coord(
        _structured_mesh(descending, descending),
        np.array([3, 3], dtype=np.uint32),
    )

    npt.assert_array_equal(snapped.nodal_coord_x[0], descending)
    npt.assert_array_equal(snapped.nodal_coord_y[:, 0], descending)


def test_vf_mesh_snapping_tolerates_half_spacing_roundoff() -> None:
    # The ideal middle virtual node is 0.55. Its nearest measured coordinate
    # is 0.5, but binary roundoff makes the computed distance exceed the
    # nominal half-spacing by roughly 4e-17.
    axis = 0.1 * np.arange(12, dtype=np.float64)

    snapped = _generate_vf_mesh_nodal_coord(
        _structured_mesh(axis, axis),
        np.array([2, 2], dtype=np.uint32),
    )

    expected = np.array([0.0, 0.5, 1.1])
    npt.assert_allclose(snapped.nodal_coord_x[0], expected, rtol=0.0, atol=1.0e-15)
    npt.assert_allclose(snapped.nodal_coord_y[:, 0], expected, rtol=0.0, atol=1.0e-15)
