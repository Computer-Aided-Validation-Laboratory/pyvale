# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Affine, VTK, and Newton inverse maps for PixInt2D."""

import numpy as np

from ..mesh2d import DisplacementSeries2D, Mesh2D
from .elements import in_natural_domain, shape_functions
from .model import EPxIntMapping


def map_points(
    mesh: Mesh2D,
    displacements: DisplacementSeries2D,
    frame: int,
    query_x: np.ndarray,
    query_y: np.ndarray,
    mode: EPxIntMapping,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map target-world points to reference coordinates for one frame."""
    if frame < 0 or frame >= displacements.values.shape[0]:
        raise IndexError("frame is outside the displacement series.")

    x_coord = np.asarray(query_x, dtype=np.float64).ravel()
    y_coord = np.asarray(query_y, dtype=np.float64).ravel()
    deformed = mesh.coords + displacements.values[frame]

    if mode is EPxIntMapping.AFFINE:
        return _affine(mesh, deformed, x_coord, y_coord)

    if mode is EPxIntMapping.VTK:
        return _vtk(mesh, deformed, x_coord, y_coord)

    if mode is EPxIntMapping.NEWTON_ONE_ELEM:
        if mesh.connectivity.shape[0] != 1:
            raise ValueError("NEWTON_ONE_ELEM requires exactly one element.")
        return _newton_candidates(mesh, deformed, x_coord, y_coord, (0,))

    if mode in (EPxIntMapping.NEWTON_MESH_UNSTRUCT,
                EPxIntMapping.NEWTON_MESH_STRUCT,
                EPxIntMapping.STRUCTURED_QUAD9):
        return _newton_mesh(mesh, deformed, x_coord, y_coord)
    raise ValueError(f"Unsupported mapping mode {mode!r}.")


def _affine(
    mesh: Mesh2D,
    deformed: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate an exact global affine inverse map."""
    design = np.column_stack((deformed, np.ones(len(deformed))))
    coefficients, _, _, _ = np.linalg.lstsq(design, mesh.coords, rcond=None)
    residual = np.max(np.abs(design @ coefficients - mesh.coords))

    if residual > 1.0e-8:
        raise ValueError("AFFINE mapping requires an affine deformation field.")

    reference = np.column_stack((query_x, query_y, np.ones(len(query_x))))
    reference = reference @ coefficients
    return reference[:, 0], reference[:, 1], np.ones(len(query_x), dtype=bool)


def _vtk(
    mesh: Mesh2D,
    deformed: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use VTK interpolation as an independent mapping baseline."""
    import pyvista as pv

    cell_types = {
        3: pv.CellType.TRIANGLE,
        4: pv.CellType.QUAD,
        6: pv.CellType.QUADRATIC_TRIANGLE,
        8: pv.CellType.QUADRATIC_QUAD,
        9: pv.CellType.BIQUADRATIC_QUAD,
    }
    nodes = mesh.connectivity.shape[1]
    cells = np.hstack((
        np.full((len(mesh.connectivity), 1), nodes, dtype=np.intp),
        mesh.connectivity,
    )).ravel()

    grid = pv.UnstructuredGrid(
        cells,
        np.full(len(mesh.connectivity), cell_types[nodes], dtype=np.uint8),
        np.column_stack((deformed, np.zeros(len(deformed)))),
    )
    grid.point_data["reference_x"] = mesh.coords[:, 0]
    grid.point_data["reference_y"] = mesh.coords[:, 1]

    query = np.column_stack((query_x, query_y, np.zeros(len(query_x))))
    sampled = pv.PolyData(query).sample(grid)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    return (
        np.asarray(sampled.point_data["reference_x"], dtype=np.float64),
        np.asarray(sampled.point_data["reference_y"], dtype=np.float64), valid,
    )


def _newton_mesh(
    mesh: Mesh2D,
    deformed: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points through every conservative bounding-box candidate."""
    reference_x = np.zeros(len(query_x), dtype=np.float64)
    reference_y = np.zeros(len(query_y), dtype=np.float64)
    valid = np.zeros(len(query_x), dtype=bool)
    element_coords = deformed[mesh.connectivity]
    lower = element_coords.min(axis=1)
    upper = element_coords.max(axis=1)

    for point_index, (x_coord, y_coord) in enumerate(zip(query_x, query_y)):
        candidates = np.flatnonzero(
            (x_coord >= lower[:, 0] - 1.0e-9)
            & (x_coord <= upper[:, 0] + 1.0e-9)
            & (y_coord >= lower[:, 1] - 1.0e-9)
            & (y_coord <= upper[:, 1] + 1.0e-9),
        )

        for element_index in candidates:
            result = _newton_element(
                mesh, deformed, int(element_index), x_coord, y_coord,
            )
            if result is not None:
                reference_x[point_index], reference_y[point_index] = result
                valid[point_index] = True
                break
    return reference_x, reference_y, valid


def _newton_candidates(
    mesh: Mesh2D,
    deformed: np.ndarray,
    query_x: np.ndarray,
    query_y: np.ndarray,
    candidates: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map points through a fixed ordered set of element candidates."""
    reference_x = np.zeros(len(query_x), dtype=np.float64)
    reference_y = np.zeros(len(query_y), dtype=np.float64)
    valid = np.zeros(len(query_x), dtype=bool)

    for point_index, (x_coord, y_coord) in enumerate(zip(query_x, query_y)):
        for element_index in candidates:
            result = _newton_element(mesh, deformed, element_index, x_coord, y_coord)
            if result is not None:
                reference_x[point_index], reference_y[point_index] = result
                valid[point_index] = True
                break
    return reference_x, reference_y, valid


def _newton_element(
    mesh: Mesh2D,
    deformed: np.ndarray,
    element_index: int,
    query_x: float,
    query_y: float,
) -> tuple[float, float] | None:
    """Invert one deformed element with a bounded Newton iteration."""
    element_deformed = deformed[mesh.connectivity[element_index]]

    if mesh.element_type.name.startswith("TRI"):
        xi, eta = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi, eta = 0.0, 0.0

    for _ in range(32):
        values, d_xi, d_eta = shape_functions(mesh.element_type, xi, eta)
        mapped = values @ element_deformed
        residual = mapped - (query_x, query_y)
        jacobian = np.array((
            d_xi @ element_deformed,
            d_eta @ element_deformed,
        )).T
        determinant = float(np.linalg.det(jacobian))

        if abs(determinant) < 1.0e-14:
            return None

        step = np.linalg.solve(jacobian, residual)
        xi -= step[0]
        eta -= step[1]

        if max(abs(step)) < 1.0e-11 and max(abs(residual)) < 1.0e-9:
            break
    else:
        return None
    if not in_natural_domain(mesh.element_type, xi, eta):
        return None

    values, _, _ = shape_functions(mesh.element_type, xi, eta)
    reference = values @ mesh.coords[mesh.connectivity[element_index]]
    return float(reference[0]), float(reference[1])


__all__ = ["map_points"]
