from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import main_gen_calplate as base


BASE_DIR = Path(__file__).resolve().parent

L = base.L
SHORT_DIM = base.SHORT_DIM
THICKNESS = SHORT_DIM / 10.0


@dataclass(frozen=True)
class PlateGeometry:
    corners: np.ndarray
    faces: list[tuple[int, int, int, int]]


def make_plate_geometry() -> PlateGeometry:
    half_length = 0.5 * L
    half_short = 0.5 * SHORT_DIM
    half_thick = 0.5 * THICKNESS

    corners = np.array(
        [
            [-half_length, -half_short, -half_thick],  # 0
            [half_length, -half_short, -half_thick],   # 1
            [half_length, half_short, -half_thick],    # 2
            [-half_length, half_short, -half_thick],   # 3
            [-half_length, -half_short, half_thick],   # 4
            [half_length, -half_short, half_thick],    # 5
            [half_length, half_short, half_thick],     # 6
            [-half_length, half_short, half_thick],    # 7
        ],
        dtype=np.float64,
    )

    # Corner order is CCW as viewed from outside the plate.
    faces = [
        (4, 5, 6, 7),  # top (+z)
        (0, 3, 2, 1),  # bottom (-z)
        (0, 1, 5, 4),  # front (-y)
        (1, 2, 6, 5),  # right (+x)
        (2, 3, 7, 6),  # back (+y)
        (3, 0, 4, 7),  # left (-x)
    ]
    return PlateGeometry(corners=corners, faces=faces)


def coord_key(coord: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.round(val, 12)) for val in coord)


def project_uv(coord: np.ndarray) -> np.ndarray:
    uu = (coord[0] + 0.5 * L) / L
    vv = (coord[1] + 0.5 * SHORT_DIM) / SHORT_DIM
    return np.array([uu, vv], dtype=np.float64)


class NodeBuilder:
    def __init__(self) -> None:
        self.coords: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.node_map: dict[tuple[float, float, float], int] = {}

    def add(self, coord: np.ndarray) -> int:
        key = coord_key(coord)
        if key in self.node_map:
            return self.node_map[key]
        idx = len(self.coords)
        self.coords.append(coord.astype(np.float64, copy=True))
        self.uvs.append(project_uv(coord))
        self.node_map[key] = idx
        return idx

    def add_midpoint(self, idx_a: int, idx_b: int) -> int:
        coord = 0.5 * (self.coords[idx_a] + self.coords[idx_b])
        return self.add(coord)

    def add_face_center(self, indices: tuple[int, int, int, int]) -> int:
        coord = sum((self.coords[ii] for ii in indices), np.zeros(3, dtype=np.float64)) / 4.0
        return self.add(coord)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.vstack(self.coords), np.vstack(self.uvs)


def quad4_mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_plate_geometry()
    coords = geom.corners.copy()
    uvs = np.vstack([project_uv(coord) for coord in coords])
    connect = np.array(geom.faces, dtype=np.int64)
    return coords, connect, uvs


def quad8_mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_plate_geometry()
    builder = NodeBuilder()
    for coord in geom.corners:
        builder.add(coord)

    connect_rows: list[list[int]] = []
    for face in geom.faces:
        n0, n1, n2, n3 = face
        e01 = builder.add_midpoint(n0, n1)
        e12 = builder.add_midpoint(n1, n2)
        e23 = builder.add_midpoint(n2, n3)
        e30 = builder.add_midpoint(n3, n0)
        connect_rows.append([n0, n1, n2, n3, e01, e12, e23, e30])

    coords, uvs = builder.arrays()
    connect = np.array(connect_rows, dtype=np.int64)
    return coords, connect, uvs


def quad9_mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_plate_geometry()
    builder = NodeBuilder()
    for coord in geom.corners:
        builder.add(coord)

    connect_rows: list[list[int]] = []
    for face in geom.faces:
        n0, n1, n2, n3 = face
        e01 = builder.add_midpoint(n0, n1)
        e12 = builder.add_midpoint(n1, n2)
        e23 = builder.add_midpoint(n2, n3)
        e30 = builder.add_midpoint(n3, n0)
        cc = builder.add_face_center(face)
        connect_rows.append([n0, n1, n2, n3, e01, e12, e23, e30, cc])

    coords, uvs = builder.arrays()
    connect = np.array(connect_rows, dtype=np.int64)
    return coords, connect, uvs


def tri3_mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_plate_geometry()
    coords = geom.corners.copy()
    uvs = np.vstack([project_uv(coord) for coord in coords])
    connect_rows: list[list[int]] = []
    for n0, n1, n2, n3 in geom.faces:
        connect_rows.append([n0, n1, n2])
        connect_rows.append([n0, n2, n3])
    connect = np.array(connect_rows, dtype=np.int64)
    return coords, connect, uvs


def tri6_mesh_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom = make_plate_geometry()
    builder = NodeBuilder()
    for coord in geom.corners:
        builder.add(coord)

    connect_rows: list[list[int]] = []
    for n0, n1, n2, n3 in geom.faces:
        # Split quad face along diagonal 0 -> 2.
        m01 = builder.add_midpoint(n0, n1)
        m12 = builder.add_midpoint(n1, n2)
        m20 = builder.add_midpoint(n2, n0)
        connect_rows.append([n0, n1, n2, m01, m12, m20])

        m02 = m20
        m23 = builder.add_midpoint(n2, n3)
        m30 = builder.add_midpoint(n3, n0)
        connect_rows.append([n0, n2, n3, m02, m23, m30])

    coords, uvs = builder.arrays()
    connect = np.array(connect_rows, dtype=np.int64)
    return coords, connect, uvs


def mesh_cases() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return {
        "tri3_calplate3d": tri3_mesh_3d(),
        "tri6_calplate3d": tri6_mesh_3d(),
        "quad4_calplate3d": quad4_mesh_3d(),
        "quad8_calplate3d": quad8_mesh_3d(),
        "quad9_calplate3d": quad9_mesh_3d(),
    }


def main() -> None:
    states = base.selected_motion_states()
    for case_name, (coords, connect, uvs) in mesh_cases().items():
        base.write_case(case_name, coords, connect, uvs, states)

    print(f"Generated 3D calplate mesh cases in {BASE_DIR}")
    print(f"Thickness: {THICKNESS:.10f} m")
    print(f"Length: {L:.10f} m")
    print(f"Short dimension: {SHORT_DIM:.10f} m")


if __name__ == "__main__":
    main()
