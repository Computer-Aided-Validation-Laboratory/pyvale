#!/usr/bin/env python3
r"""
Application test 4: comparison of stereo-DIC outputs with FEA ground truth.

This script compares the stereo DIC results produced by
``application_4_fea_dic.py`` across the three optical cases (``air_diffuse``,
``pipe`` and ``water``) against the original quadratic (``QUAD8``) FEA frame
rendered by ``application_4_fea.py``.
The quadratic ``QUAD8`` displacement field is sampled on the undeformed front
face with PyVista/VTK (``vtkResampleWithDataSet``), using VTK's high-order cell
location and shape-function interpolation.
Each DIC subset ``(subset_x, subset_y)`` in Camera 0 is mapped onto the
specimen front face $z = +0.5$ mm in two ways:
1. ``refracted`` (primary): analytical 3D Snell's-law ray trace from Camera 0
   through the cylindrical acrylic pipe wall ($R_{\text{out}} = 18.5$ mm,
   $R_{\text{in}} = 16.5$ mm, $n = 1.490$) and optional water fill
   ($n = 1.333$) onto $z = +0.5$ mm, yielding the true physical material point
   imaged by each subset.
2. ``nominal``: uncorrected straight pinhole back-projection in air, quantifying
   the apparent spatial registration error when cylindrical refraction
   magnification is neglected.

Coordinate and displacement conventions
---------------------------------------
* World / FEA coordinates have $+x$ right, $+y$ up and $+z$ towards Camera 0
  (specimen front face on $z = +0.5$ mm, circular hole centred at
  $(0.0, -1.5)$ mm).
* Pyvale's stereo-DIC engine reports displacements and 3D surface coordinates
  in Camera 0's right-handed pinhole frame:
  $u = +\Delta x_{\text{world}}$, $v = -\Delta y_{\text{world}}$ (down),
  $w = -\Delta z_{\text{world}}$ (depth away from Camera 0).  Pass
  ``--w-convention world`` if you prefer $w = +\Delta z_{\text{world}}$ as
  drawn in ``application_4_fea.py``'s ``plot_fea_frame()``.


Outputs layout (written to ``thesis-output/<case-base-dir>/compare/`` by default)::

    field_summary_comparison.csv         distribution stats (FEA & DIC cases)
    dic_vs_fea_errors.csv                pointwise DIC - FEA error metrics
    dic_intercase_errors.csv             pairwise DIC - DIC differences
    calibration_and_shape_comparison.csv calibration, ZNCC & 3D shape errors
    compare_displacement_fields.png      side-by-side u, v, w, |d| maps
    compare_dic_vs_fea_errors.png        spatial DIC - FEA residual maps
    compare_intercase_differences.png    spatial DIC - DIC difference maps
    compare_line_profiles.png            horizontal & vertical line profiles
    compare_error_summary_and_shape.png  error bars/boxplots & 3D depth Z(x)

Usage::

    /home/wiera/anaconda3/envs/pyvale/bin/python \
        tests/raytracer/thesis-tests/application_4_fea_compare.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from pyvale import calib, dic  # noqa: F401 - verifies newer pyvale env


# ================================================================================
# Layout, optical and specimen constants
#
# Mirrors application_4_fea_dic.py without importing global_utils or
# pyvale.raytracer so this script runs directly in the conda "pyvale" env.
# ================================================================================

BASE_TEST_DIR = Path(__file__).resolve().parent.parent / "thesis-output"
THESIS_DATA_DIR = Path(__file__).resolve().parent.parent / "thesis-data"

DEFAULT_CASE_BASE_DIR = "app4_fea/ssaa_2_8"
CASE_NAMES = ("air_diffuse", "pipe", "water")
CASE_LABELS = {
    "air_diffuse": "air_diffuse",
    "pipe": "pipe",
    "water": "water",
}
CASE_DISPLAY = {
    "air_diffuse": "Air (diffuse)",
    "pipe": "Pipe (acrylic)",
    "water": "Water + pipe",
}
CASE_COLORS = {
    "air_diffuse": "#1f77b4",
    "pipe": "#ff7f0e",
    "water": "#2ca02c",
}

DIC3D_DIRNAME = "dic3d"
PLATE_SIMDATA_DIR = THESIS_DATA_DIR / "plate_hole/platehole3d_2mr_63f"

# Nominal camera model from application_4_fea.py:
# 50 mm focal length, 20.48 mm sensor, 1024 x 1024 full frame.
# In rtcamera.py pixel_00_center sits at -511.5 * pixel_spacing, so pixel index
# 511.5 (= NOMINAL_CX - 0.5) lies on the optical axis.
NOMINAL_FX = 50.0 / (20.48 / 1024.0)  # 2500.0 px
NOMINAL_CX = 512.0
NOMINAL_CY = 512.0
NOMINAL_PIXEL_CENTER_OFFSET = 0.5
NOMINAL_BASELINE_MM = 8.475792207426851
NOMINAL_STEREO_ANGLE_DEG = 5.0
MM_PER_PX = 40.0 / 1024.0  # 0.0390625 mm/px

# Specimen placement from application_4_fea.py
PLATE_HEIGHT_MM = 35.0
PLATE_BASE_Y_MM = -19.0  # pipe_bottom_inner_y = -23 + 4 mm

# Cylindrical pipe and refractive media parameters from application_4_fea.py
# and rtpresets.py (MaterialPresets.AIR, PLASTIC_ACRYLIC, WATER).
PIPE_OUTER_RADIUS_MM = 18.5
PIPE_INNER_RADIUS_MM = 16.5
RI_AIR = 1.000277
RI_ACRYLIC = 1.490
RI_WATER = 1.333

CMAP_DIC_FEA = "coolwarm"
CMAP_MAGNITUDE = "viridis"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "text.usetex": True,
    "font.family": "serif",
})


# ================================================================================
# Data containers
# ================================================================================

@dataclass
class FEAGroundTruth:
    """Scaled, positioned FEA mesh and displaced frame in stereo-DIC units."""
    simdata_dir: Path
    frame_idx: int
    frame_scale: float
    mesh_scale: float
    w_convention: str
    coords_mm: np.ndarray          # (N_nodes, 3) world coordinates [mm]
    connectivity: np.ndarray       # (N_cells, 8) QUAD8 node indices
    disp_all_mm: np.ndarray        # (N_nodes, 3) [u, v, w] [mm]
    front_node_mask: np.ndarray    # (N_nodes,) bool on z = z_front_mm
    front_cells: np.ndarray        # (N_front_cells, 8) QUAD8 front-face cells
    z_front_mm: float
    half_size_mm: np.ndarray       # (2,) [half_width, half_height] [mm]
    centre_world_mm: np.ndarray    # (3,) [x_c, y_c, z_front] [mm]
    hole_radius_mm: float
    triangulation: mtri.Triangulation
    tri_disp_mm: np.ndarray        # (N_tri_pts, 3) [u, v, w] for smooth plots

    @cached_property
    def sampling_grid(self) -> pv.UnstructuredGrid:
        """
        Undeformed front-face QUAD8 mesh with displacement as point data.

        The geometry is deliberately the reference configuration: VTK samples
        the nodal displacement field at the requested *undeformed* ``(x, y)``
        coordinates, matching the previous isoparametric evaluator.
        """
        n_cells, n_nodes = self.front_cells.shape
        cells = np.empty((n_cells, n_nodes + 1), dtype=np.int64)
        cells[:, 0] = n_nodes
        cells[:, 1:] = self.front_cells
        grid = pv.UnstructuredGrid(
            cells.ravel(),
            np.full(n_cells, pv.CellType.QUADRATIC_QUAD, dtype=np.uint8),
            self.coords_mm,
        )
        grid.point_data["disp_mm"] = self.disp_all_mm
        return grid


@dataclass
class DICCaseData:
    """Loaded stereo-DIC, calibration, geometry and mapped FEA data for one case."""
    case: str
    case_dir: Path
    cam0_mm: np.ndarray
    nominal_baseline_mm: float
    nominal_stereo_angle_deg: float
    calib_row: dict[str, str]
    dic_summary_row: dict[str, str]
    total_subsets: int
    converged_fraction: float
    # Arrays below hold valid (converged & stereo_converged & finite) subsets:
    subset_x: np.ndarray
    subset_y: np.ndarray
    disp_mm: np.ndarray            # (N_valid, 3) [u, v, w] [mm]
    mag_mm: np.ndarray             # (N_valid,) |d| [mm]
    stereo_cam0_mm: np.ndarray     # (N_valid, 3) reconstructed cam0 coords [mm]
    stereo_world_mm: np.ndarray    # (N_valid, 3) reconstructed world coords [mm]
    cost_zncc: np.ndarray
    stereo_cost_zncc: np.ndarray
    epi_dist_px: np.ndarray
    xy_refracted_mm: np.ndarray    # (N_valid, 2) true plate (x, y) [mm]
    xy_nominal_mm: np.ndarray      # (N_valid, 2) nominal pinhole (x, y) [mm]
    fea_refracted_mm: np.ndarray   # (N_valid, 3) FEA [u, v, w] at xy_refracted
    fea_nominal_mm: np.ndarray     # (N_valid, 3) FEA [u, v, w] at xy_nominal
    # Structured 2D grid over Camera 0's subset lattice for gap-free plotting:
    grid_x_refracted_mm: np.ndarray
    grid_y_refracted_mm: np.ndarray
    grid_disp_mm: np.ndarray       # (Ny, Nx, 3) [u, v, w] with NaN outside ROI
    grid_fea_mm: np.ndarray        # (Ny, Nx, 3) FEA [u, v, w] on subset lattice


# ================================================================================
# Quadratic QUAD8 plotting helpers and FEA front-face sampling
# ================================================================================

def _quad8_shape_and_derivs(
    xi: np.ndarray,
    eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate standard 8-node quadrilateral shape functions and
    their derivatives on [-1, 1]^2.

    Node ordering matches VTK / Exodus / meshio ``QUAD8``:
    corners 0:(-1,-1), 1:(+1,-1), 2:(+1,+1), 3:(-1,+1);
    mid-edges 4:(0,-1), 5:(+1,0), 6:(0,+1), 7:(-1,0).
    """
    xm, xp = 1.0 - xi, 1.0 + xi
    em, ep = 1.0 - eta, 1.0 + eta
    x2, e2 = 1.0 - xi * xi, 1.0 - eta * eta

    shape = np.stack([
        0.25 * xm * em * (-xi - eta - 1.0),
        0.25 * xp * em * ( xi - eta - 1.0),
        0.25 * xp * ep * ( xi + eta - 1.0),
        0.25 * xm * ep * (-xi + eta - 1.0),
        0.50 * x2 * em,
        0.50 * xp * e2,
        0.50 * x2 * ep,
        0.50 * xm * e2,
    ], axis=-1)

    d_xi = np.stack([
        0.25 * em * (2.0 * xi + eta),
        0.25 * em * (2.0 * xi - eta),
        0.25 * ep * (2.0 * xi + eta),
        0.25 * ep * (2.0 * xi - eta),
        -xi * em,
        0.50 * e2,
        -xi * ep,
        -0.50 * e2,
    ], axis=-1)

    d_eta = np.stack([
        0.25 * xm * (xi + 2.0 * eta),
        0.25 * xp * (-xi + 2.0 * eta),
        0.25 * xp * (xi + 2.0 * eta),
        0.25 * xm * (-xi + 2.0 * eta),
        -0.50 * x2,
        -xp * eta,
        0.50 * x2,
        -xm * eta,
    ], axis=-1)
    return shape, d_xi, d_eta


def _subdivide_quad8_surface(
    coords_mm: np.ndarray,
    front_cells: np.ndarray,
    disp_mm: np.ndarray,
    subdivisions: int = 5,
) -> tuple[mtri.Triangulation, np.ndarray]:
    """
    Subdivide each front-face ``QUAD8`` cell using its exact quadratic shape
    functions to produce a smooth triangulation (including curved hole edges).
    """
    lin = np.linspace(-1.0, 1.0, subdivisions + 1)
    xi_grid, eta_grid = np.meshgrid(lin, lin, indexing="xy")
    xi_flat = xi_grid.ravel()
    eta_flat = eta_grid.ravel()
    shape, _, _ = _quad8_shape_and_derivs(xi_flat, eta_flat)  # (P, 8)

    cell_xy = coords_mm[front_cells, :2]    # (C, 8, 2)
    cell_uvw = disp_mm[front_cells]         # (C, 8, 3)

    # Evaluate (C, P, 2) and (C, P, 3)
    pts_xy = np.einsum("pa,cad->cpd", shape, cell_xy).reshape(-1, 2)
    pts_uvw = np.einsum("pa,cad->cpd", shape, cell_uvw).reshape(-1, 3)

    stride = subdivisions + 1
    local_tris: list[list[int]] = []
    for row in range(subdivisions):
        for col in range(subdivisions):
            n0 = row * stride + col
            n1 = n0 + 1
            n2 = n1 + stride
            n3 = n0 + stride
            local_tris.append([n0, n1, n2])
            local_tris.append([n0, n2, n3])
    local_tris_arr = np.asarray(local_tris, dtype=np.int64)  # (T_sub, 3)

    pts_per_cell = stride * stride
    offsets = (np.arange(len(front_cells), dtype=np.int64) * pts_per_cell)[:, None, None]
    all_tris = (local_tris_arr[None, :, :] + offsets).reshape(-1, 3)

    triangulation = mtri.Triangulation(pts_xy[:, 0], pts_xy[:, 1], all_tris)
    return triangulation, pts_uvw


def evaluate_fea_front_face(
    fea: FEAGroundTruth,
    xy_query_mm: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the quadratic ``QUAD8`` FEA displacement field ``[u, v, w]`` [mm]
    at arbitrary ``(x, y)`` coordinates on the camera-facing front face.

    VTK/PyVista performs the high-order cell location and shape-function
    interpolation via :meth:`pyvista.DataSetFilters.sample`.  The mesh is the
    undeformed front face.

    Points outside the plate or inside the central circular hole return ``NaN``.
    """
    xy = np.asarray(xy_query_mm, dtype=np.float64)
    n_pts = xy.shape[0]
    result = np.full((n_pts, 3), np.nan, dtype=np.float64)
    if n_pts == 0:
        return result

    finite = np.all(np.isfinite(xy), axis=1)
    active = np.flatnonzero(finite)
    if active.size == 0:
        return result

    query_xy = xy[active]
    query = pv.PolyData(
        np.column_stack([
            query_xy,
            np.full(query_xy.shape[0], fea.z_front_mm, dtype=np.float64),
        ])
    )
    sampled = query.sample(
        fea.sampling_grid,
        tolerance=1e-6,
        locator="static_cell",
        pass_cell_data=False,
        pass_field_data=False,
    )
    values = np.asarray(sampled.point_data["disp_mm"], dtype=np.float64)
    valid = np.asarray(sampled.point_data["vtkValidPointMask"], dtype=bool)
    result[active[valid]] = values[valid]
    return result


def load_fea_ground_truth(
    simdata_dir: Path = PLATE_SIMDATA_DIR,
    forced_frame_idx: int | None = None,
    w_convention: str = "cam0",
) -> FEAGroundTruth:
    """
    Reconstruct the exact scaled, positioned FEA mesh and displaced frame from
    ``application_4_fea.py`` without importing ``pyvale.raytracer``.
    """
    simdata_dir = Path(simdata_dir)
    coords_m = np.loadtxt(simdata_dir / "coords.csv", delimiter=",", dtype=np.float64)
    connect_path = simdata_dir / "connect.csv"
    if not connect_path.is_file():
        connect_path = simdata_dir / "connectivity.csv"
    connectivity = np.loadtxt(connect_path, delimiter=",", dtype=np.int64)
    if connectivity.ndim == 1:
        connectivity = connectivity[None, :]

    disp_x_m = np.atleast_2d(
        np.loadtxt(simdata_dir / "field_disp_x.csv", delimiter=",", dtype=np.float64)
    )
    disp_y_m = np.atleast_2d(
        np.loadtxt(simdata_dir / "field_disp_y.csv", delimiter=",", dtype=np.float64)
    )
    disp_z_m = np.atleast_2d(
        np.loadtxt(simdata_dir / "field_disp_z.csv", delimiter=",", dtype=np.float64)
    )
    # Shape (T, N_nodes, 3)
    disp_m = np.stack([disp_x_m, disp_y_m, disp_z_m], axis=-1).transpose(1, 0, 2)

    # 1. Scale coordinates from m to mm so y-span equals PLATE_HEIGHT_MM (35 mm)
    y_span_m = float(coords_m[:, 1].max() - coords_m[:, 1].min())
    if y_span_m <= 0.0:
        raise RuntimeError(f"Degenerate plate y span in {simdata_dir}")
    mesh_scale = PLATE_HEIGHT_MM / y_span_m
    coords_mm = coords_m * mesh_scale

    # 2. Place with Anchor.BASE at (0, PLATE_BASE_Y_MM, 0)
    bbox_min = coords_mm.min(axis=0)
    bbox_max = coords_mm.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    anchor_base = np.array([bbox_center[0], bbox_min[1], bbox_center[2]])
    target_pos = np.array([0.0, PLATE_BASE_Y_MM, 0.0])
    coords_mm = coords_mm + (target_pos - anchor_base)

    # 3. Rescale nodal displacements to mm and select frame <= 1 px (application_4_fea.py)
    node_displacements = (disp_m - disp_m[0:1]) * mesh_scale
    node_displacements[0] = 0.0

    if forced_frame_idx is None:
        displaced_frame_idx = 0
        max_displacement_mm = 0.0
        for t in range(1, node_displacements.shape[0]):
            curr_max_mm = float(np.max(node_displacements[t]))
            if MM_PER_PX >= curr_max_mm > max_displacement_mm:
                max_displacement_mm = curr_max_mm
                displaced_frame_idx = t
    else:
        displaced_frame_idx = int(forced_frame_idx)
        max_displacement_mm = float(np.max(node_displacements[displaced_frame_idx]))

    frame_scale = 1.0
    frame_disp_world_mm = node_displacements[displaced_frame_idx].copy()
    if 0.0 < max_displacement_mm <= 0.5 * MM_PER_PX:
        frame_scale = MM_PER_PX / max_displacement_mm
        frame_disp_world_mm = frame_disp_world_mm * frame_scale

    # 4. Convert to stereo-DIC Camera 0 displacement components:
    #    u = +dx_world, v = -dy_world (down),
    #    w = -dz_world (cam0 optical axis +Z points away from camera) or +dz_world
    disp_dic_mm = frame_disp_world_mm.copy()
    disp_dic_mm[:, 1] *= -1.0
    if w_convention == "cam0":
        disp_dic_mm[:, 2] *= -1.0
    elif w_convention != "world":
        raise ValueError(f"Unsupported w_convention: {w_convention}")

    # 5. Extract camera-facing front face (z = z_front_mm = +0.5 mm)
    z_front_mm = float(coords_mm[:, 2].max())
    front_node_mask = np.isclose(coords_mm[:, 2], z_front_mm)
    front_cells = connectivity[
        np.all(np.isclose(coords_mm[connectivity, 2], z_front_mm), axis=1)
    ]
    if front_cells.size == 0:
        raise RuntimeError(f"No front-face cells found at z={z_front_mm:.4f} mm")

    xy_min = coords_mm[front_node_mask, :2].min(axis=0)
    xy_max = coords_mm[front_node_mask, :2].max(axis=0)
    half_size_mm = np.round(0.5 * (xy_max - xy_min), 9)
    centre_xy = 0.5 * (xy_min + xy_max)
    hole_radius_mm = float(np.round(
        np.linalg.norm(coords_mm[front_node_mask, :2] - centre_xy, axis=1).min(),
        9,
    ))
    centre_world_mm = np.array([centre_xy[0], centre_xy[1], z_front_mm], dtype=np.float64)

    triangulation, tri_disp_mm = _subdivide_quad8_surface(
        coords_mm, front_cells, disp_dic_mm, subdivisions=5
    )

    return FEAGroundTruth(
        simdata_dir=simdata_dir,
        frame_idx=displaced_frame_idx,
        frame_scale=frame_scale,
        mesh_scale=mesh_scale,
        w_convention=w_convention,
        coords_mm=coords_mm,
        connectivity=connectivity,
        disp_all_mm=disp_dic_mm,
        front_node_mask=front_node_mask,
        front_cells=front_cells,
        z_front_mm=z_front_mm,
        half_size_mm=half_size_mm,
        centre_world_mm=centre_world_mm,
        hole_radius_mm=hole_radius_mm,
        triangulation=triangulation,
        tri_disp_mm=tri_disp_mm,
    )


# ================================================================================
# Analytical camera back-projection (nominal pinhole and cylindrical Snell's law)
# ================================================================================

def _intersect_and_refract_cylinder(
    origins: np.ndarray,
    directions: np.ndarray,
    radius_mm: float,
    n_incident: float,
    n_transmitted: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intersect rays with the front (+z) half of a vertical cylinder
    $x^2 + z^2 = R^2$ and refract their directions via 3D Snell's law.
    """
    dx = directions[:, 0]
    dz = directions[:, 2]
    ox = origins[:, 0]
    oz = origins[:, 2]

    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius_mm * radius_mm
    disc = b * b - 4.0 * a * c

    hit_points = np.full_like(origins, np.nan)
    new_dirs = np.full_like(directions, np.nan)

    valid = (a > 1e-15) & (disc >= 0.0)
    if not np.any(valid):
        return hit_points, new_dirs

    sqrt_disc = np.sqrt(np.maximum(disc[valid], 0.0))
    # Camera sits outside the cylinder looking inward: nearest entry root is (-b - sqrt_disc)/(2a)
    t = (-b[valid] - sqrt_disc) / (2.0 * a[valid])
    pts = origins[valid] + t[:, None] * directions[valid]

    # Outward cylinder normal n = (x/R, 0, z/R) opposes incoming ray (dot(d, n) < 0)
    normals = np.zeros_like(pts)
    normals[:, 0] = pts[:, 0] / radius_mm
    normals[:, 2] = pts[:, 2] / radius_mm

    cos_i = -np.einsum("ij,ij->i", directions[valid], normals)
    eta = n_incident / n_transmitted
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)

    refr_ok = k >= 0.0
    if np.any(refr_ok):
        refracted = (
            eta * directions[valid][refr_ok]
            + (eta * cos_i[refr_ok] - np.sqrt(k[refr_ok]))[:, None] * normals[refr_ok]
        )
        refracted /= np.linalg.norm(refracted, axis=1, keepdims=True)

        valid_indices = np.flatnonzero(valid)[refr_ok]
        hit_points[valid_indices] = pts[refr_ok]
        new_dirs[valid_indices] = refracted

    return hit_points, new_dirs


def project_cam0_pixels_to_plate(
    px: np.ndarray,
    py: np.ndarray,
    cam0_mm: np.ndarray,
    case: str,
    z_plate_mm: float = 0.5,
    through_media: bool = True,
) -> np.ndarray:
    """
    Back-project Camera 0 pixel coordinates ``(px, py)`` onto the specimen
    front plane ``z = z_plate_mm``, returning ``(N, 2)`` world ``(x, y)`` [mm].

    When ``through_media=True``, rays for ``pipe`` and ``water`` are refracted
    across the outer ($R = 18.5$ mm) and inner ($R = 16.5$ mm) cylindrical
    acrylic walls using exact 3D Snell's law, matching the ray-tracer physics.
    When ``through_media=False``, straight pinhole rays in air are used.
    """
    px_arr = np.asarray(px, dtype=np.float64).ravel()
    py_arr = np.asarray(py, dtype=np.float64).ravel()

    # Camera 0 looks straight along -Z_world with pixel_00_center at -511.5 * spacing
    cx_opt = NOMINAL_CX - NOMINAL_PIXEL_CENTER_OFFSET
    cy_opt = NOMINAL_CY - NOMINAL_PIXEL_CENTER_OFFSET
    dir_x = (px_arr - cx_opt) / NOMINAL_FX
    dir_y = -(py_arr - cy_opt) / NOMINAL_FX
    dir_z = -np.ones_like(dir_x)

    directions = np.column_stack([dir_x, dir_y, dir_z])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.broadcast_to(np.asarray(cam0_mm, dtype=np.float64), directions.shape).copy()

    if through_media and case in ("pipe", "water"):
        origins, directions = _intersect_and_refract_cylinder(
            origins, directions, PIPE_OUTER_RADIUS_MM, RI_AIR, RI_ACRYLIC
        )
        inner_ri = RI_WATER if case == "water" else RI_AIR
        origins, directions = _intersect_and_refract_cylinder(
            origins, directions, PIPE_INNER_RADIUS_MM, RI_ACRYLIC, inner_ri
        )

    t_plate = (z_plate_mm - origins[:, 2]) / directions[:, 2]
    hit_xy = origins[:, :2] + t_plate[:, None] * directions[:, :2]
    hit_xy[t_plate <= 0.0] = np.nan
    return hit_xy


# ================================================================================
# Loading stereo-DIC case outputs
# ================================================================================

def _read_single_row_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with open(path, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            return dict(row)
    return {}


def _load_stereo_geometry(case_dir: Path) -> tuple[np.ndarray, float, float]:
    geom_path = case_dir / "stereo_geometry.csv"
    if not geom_path.is_file():
        raise FileNotFoundError(f"Missing stereo_geometry.csv in {case_dir}")

    kv: dict[str, float] = {}
    with open(geom_path, newline="") as file:
        for row in csv.reader(file):
            if len(row) >= 2:
                kv[row[0].strip()] = float(row[1])

    cam0 = np.array([kv["cam0_x_mm"], kv["cam0_y_mm"], kv["cam0_z_mm"]], dtype=np.float64)
    baseline = float(kv.get("baseline_len_mm", NOMINAL_BASELINE_MM))
    angle = float(kv.get("stereo_angle_deg", NOMINAL_STEREO_ANGLE_DEG))
    return cam0, baseline, angle


def load_dic_case(case: str, case_dir: Path, fea: FEAGroundTruth) -> DICCaseData:
    """Load one stereo-DIC case and evaluate FEA ground truth at its subsets."""
    dic_dir = case_dir / DIC3D_DIRNAME
    deformed_csv = dic_dir / "dic3d_rtimage_1_cam0.csv"
    if not deformed_csv.is_file():
        raise FileNotFoundError(
            f"Missing deformed stereo-DIC CSV: {deformed_csv}. "
            "Run application_4_fea_dic.py first."
        )

    cam0_mm, nom_baseline, nom_angle = _load_stereo_geometry(case_dir)
    calib_row = _read_single_row_csv(dic_dir / "calibration_summary.csv")
    dic_summary_row = _read_single_row_csv(dic_dir / "dic3d_summary.csv")

    raw = np.genfromtxt(deformed_csv, delimiter=",", names=True)
    if raw.dtype.names is None:
        raise RuntimeError(f"Failed to parse header in {deformed_csv}")

    def col(name: str) -> np.ndarray:
        return np.asarray(raw[name], dtype=np.float64).ravel()

    sx_all = col("subset_x")
    sy_all = col("subset_y")
    u_all = col("stereo_disp_u_mm")
    v_all = col("stereo_disp_v_mm")
    w_all = col("stereo_disp_w_mm")
    if fea.w_convention == "world":
        # Pyvale stereo_disp_w_mm is +dz_cam0 = -dz_world; flip if comparing in +z_world
        w_all = -w_all

    x_cam0_all = col("stereo_x_mm")
    y_cam0_all = col("stereo_y_mm")
    z_cam0_all = col("stereo_z_mm")

    converged_mask = (
        (col("converged").astype(int) == 1)
        & (col("stereo_converged").astype(int) == 1)
    )
    finite_mask = (
        np.isfinite(u_all)
        & np.isfinite(v_all)
        & np.isfinite(w_all)
        & np.isfinite(x_cam0_all)
        & np.isfinite(y_cam0_all)
        & np.isfinite(z_cam0_all)
    )
    valid = converged_mask & finite_mask
    if not np.any(valid):
        raise RuntimeError(f"No valid converged stereo-DIC subsets in {deformed_csv}")

    sx = sx_all[valid]
    sy = sy_all[valid]
    disp_mm = np.column_stack([u_all[valid], v_all[valid], w_all[valid]])
    mag_mm = np.linalg.norm(disp_mm, axis=1)

    stereo_cam0 = np.column_stack([
        x_cam0_all[valid], y_cam0_all[valid], z_cam0_all[valid]
    ])
    # Transform Camera 0 coordinates (+x right, +y down, +z away) to world frame
    stereo_world = np.column_stack([
        cam0_mm[0] + stereo_cam0[:, 0],
        cam0_mm[1] - stereo_cam0[:, 1],
        cam0_mm[2] - stereo_cam0[:, 2],
    ])

    xy_refracted = project_cam0_pixels_to_plate(
        sx, sy, cam0_mm, case, z_plate_mm=fea.z_front_mm, through_media=True
    )
    xy_nominal = project_cam0_pixels_to_plate(
        sx, sy, cam0_mm, case, z_plate_mm=fea.z_front_mm, through_media=False
    )

    fea_refracted = evaluate_fea_front_face(fea, xy_refracted)
    fea_nominal = evaluate_fea_front_face(fea, xy_nominal)

    # Build structured 2D grid over subset lattice for clean pcolormesh rendering
    unique_dx = np.diff(np.unique(sx))
    unique_dy = np.diff(np.unique(sy))
    step_x = float(np.median(unique_dx)) if unique_dx.size else 5.0
    step_y = float(np.median(unique_dy)) if unique_dy.size else 5.0
    xs_grid = np.arange(sx.min(), sx.max() + 0.5 * step_x, step_x)
    ys_grid = np.arange(sy.min(), sy.max() + 0.5 * step_y, step_y)
    px_mesh, py_mesh = np.meshgrid(xs_grid, ys_grid, indexing="xy")

    grid_xy_ref = project_cam0_pixels_to_plate(
        px_mesh.ravel(), py_mesh.ravel(), cam0_mm, case,
        z_plate_mm=fea.z_front_mm, through_media=True,
    )
    grid_x_ref = grid_xy_ref[:, 0].reshape(px_mesh.shape)
    grid_y_ref = grid_xy_ref[:, 1].reshape(px_mesh.shape)

    grid_disp = np.full((*px_mesh.shape, 3), np.nan, dtype=np.float64)
    grid_fea = np.full((*px_mesh.shape, 3), np.nan, dtype=np.float64)
    ix = np.rint((sx - xs_grid[0]) / step_x).astype(int)
    iy = np.rint((sy - ys_grid[0]) / step_y).astype(int)
    in_bounds = (
        (ix >= 0) & (ix < len(xs_grid))
        & (iy >= 0) & (iy < len(ys_grid))
    )
    grid_disp[iy[in_bounds], ix[in_bounds]] = disp_mm[in_bounds]
    grid_fea[iy[in_bounds], ix[in_bounds]] = fea_refracted[in_bounds]

    return DICCaseData(
        case=case,
        case_dir=case_dir,
        cam0_mm=cam0_mm,
        nominal_baseline_mm=nom_baseline,
        nominal_stereo_angle_deg=nom_angle,
        calib_row=calib_row,
        dic_summary_row=dic_summary_row,
        total_subsets=int(sx_all.size),
        converged_fraction=float(np.mean(converged_mask)),
        subset_x=sx,
        subset_y=sy,
        disp_mm=disp_mm,
        mag_mm=mag_mm,
        stereo_cam0_mm=stereo_cam0,
        stereo_world_mm=stereo_world,
        cost_zncc=col("cost_zncc")[valid],
        stereo_cost_zncc=col("stereo_cost_zncc")[valid],
        epi_dist_px=col("epi_dist_px")[valid],
        xy_refracted_mm=xy_refracted,
        xy_nominal_mm=xy_nominal,
        fea_refracted_mm=fea_refracted,
        fea_nominal_mm=fea_nominal,
        grid_x_refracted_mm=grid_x_ref,
        grid_y_refracted_mm=grid_y_ref,
        grid_disp_mm=grid_disp,
        grid_fea_mm=grid_fea,
    )


# ================================================================================
# Quantitative statistics and pairwise grid interpolation
# ================================================================================

COMPONENTS = (
    ("u", 0, r"$u$"),
    ("v", 1, r"$v$"),
    ("w", 2, r"$w$"),
    ("magnitude", 3, r"$|\mathbf{d}|$"),
)


def _extract_component(disp_uvw: np.ndarray, comp_idx: int) -> np.ndarray:
    if comp_idx < 3:
        return disp_uvw[:, comp_idx]
    return np.linalg.norm(disp_uvw, axis=1)


def _distribution_row(
    dataset: str,
    component: str,
    values_mm: np.ndarray,
    total_points: int,
    converged_fraction: float,
) -> dict[str, object]:
    finite = values_mm[np.isfinite(values_mm)]
    if finite.size == 0:
        return {
            "dataset": dataset,
            "component": component,
            "points": total_points,
            "valid_points": 0,
            "converged_fraction": converged_fraction,
            "median_mm": np.nan,
            "mean_mm": np.nan,
            "std_mm": np.nan,
            "min_mm": np.nan,
            "max_mm": np.nan,
            "p05_mm": np.nan,
            "p95_mm": np.nan,
            "max_abs_px": np.nan,
        }
    p05, p95 = np.percentile(finite, [5.0, 95.0])
    max_abs = float(np.max(np.abs(finite)))
    return {
        "dataset": dataset,
        "component": component,
        "points": total_points,
        "valid_points": int(finite.size),
        "converged_fraction": converged_fraction,
        "median_mm": float(np.median(finite)),
        "mean_mm": float(np.mean(finite)),
        "std_mm": float(np.std(finite)),
        "min_mm": float(np.min(finite)),
        "max_mm": float(np.max(finite)),
        "p05_mm": float(p05),
        "p95_mm": float(p95),
        "max_abs_px": max_abs / MM_PER_PX,
    }


def _error_metrics_row(
    label_fields: dict[str, object],
    diff_mm: np.ndarray,
    ref_scale_mm: float,
) -> dict[str, object]:
    finite = diff_mm[np.isfinite(diff_mm)]
    row = dict(label_fields)
    row["n_matched"] = int(finite.size)
    if finite.size == 0:
        for key in (
            "bias_mm", "bias_um", "mae_mm", "mae_um", "rmse_mm", "rmse_um",
            "std_mm", "std_um", "median_abs_um", "p95_abs_um", "max_abs_um",
            "rmse_px", "max_abs_px", "nrmse_pct",
        ):
            row[key] = np.nan
        return row

    abs_err = np.abs(finite)
    bias_mm = float(np.mean(finite))
    mae_mm = float(np.mean(abs_err))
    rmse_mm = float(np.sqrt(np.mean(finite * finite)))
    std_mm = float(np.std(finite))
    med_abs_mm = float(np.median(abs_err))
    p95_abs_mm = float(np.percentile(abs_err, 95.0))
    max_abs_mm = float(np.max(abs_err))

    row.update({
        "bias_mm": bias_mm,
        "bias_um": bias_mm * 1e3,
        "mae_mm": mae_mm,
        "mae_um": mae_mm * 1e3,
        "rmse_mm": rmse_mm,
        "rmse_um": rmse_mm * 1e3,
        "std_mm": std_mm,
        "std_um": std_mm * 1e3,
        "median_abs_um": med_abs_mm * 1e3,
        "p95_abs_um": p95_abs_mm * 1e3,
        "max_abs_um": max_abs_mm * 1e3,
        "rmse_px": rmse_mm / MM_PER_PX,
        "max_abs_px": max_abs_mm / MM_PER_PX,
        "nrmse_pct": 100.0 * rmse_mm / ref_scale_mm if ref_scale_mm > 0.0 else np.nan,
    })
    return row


def interpolate_dic_to_points(
    dic_case: DICCaseData,
    xy_query_mm: np.ndarray,
    max_neighbor_dist_mm: float = 0.28,
) -> np.ndarray:
    """
    Interpolate a stereo-DIC case's ``[u, v, w]`` field onto arbitrary physical
    plate points ``xy_query_mm`` [mm] using piecewise-linear triangulation of
    its refracted subset coordinates, masked by nearest-subset distance so the
    central hole and exterior margins are never bridged across gaps.
    """
    src_xy = dic_case.xy_refracted_mm
    valid_src = np.all(np.isfinite(src_xy), axis=1) & np.all(
        np.isfinite(dic_case.disp_mm), axis=1
    )
    src_xy = src_xy[valid_src]
    src_uvw = dic_case.disp_mm[valid_src]

    out = np.full((len(xy_query_mm), 3), np.nan, dtype=np.float64)
    if len(src_xy) < 4 or len(xy_query_mm) == 0:
        return out

    finite_q = np.all(np.isfinite(xy_query_mm), axis=1)
    if not np.any(finite_q):
        return out

    tree = cKDTree(src_xy)
    dists, _ = tree.query(xy_query_mm[finite_q], k=1)
    close_mask = dists <= max_neighbor_dist_mm
    if not np.any(close_mask):
        return out

    query_idx = np.flatnonzero(finite_q)[close_mask]
    interpolator = LinearNDInterpolator(src_xy, src_uvw, fill_value=np.nan)
    out[query_idx] = interpolator(xy_query_mm[query_idx])
    return out


def build_shared_plate_grid(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    step_mm: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Construct a regular physical ``(x, y)`` grid [mm] over the specimen front
    face and evaluate FEA as well as every available DIC case on it.
    """
    cx, cy = fea.centre_world_mm[0], fea.centre_world_mm[1]
    hx, hy = fea.half_size_mm[0], fea.half_size_mm[1]

    xs = np.arange(cx - hx, cx + hx + 0.5 * step_mm, step_mm)
    ys = np.arange(cy - hy, cy + hy + 0.5 * step_mm, step_mm)
    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    xy_flat = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    fea_grid = evaluate_fea_front_face(fea, xy_flat).reshape((*x_grid.shape, 3))
    dic_grids: dict[str, np.ndarray] = {}
    for name, case_data in dic_cases.items():
        dic_grids[name] = interpolate_dic_to_points(case_data, xy_flat).reshape(
            (*x_grid.shape, 3)
        )
    return x_grid, y_grid, fea_grid, dic_grids


def _write_dict_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_and_export_tables(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    fea_grid: np.ndarray,
    dic_grids: dict[str, np.ndarray],
    out_dir: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Compute all comparison tables and write them as CSV files."""
    # Common ROI mask on shared plate grid where FEA and all DIC cases are valid
    common_valid = np.all(np.isfinite(fea_grid), axis=-1)
    for grid in dic_grids.values():
        common_valid &= np.all(np.isfinite(grid), axis=-1)

    fea_front_uvw = fea.disp_all_mm[fea.front_node_mask]
    fea_common_uvw = fea_grid[common_valid] if np.any(common_valid) else np.empty((0, 3))

    # 1. Field distribution summary table
    summary_rows: list[dict[str, object]] = []
    for comp_name, comp_idx, _ in COMPONENTS:
        summary_rows.append(_distribution_row(
            "fea_all_nodes", comp_name,
            _extract_component(fea.disp_all_mm, comp_idx),
            len(fea.disp_all_mm), 1.0,
        ))
        summary_rows.append(_distribution_row(
            "fea_front_face", comp_name,
            _extract_component(fea_front_uvw, comp_idx),
            len(fea_front_uvw), 1.0,
        ))
        if len(fea_common_uvw) > 0:
            summary_rows.append(_distribution_row(
                "fea_common_roi", comp_name,
                _extract_component(fea_common_uvw, comp_idx),
                int(np.sum(common_valid)), 1.0,
            ))
        for case_name, cdata in dic_cases.items():
            summary_rows.append(_distribution_row(
                case_name, comp_name,
                _extract_component(cdata.disp_mm, comp_idx),
                cdata.total_subsets, cdata.converged_fraction,
            ))
    _write_dict_rows_csv(out_dir / "field_summary_comparison.csv", summary_rows)

    # Reference scale per component (max |FEA| on front face) for NRMSE%
    fea_ref_scales: dict[str, float] = {}
    for comp_name, comp_idx, _ in COMPONENTS:
        vals = _extract_component(fea_front_uvw, comp_idx)
        fea_ref_scales[comp_name] = float(np.nanmax(np.abs(vals)))

    # 2. Pointwise DIC - FEA error table (both refracted and nominal mappings)
    dic_vs_fea_rows: list[dict[str, object]] = []
    for case_name, cdata in dic_cases.items():
        for mapping_name, fea_mapped in (
            ("refracted", cdata.fea_refracted_mm),
            ("nominal", cdata.fea_nominal_mm),
        ):
            valid_fea = np.all(np.isfinite(fea_mapped), axis=1)
            dic_sub = cdata.disp_mm[valid_fea]
            fea_sub = fea_mapped[valid_fea]
            for comp_name, comp_idx, _ in COMPONENTS:
                diff = (
                    _extract_component(dic_sub, comp_idx)
                    - _extract_component(fea_sub, comp_idx)
                )
                dic_vs_fea_rows.append(_error_metrics_row(
                    {
                        "case": case_name,
                        "mapping": mapping_name,
                        "component": comp_name,
                    },
                    diff,
                    fea_ref_scales[comp_name],
                ))
    _write_dict_rows_csv(out_dir / "dic_vs_fea_errors.csv", dic_vs_fea_rows)

    # 3. Pairwise DIC - DIC differences on shared physical grid
    pair_specs = [
        ("pipe", "air_diffuse"),
        ("water", "air_diffuse"),
        ("water", "pipe"),
    ]
    intercase_rows: list[dict[str, object]] = []
    for case_a, case_b in pair_specs:
        if case_a not in dic_grids or case_b not in dic_grids:
            continue
        grid_a = dic_grids[case_a]
        grid_b = dic_grids[case_b]
        overlap = np.all(np.isfinite(grid_a), axis=-1) & np.all(
            np.isfinite(grid_b), axis=-1
        )
        sub_a = grid_a[overlap]
        sub_b = grid_b[overlap]
        for comp_name, comp_idx, _ in COMPONENTS:
            diff = (
                _extract_component(sub_a, comp_idx)
                - _extract_component(sub_b, comp_idx)
            )
            intercase_rows.append(_error_metrics_row(
                {
                    "comparison": f"{case_a} - {case_b}",
                    "case_a": case_a,
                    "case_b": case_b,
                    "component": comp_name,
                },
                diff,
                fea_ref_scales[comp_name],
            ))
    _write_dict_rows_csv(out_dir / "dic_intercase_errors.csv", intercase_rows)

    # 4. Calibration, correlation quality & 3D surface reconstruction table
    calib_shape_rows: list[dict[str, object]] = [{
        "case": "nominal_rig",
        "quality_passed": 1,
        "fx0_px": NOMINAL_FX,
        "fy0_px": NOMINAL_FX,
        "cx0_px": NOMINAL_CX - NOMINAL_PIXEL_CENTER_OFFSET,
        "cy0_px": NOMINAL_CY - NOMINAL_PIXEL_CENTER_OFFSET,
        "fx1_px": NOMINAL_FX,
        "fy1_px": NOMINAL_FX,
        "baseline_mm": NOMINAL_BASELINE_MM,
        "ry_deg": NOMINAL_STEREO_ANGLE_DEG,
        "reproj_mean0_px": 0.0,
        "reproj_mean1_px": 0.0,
        "subsets": 0,
        "converged_fraction": 1.0,
        "zncc_median": 1.0,
        "stereo_zncc_median": 1.0,
        "epi_dist_median_px": 0.0,
        "epi_dist_p95_px": 0.0,
        "z_world_median_mm": fea.z_front_mm,
        "z_world_min_mm": fea.z_front_mm,
        "z_world_max_mm": fea.z_front_mm,
        "z_world_ptp_mm": 0.0,
        "z_err_bias_mm": 0.0,
        "z_err_rmse_mm": 0.0,
        "x_err_bias_mm": 0.0,
        "x_err_rmse_mm": 0.0,
        "y_err_bias_mm": 0.0,
        "y_err_rmse_mm": 0.0,
        "lateral_refraction_max_mm": 0.0,
    }]

    def _fval(row: dict[str, str], key: str) -> float:
        val = row.get(key, "")
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    for case_name, cdata in dic_cases.items():
        crow = cdata.calib_row
        xw_err = cdata.stereo_world_mm[:, 0] - cdata.xy_refracted_mm[:, 0]
        yw_err = cdata.stereo_world_mm[:, 1] - cdata.xy_refracted_mm[:, 1]
        zw = cdata.stereo_world_mm[:, 2]
        zw_err = zw - fea.z_front_mm
        lat_shift = np.hypot(
            cdata.xy_nominal_mm[:, 0] - cdata.xy_refracted_mm[:, 0],
            cdata.xy_nominal_mm[:, 1] - cdata.xy_refracted_mm[:, 1],
        )

        calib_shape_rows.append({
            "case": case_name,
            "quality_passed": int(_fval(crow, "quality_passed"))
            if np.isfinite(_fval(crow, "quality_passed")) else -1,
            "fx0_px": _fval(crow, "fx0_px"),
            "fy0_px": _fval(crow, "fy0_px"),
            "cx0_px": _fval(crow, "cx0_px"),
            "cy0_px": _fval(crow, "cy0_px"),
            "fx1_px": _fval(crow, "fx1_px"),
            "fy1_px": _fval(crow, "fy1_px"),
            "baseline_mm": _fval(crow, "baseline_mm"),
            "ry_deg": _fval(crow, "ry_deg"),
            "reproj_mean0_px": _fval(crow, "reproj_mean0_px"),
            "reproj_mean1_px": _fval(crow, "reproj_mean1_px"),
            "subsets": cdata.total_subsets,
            "converged_fraction": cdata.converged_fraction,
            "zncc_median": float(np.nanmedian(cdata.cost_zncc)),
            "stereo_zncc_median": float(np.nanmedian(cdata.stereo_cost_zncc)),
            "epi_dist_median_px": float(np.nanmedian(cdata.epi_dist_px)),
            "epi_dist_p95_px": float(np.nanpercentile(cdata.epi_dist_px, 95.0)),
            "z_world_median_mm": float(np.nanmedian(zw)),
            "z_world_min_mm": float(np.nanmin(zw)),
            "z_world_max_mm": float(np.nanmax(zw)),
            "z_world_ptp_mm": float(np.nanmax(zw) - np.nanmin(zw)),
            "z_err_bias_mm": float(np.nanmean(zw_err)),
            "z_err_rmse_mm": float(np.sqrt(np.nanmean(zw_err * zw_err))),
            "x_err_bias_mm": float(np.nanmean(xw_err)),
            "x_err_rmse_mm": float(np.sqrt(np.nanmean(xw_err * xw_err))),
            "y_err_bias_mm": float(np.nanmean(yw_err)),
            "y_err_rmse_mm": float(np.sqrt(np.nanmean(yw_err * yw_err))),
            "lateral_refraction_max_mm": float(np.nanmax(lat_shift)),
        })
    _write_dict_rows_csv(
        out_dir / "calibration_and_shape_comparison.csv", calib_shape_rows
    )

    return summary_rows, dic_vs_fea_rows, intercase_rows, calib_shape_rows


# ================================================================================
# Plotting helpers and thesis-ready comparison figures
# ================================================================================

def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a matplotlib figure, falling back to mathtext if LaTeX fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except RuntimeError:
        plt.rcParams["text.usetex"] = False
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def _draw_plate_outline(ax: plt.Axes, fea: FEAGroundTruth) -> None:
    """Overlay the physical plate perimeter and circular hole boundary."""
    cx, cy = fea.centre_world_mm[0], fea.centre_world_mm[1]
    hx, hy = fea.half_size_mm[0], fea.half_size_mm[1]
    rect = mpatches.Rectangle(
        (cx - hx, cy - hy),
        2.0 * hx,
        2.0 * hy,
        fill=False,
        edgecolor="0.35",
        linewidth=0.8,
        linestyle="--",
    )
    hole = mpatches.Circle(
        (cx, cy),
        fea.hole_radius_mm,
        fill=False,
        edgecolor="0.35",
        linewidth=0.8,
        linestyle="--",
    )
    ax.add_patch(rect)
    ax.add_patch(hole)
    pad = 1.0
    ax.set_xlim(cx - hx - pad, cx + hx + pad)
    ax.set_ylim(cy - hy - pad, cy + hy + pad)
    ax.set_aspect("equal")


def plot_displacement_fields(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    out_dir: Path,
) -> None:
    """
    Plot a 4 x (1 + N_cases) comparison of ``u``, ``v``, ``w`` and ``|d|`` [mm]
    in physical specimen coordinates ``(x, y)`` with shared row colour limits.
    """
    case_list = list(dic_cases.values())
    n_cols = 1 + len(case_list)

    fig = plt.figure(figsize=(3.5 * n_cols + 1.0, 15.5))
    gs = fig.add_gridspec(
        4, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.055],
        wspace=0.22,
        hspace=0.28,
        top=0.92,
    )

    w_sign_label = (
        r"$w=-\Delta z$ (away from cam0)"
        if fea.w_convention == "cam0"
        else r"$w=+\Delta z$ (world)"
    )

    for row_idx, (comp_name, comp_idx, comp_tex) in enumerate(COMPONENTS):
        fea_vals = _extract_component(fea.tri_disp_mm, comp_idx)
        fea_nodal = _extract_component(fea.disp_all_mm[fea.front_node_mask], comp_idx)

        if comp_name == "magnitude":
            vmin = 0.0
            vmax = max(float(np.nanmax(fea_nodal)), 1e-6)
            cmap = CMAP_MAGNITUDE
            extend = "max"
        elif comp_name == "w":
            # w in FEA is <= 0.52 um while stereo-DIC w noise/bias reaches a few um;
            # use 99th percentile across datasets so all panels share an informative scale
            pooled = [np.abs(fea_nodal)]
            for cdata in case_list:
                pooled.append(np.abs(_extract_component(cdata.disp_mm, comp_idx)))
            all_abs = np.concatenate([arr[np.isfinite(arr)] for arr in pooled])
            lim = max(float(np.percentile(all_abs, 99.0)), 1e-6)
            vmin, vmax = -lim, lim
            cmap = CMAP_DIC_FEA
            extend = "both"
        elif comp_name == "v":
            lim = max(float(np.nanmax(np.abs(fea_nodal))), 1e-6)
            vmin, vmax = -lim, 0.0
            cmap = CMAP_DIC_FEA
            extend = "both"
        else:
            lim = max(float(np.nanmax(np.abs(fea_nodal))), 1e-6)
            vmin, vmax = -lim, lim
            cmap = CMAP_DIC_FEA
            extend = "both"

        # Column 0: FEA ground truth
        ax_fea = fig.add_subplot(gs[row_idx, 0])
        mappable = ax_fea.tripcolor(
            fea.triangulation,
            fea_vals,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        _draw_plate_outline(ax_fea, fea)
        ax_fea.set_ylabel(f"{comp_tex}\n" + r"$y$ [mm]")
        if row_idx == 3:
            ax_fea.set_xlabel(r"$x$ [mm]")
        if row_idx == 0:
            ax_fea.set_title(
                f"FEA (frame {fea.frame_idx})\n"
                f"[{float(np.nanmin(fea_nodal)):+.4f}, "
                f"{float(np.nanmax(fea_nodal)):+.4f}] mm"
            )
        else:
            ax_fea.set_title(
                f"[{float(np.nanmin(fea_nodal)):+.4f}, "
                f"{float(np.nanmax(fea_nodal)):+.4f}] mm"
            )

        # Remaining columns: stereo-DIC cases mapped onto physical plate (x, y)
        for col_offset, cdata in enumerate(case_list, start=1):
            ax = fig.add_subplot(gs[row_idx, col_offset])
            if comp_idx < 3:
                grid_val = cdata.grid_disp_mm[:, :, comp_idx]
            else:
                grid_val = np.linalg.norm(cdata.grid_disp_mm, axis=-1)

            masked_val = np.ma.masked_invalid(grid_val)
            mappable = ax.pcolormesh(
                cdata.grid_x_refracted_mm,
                cdata.grid_y_refracted_mm,
                masked_val,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                rasterized=True,
            )
            _draw_plate_outline(ax, fea)
            if row_idx == 3:
                ax.set_xlabel(r"$x$ [mm]")
            ax.set_yticklabels([])

            c_vals = _extract_component(cdata.disp_mm, comp_idx)
            range_txt = (
                f"[{float(np.nanmin(c_vals)):+.4f}, "
                f"{float(np.nanmax(c_vals)):+.4f}] mm"
            )
            if row_idx == 0:
                ax.set_title(f"{CASE_DISPLAY[cdata.case]}\n{range_txt}")
            else:
                ax.set_title(range_txt)

        cbar_ax = fig.add_subplot(gs[row_idx, -1])
        cbar = fig.colorbar(mappable, cax=cbar_ax, extend=extend)
        cbar.set_label(f"{comp_tex} [mm]")

    fig.suptitle(
        f"Displacement field comparison: FEA frame {fea.frame_idx} "
        f"(scaled $\\times${fea.frame_scale:.3f}) vs stereo DIC\n"
        r"Camera 0 convention: $u=+\Delta x$, $v=-\Delta y$ (down), "
        + w_sign_label,
        y=0.968,
    )
    _save_figure(fig, out_dir / "compare_displacement_fields.png")


def plot_dic_vs_fea_errors(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    out_dir: Path,
) -> None:
    """
    Plot spatial residual maps ``DIC - FEA`` [um] on the physical plate front
    face ``(x, y)`` [mm] for ``u``, ``v``, ``w`` and ``|d|``.
    """
    case_list = list(dic_cases.values())
    if not case_list:
        return
    n_cols = len(case_list)

    fig = plt.figure(figsize=(3.7 * n_cols + 1.0, 15.5))
    gs = fig.add_gridspec(
        4, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        wspace=0.20,
        hspace=0.30,
        top=0.925,
    )

    for row_idx, (comp_name, comp_idx, comp_tex) in enumerate(COMPONENTS):
        # Compute 2D error grids in micrometres [um]
        err_grids_um: list[np.ndarray] = []
        pooled_abs: list[np.ndarray] = []
        for cdata in case_list:
            if comp_idx < 3:
                err_um = 1e3 * (
                    cdata.grid_disp_mm[:, :, comp_idx]
                    - cdata.grid_fea_mm[:, :, comp_idx]
                )
            else:
                err_um = 1e3 * (
                    np.linalg.norm(cdata.grid_disp_mm, axis=-1)
                    - np.linalg.norm(cdata.grid_fea_mm, axis=-1)
                )
            err_grids_um.append(err_um)
            finite = np.abs(err_um[np.isfinite(err_um)])
            if finite.size:
                pooled_abs.append(finite)

        lim_um = (
            max(float(np.percentile(np.concatenate(pooled_abs), 99.0)), 0.05)
            if pooled_abs else 1.0
        )

        mappable = None
        for col_idx, (cdata, err_um) in enumerate(zip(case_list, err_grids_um)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            mappable = ax.pcolormesh(
                cdata.grid_x_refracted_mm,
                cdata.grid_y_refracted_mm,
                np.ma.masked_invalid(err_um),
                cmap=CMAP_DIC_FEA,
                vmin=-lim_um,
                vmax=lim_um,
                shading="nearest",
                rasterized=True,
            )
            _draw_plate_outline(ax, fea)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{comp_tex}" + r"$_{\mathrm{DIC}} - $"
                    + f"{comp_tex}" + r"$_{\mathrm{FEA}}$" + "\n" + r"$y$ [mm]"
                )
            else:
                ax.set_yticklabels([])
            if row_idx == 3:
                ax.set_xlabel(r"$x$ [mm]")

            finite_err = err_um[np.isfinite(err_um)]
            rmse_um = float(np.sqrt(np.mean(finite_err ** 2))) if finite_err.size else np.nan
            bias_um = float(np.mean(finite_err)) if finite_err.size else np.nan
            stats_str = f"RMSE={rmse_um:.2f}, bias={bias_um:+.2f} $\\mu$m"
            if row_idx == 0:
                ax.set_title(f"{CASE_DISPLAY[cdata.case]} $-$ FEA\n{stats_str}")
            else:
                ax.set_title(stats_str)

        if mappable is not None:
            cbar_ax = fig.add_subplot(gs[row_idx, -1])
            cbar = fig.colorbar(mappable, cax=cbar_ax, extend="both")
            cbar.set_label(r"Residual [$\mu$m]")

    fig.suptitle(
        "Pointwise stereo-DIC residuals relative to FEA ground truth "
        r"(refracted ray registration, $1\,\mathrm{px} = 39.06\,\mu\mathrm{m}$)",
        y=0.962,
    )
    _save_figure(fig, out_dir / "compare_dic_vs_fea_errors.png")


def plot_intercase_differences(
    fea: FEAGroundTruth,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    dic_grids: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    Plot spatial pairwise DIC differences (``pipe - air_diffuse``,
    ``water - air_diffuse``, ``water - pipe``) [um] on the shared plate grid.
    """
    pairs = [
        (a, b) for a, b in (
            ("pipe", "air_diffuse"),
            ("water", "air_diffuse"),
            ("water", "pipe"),
        )
        if a in dic_grids and b in dic_grids
    ]
    if not pairs:
        return

    n_cols = len(pairs)
    fig = plt.figure(figsize=(3.7 * n_cols + 1.0, 15.5))
    gs = fig.add_gridspec(
        4, n_cols + 1,
        width_ratios=[1.0] * n_cols + [0.06],
        wspace=0.20,
        hspace=0.30,
        top=0.925,
    )

    for row_idx, (comp_name, comp_idx, comp_tex) in enumerate(COMPONENTS):
        diff_grids_um: list[np.ndarray] = []
        pooled_abs: list[np.ndarray] = []
        for case_a, case_b in pairs:
            ga = dic_grids[case_a]
            gb = dic_grids[case_b]
            if comp_idx < 3:
                diff_um = 1e3 * (ga[:, :, comp_idx] - gb[:, :, comp_idx])
            else:
                diff_um = 1e3 * (
                    np.linalg.norm(ga, axis=-1) - np.linalg.norm(gb, axis=-1)
                )
            diff_grids_um.append(diff_um)
            finite = np.abs(diff_um[np.isfinite(diff_um)])
            if finite.size:
                pooled_abs.append(finite)

        lim_um = (
            max(float(np.percentile(np.concatenate(pooled_abs), 99.0)), 0.05)
            if pooled_abs else 1.0
        )

        mappable = None
        for col_idx, ((case_a, case_b), diff_um) in enumerate(zip(pairs, diff_grids_um)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            mappable = ax.pcolormesh(
                x_grid,
                y_grid,
                np.ma.masked_invalid(diff_um),
                cmap=CMAP_DIC_FEA,
                vmin=-lim_um,
                vmax=lim_um,
                shading="nearest",
                rasterized=True,
            )
            _draw_plate_outline(ax, fea)
            if col_idx == 0:
                ax.set_ylabel(f"$\\Delta${comp_tex}\n" + r"$y$ [mm]")
            else:
                ax.set_yticklabels([])
            if row_idx == 3:
                ax.set_xlabel(r"$x$ [mm]")

            finite = diff_um[np.isfinite(diff_um)]
            rmse_um = float(np.sqrt(np.mean(finite ** 2))) if finite.size else np.nan
            bias_um = float(np.mean(finite)) if finite.size else np.nan
            stats_str = f"RMSE={rmse_um:.2f}, bias={bias_um:+.2f} $\\mu$m"
            if row_idx == 0:
                ax.set_title(
                    f"{CASE_DISPLAY[case_a]} $-$ {CASE_DISPLAY[case_b]}\n{stats_str}"
                )
            else:
                ax.set_title(stats_str)

        if mappable is not None:
            cbar_ax = fig.add_subplot(gs[row_idx, -1])
            cbar = fig.colorbar(mappable, cax=cbar_ax, extend="both")
            cbar.set_label(r"Difference [$\mu$m]")

    fig.suptitle(
        "Pairwise stereo-DIC differences isolating cylindrical pipe and water refraction",
        y=0.962,
    )
    _save_figure(fig, out_dir / "compare_intercase_differences.png")


def plot_line_profiles(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    out_dir: Path,
) -> None:
    r"""
    Compare horizontal ($y = y_{\mathrm{hole}} = -1.5$ mm) and vertical
    ($x = -6.5$ mm) displacement profiles [um] across FEA and all DIC cases.
    """
    cx, cy = fea.centre_world_mm[0], fea.centre_world_mm[1]
    hx, hy = fea.half_size_mm[0], fea.half_size_mm[1]
    x_vert_cut = -6.5  # halfway between hole edge (-3.125 mm) and left edge (-12.5 mm)

    xs_line = np.linspace(cx - hx, cx + hx, 400)
    xy_horiz = np.column_stack([xs_line, np.full_like(xs_line, cy)])
    fea_horiz_um = 1e3 * evaluate_fea_front_face(fea, xy_horiz)

    ys_line = np.linspace(cy - hy, cy + hy, 400)
    xy_vert = np.column_stack([np.full_like(ys_line, x_vert_cut), ys_line])
    fea_vert_um = 1e3 * evaluate_fea_front_face(fea, xy_vert)

    dic_horiz_um = {
        name: 1e3 * interpolate_dic_to_points(cdata, xy_horiz)
        for name, cdata in dic_cases.items()
    }
    dic_vert_um = {
        name: 1e3 * interpolate_dic_to_points(cdata, xy_vert)
        for name, cdata in dic_cases.items()
    }

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 13.0), sharex="col")

    for row_idx, (comp_name, comp_idx, comp_tex) in enumerate(COMPONENTS):
        ax_h = axes[row_idx, 0]
        ax_v = axes[row_idx, 1]

        # Shade hole cutout on horizontal midline
        ax_h.axvspan(
            cx - fea.hole_radius_mm,
            cx + fea.hole_radius_mm,
            color="0.90",
            zorder=0,
            label="Hole" if row_idx == 0 else None,
        )

        ax_h.plot(
            xs_line,
            _extract_component(fea_horiz_um, comp_idx),
            color="black",
            linewidth=1.8,
            label=f"FEA (frame {fea.frame_idx})",
            zorder=4,
        )
        ax_v.plot(
            ys_line,
            _extract_component(fea_vert_um, comp_idx),
            color="black",
            linewidth=1.8,
            label=f"FEA (frame {fea.frame_idx})",
            zorder=4,
        )

        for name, cdata in dic_cases.items():
            color = CASE_COLORS.get(name, None)
            label = CASE_DISPLAY.get(name, name)
            ax_h.plot(
                xs_line,
                _extract_component(dic_horiz_um[name], comp_idx),
                color=color,
                linewidth=1.2,
                alpha=0.9,
                label=label,
                zorder=3,
            )
            ax_v.plot(
                ys_line,
                _extract_component(dic_vert_um[name], comp_idx),
                color=color,
                linewidth=1.2,
                alpha=0.9,
                label=label,
                zorder=3,
            )

        ax_h.set_ylabel(f"{comp_tex} [$\\mu$m]")
        ax_h.grid(True, alpha=0.3)
        ax_v.grid(True, alpha=0.3)

        if row_idx == 0:
            ax_h.set_title(f"Horizontal profile through hole centre ($y = {cy:.1f}$ mm)")
            ax_v.set_title(f"Vertical profile alongside hole ($x = {x_vert_cut:.1f}$ mm)")
            ax_h.legend(loc="best", framealpha=0.9)

    axes[3, 0].set_xlabel(r"$x$ [mm]")
    axes[3, 1].set_xlabel(r"$y$ [mm]")
    fig.suptitle(
        "Cross-sectional displacement profiles: FEA ground truth vs stereo DIC",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    _save_figure(fig, out_dir / "compare_line_profiles.png")


def plot_error_summary_and_shape(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    dic_vs_fea_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    """
    2 x 2 diagnostic summary figure:
      (a) Pointwise RMSE(DIC - FEA) [um / mpx] comparing refracted vs nominal mapping
      (b) Boxplot distributions of |DIC - FEA| [um] (refracted mapping)
      (c) Reconstructed 3D surface depth profile z_world(x) [mm] vs z = +0.5 mm
      (d) Lateral cylindrical refraction shift x_nominal - x_refracted [mm]
    """
    case_names = list(dic_cases.keys())
    if not case_names:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.2))
    ax_bar, ax_box, ax_z, ax_lat = axes.ravel()

    # Lookup RMSE [um] from table
    rmse_lookup: dict[tuple[str, str, str], float] = {}
    for row in dic_vs_fea_rows:
        rmse_lookup[(str(row["case"]), str(row["mapping"]), str(row["component"]))] = (
            float(row["rmse_um"])
        )

    comp_keys = [c[0] for c in COMPONENTS]
    comp_labels = [c[2] for c in COMPONENTS]
    x_pos = np.arange(len(comp_keys), dtype=float)
    n_cases = len(case_names)
    group_width = 0.78
    bar_w = group_width / (2 * n_cases)

    for idx, cname in enumerate(case_names):
        color = CASE_COLORS.get(cname, f"C{idx}")
        disp_label = CASE_DISPLAY.get(cname, cname)
        offset_ref = (2 * idx - n_cases + 0.5) * bar_w
        offset_nom = (2 * idx + 1 - n_cases + 0.5) * bar_w

        vals_ref = [rmse_lookup.get((cname, "refracted", ck), np.nan) for ck in comp_keys]
        vals_nom = [rmse_lookup.get((cname, "nominal", ck), np.nan) for ck in comp_keys]

        ax_bar.bar(
            x_pos + offset_ref, vals_ref, width=bar_w * 0.92,
            color=color, edgecolor="black", linewidth=0.5,
            label=f"{disp_label} (refracted)",
        )
        ax_bar.bar(
            x_pos + offset_nom, vals_nom, width=bar_w * 0.92,
            color=color, alpha=0.42, edgecolor="black", linewidth=0.5, hatch="//",
            label=f"{disp_label} (nominal)",
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(comp_labels)
    ax_bar.set_ylabel(r"RMSE($\mathrm{DIC} - \mathrm{FEA}$) [$\mu$m]")
    ax_bar.set_title("(a) Pointwise displacement RMSE: refracted vs nominal mapping")
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.set_ylim(0.0, ax_bar.get_ylim()[1] * 1.22)
    ax_bar.legend(ncol=2, loc="upper left", fontsize=7.5, framealpha=0.9)

    ax_bar_mpx = ax_bar.twinx()
    y0, y1 = ax_bar.get_ylim()
    # 1 um = 1e-3 mm = (1e-3 / MM_PER_PX) * 1e3 mpx = 1.0 / MM_PER_PX mpx
    um_to_mpx = 1.0 / MM_PER_PX
    ax_bar_mpx.set_ylim(y0 * um_to_mpx, y1 * um_to_mpx)
    ax_bar_mpx.set_ylabel("RMSE [milli-px]")

    # (b) Boxplots of |DIC - FEA| [um] on log scale for refracted mapping
    box_positions: list[float] = []
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    sep = 0.24
    for c_i, (_, comp_idx, _) in enumerate(COMPONENTS):
        for k_i, cname in enumerate(case_names):
            cdata = dic_cases[cname]
            valid_f = np.all(np.isfinite(cdata.fea_refracted_mm), axis=1)
            err_um = 1e3 * np.abs(
                _extract_component(cdata.disp_mm[valid_f], comp_idx)
                - _extract_component(cdata.fea_refracted_mm[valid_f], comp_idx)
            )
            err_um = err_um[np.isfinite(err_um) & (err_um > 0.0)]
            if err_um.size == 0:
                err_um = np.array([1e-6])
            pos = c_i + (k_i - 0.5 * (n_cases - 1)) * sep
            box_positions.append(pos)
            box_data.append(err_um)
            box_colors.append(CASE_COLORS.get(cname, f"C{k_i}"))

    bp = ax_box.boxplot(
        box_data,
        positions=box_positions,
        widths=sep * 0.78,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.2)

    ax_box.set_yscale("log")
    ax_box.set_xticks(np.arange(len(comp_labels)))
    ax_box.set_xticklabels(comp_labels)
    ax_box.set_ylabel(r"$|\mathrm{DIC} - \mathrm{FEA}|$ [$\mu$m] (log scale)")
    ax_box.set_title("(b) Absolute error distribution (median, IQR, 5--95\\% whiskers)")
    ax_box.grid(True, which="both", axis="y", alpha=0.3)
    legend_handles = [
        mpatches.Patch(
            facecolor=CASE_COLORS.get(cn, f"C{i}"),
            edgecolor="black",
            alpha=0.75,
            label=CASE_DISPLAY.get(cn, cn),
        )
        for i, cn in enumerate(case_names)
    ]
    ax_box.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    # (c) Reconstructed 3D surface depth z_world(x) near y = y_hole
    cy = fea.centre_world_mm[1]
    hx = fea.half_size_mm[0]
    ax_z.axhline(
        fea.z_front_mm, color="black", linewidth=1.8, linestyle="--",
        label=f"True plate front face ($z = {fea.z_front_mm:+.2f}$ mm)",
    )
    for cname, cdata in dic_cases.items():
        # Take subsets within +/- 2 mm of y_hole (or nearest band)
        band = np.abs(cdata.xy_refracted_mm[:, 1] - cy) <= 2.5
        if not np.any(band):
            band = np.ones(len(cdata.xy_refracted_mm), dtype=bool)
        order = np.argsort(cdata.xy_refracted_mm[band, 0])
        ax_z.plot(
            cdata.xy_refracted_mm[band, 0][order],
            cdata.stereo_world_mm[band, 2][order],
            ".",
            markersize=3.0,
            color=CASE_COLORS.get(cname, None),
            label=(
                f"{CASE_DISPLAY.get(cname, cname)} "
                f"(median $z = {np.nanmedian(cdata.stereo_world_mm[:, 2]):+.2f}$ mm)"
            ),
        )
    ax_z.set_xlim(-hx - 1.0, hx + 1.0)
    ax_z.set_xlabel(r"True plate $x$ [mm]")
    ax_z.set_ylabel(r"Reconstructed $z_{\mathrm{world}}$ [mm]")
    ax_z.set_title("(c) Stereo-triangulated surface depth vs true specimen plane")
    ax_z.grid(True, alpha=0.3)
    ax_z.legend(loc="best", framealpha=0.9)

    # (d) Apparent lateral shift x_nominal - x_refracted across the plate width
    xs_eval = np.linspace(-hx, hx, 250)
    for cname, cdata in dic_cases.items():
        # Forward-trace pixel row at y = y_hole
        px_scan = np.linspace(0.0, 1023.0, 600)
        cy_opt = NOMINAL_CY - NOMINAL_PIXEL_CENTER_OFFSET
        depth = cdata.cam0_mm[2] - fea.z_front_mm
        py_mid = np.full_like(px_scan, cy_opt - NOMINAL_FX * (cy - cdata.cam0_mm[1]) / depth)
        xy_ref_scan = project_cam0_pixels_to_plate(
            px_scan, py_mid, cdata.cam0_mm, cname,
            z_plate_mm=fea.z_front_mm, through_media=True,
        )
        xy_nom_scan = project_cam0_pixels_to_plate(
            px_scan, py_mid, cdata.cam0_mm, cname,
            z_plate_mm=fea.z_front_mm, through_media=False,
        )
        ok = (
            np.isfinite(xy_ref_scan[:, 0])
            & (np.abs(xy_ref_scan[:, 0]) <= hx)
        )
        ax_lat.plot(
            xy_ref_scan[ok, 0],
            xy_nom_scan[ok, 0] - xy_ref_scan[ok, 0],
            linewidth=1.6,
            color=CASE_COLORS.get(cname, None),
            label=(
                f"{CASE_DISPLAY.get(cname, cname)} "
                f"(median epi dist = {np.nanmedian(cdata.epi_dist_px):.3f} px)"
            ),
        )
    ax_lat.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    ax_lat.set_xlim(-hx - 1.0, hx + 1.0)
    ax_lat.set_xlabel(r"True plate $x_{\mathrm{refracted}}$ [mm]")
    ax_lat.set_ylabel(r"$x_{\mathrm{nominal}} - x_{\mathrm{refracted}}$ [mm]")
    ax_lat.set_title("(d) Apparent horizontal refraction shift in Camera 0")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend(loc="best", framealpha=0.9)

    fig.suptitle(
        "Summary of stereo-DIC displacement accuracy, 3D shape distortion and optical refraction",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, out_dir / "compare_error_summary_and_shape.png")


# ================================================================================
# Console summary report and CLI
# ================================================================================

def print_comparison_report(
    fea: FEAGroundTruth,
    dic_cases: dict[str, DICCaseData],
    dic_vs_fea_rows: list[dict[str, object]],
    intercase_rows: list[dict[str, object]],
    calib_shape_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    print("=" * 86)
    print("Application 4: Stereo-DIC vs FEA Ground-Truth Comparison")
    print("=" * 86)
    print(
        f"FEA ground truth: {fea.simdata_dir.name}, frame {fea.frame_idx}, "
        f"frame_scale={fea.frame_scale:.6f}, w_convention={fea.w_convention}"
    )
    front_uvw = fea.disp_all_mm[fea.front_node_mask]
    for cname, cidx, _ in COMPONENTS:
        vals = _extract_component(front_uvw, cidx)
        print(
            f"  FEA front-face {cname:9s}: "
            f"median={np.median(vals):+.6f} mm, "
            f"range=[{np.min(vals):+.6f}, {np.max(vals):+.6f}] mm"
        )

    print("\nCalibration & 3D Surface Reconstruction (world true z = +0.500 mm):")
    print(
        f"  {'Case':13s} {'Pass':>4s} {'fx0 [px]':>9s} {'Base [mm]':>10s} "
        f"{'Reproj0':>8s} {'EpiMed':>8s} {'z_med [mm]':>11s} {'z_ptp [mm]':>11s}"
    )
    for row in calib_shape_rows:
        if row["case"] == "nominal_rig":
            continue
        print(
            f"  {str(row['case']):13s} "
            f"{int(row['quality_passed']):4d} "
            f"{float(row['fx0_px']):9.2f} "
            f"{float(row['baseline_mm']):10.3f} "
            f"{float(row['reproj_mean0_px']):8.3f} "
            f"{float(row['epi_dist_median_px']):8.3f} "
            f"{float(row['z_world_median_mm']):+11.3f} "
            f"{float(row['z_world_ptp_mm']):11.3f}"
        )

    print("\nPointwise Displacement Errors (DIC - FEA, refracted true-point mapping):")
    print(
        f"  {'Case':13s} {'Comp':9s} {'N':>6s} {'Bias [um]':>10s} "
        f"{'MAE [um]':>10s} {'RMSE [um]':>10s} {'RMSE [mpx]':>11s} {'P95 [um]':>10s}"
    )
    for row in dic_vs_fea_rows:
        if row["mapping"] != "refracted":
            continue
        print(
            f"  {str(row['case']):13s} "
            f"{str(row['component']):9s} "
            f"{int(row['n_matched']):6d} "
            f"{float(row['bias_um']):+10.4f} "
            f"{float(row['mae_um']):10.4f} "
            f"{float(row['rmse_um']):10.4f} "
            f"{float(row['rmse_px']) * 1e3:11.3f} "
            f"{float(row['p95_abs_um']):10.4f}"
        )

    if intercase_rows:
        print("\nPairwise Inter-Case DIC Differences (shared physical plate grid):")
        print(
            f"  {'Comparison':22s} {'Comp':9s} {'N':>6s} {'Bias [um]':>10s} "
            f"{'MAE [um]':>10s} {'RMSE [um]':>10s} {'RMSE [mpx]':>11s}"
        )
        for row in intercase_rows:
            print(
                f"  {str(row['comparison']):22s} "
                f"{str(row['component']):9s} "
                f"{int(row['n_matched']):6d} "
                f"{float(row['bias_um']):+10.4f} "
                f"{float(row['mae_um']):10.4f} "
                f"{float(row['rmse_um']):10.4f} "
                f"{float(row['rmse_px']) * 1e3:11.3f}"
            )

    print(f"\nComparison tables and figures written to: {out_dir}")


def _resolve_case_base_dir(case_base_arg: str) -> Path:
    candidate = Path(case_base_arg)
    if candidate.is_dir():
        return candidate.resolve()
    under_base = BASE_TEST_DIR / case_base_arg
    return under_base.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 3D stereo-DIC outputs (air_diffuse, pipe, water) "
            "against the original FEA frame data from application_4_fea.py."
        )
    )
    parser.add_argument(
        "--case-base-dir",
        type=str,
        default=DEFAULT_CASE_BASE_DIR,
        help=(
            "Case directory under thesis-output/ containing air_diffuse, pipe "
            f"and water subdirectories (default: {DEFAULT_CASE_BASE_DIR})"
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASE_NAMES),
        choices=list(CASE_NAMES),
        help=f"Optical cases to include (default: {' '.join(CASE_NAMES)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison CSVs and plots (default: <case-base-dir>/compare)",
    )
    parser.add_argument(
        "--fea-dir",
        type=Path,
        default=PLATE_SIMDATA_DIR,
        help="Path to the plate-with-hole SimData CSV directory",
    )
    parser.add_argument(
        "--fea-frame",
        type=int,
        default=None,
        help="Override FEA deformed frame index (default: auto-select <= 1 px as in application_4_fea.py)",
    )
    parser.add_argument(
        "--w-convention",
        choices=("cam0", "world"),
        default="cam0",
        help=(
            "Sign convention for out-of-plane w: 'cam0' uses right-handed "
            "Camera 0 depth (+w = -dz_world, matching Pyvale stereo_disp_w_mm); "
            "'world' uses +w = +dz_world as drawn in application_4_fea.py (default: cam0)"
        ),
    )
    parser.add_argument(
        "--grid-step-mm",
        type=float,
        default=0.20,
        help="Grid spacing [mm] for shared inter-case DIC interpolation (default: 0.20)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write comparison CSVs only and skip matplotlib figures",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_dir = _resolve_case_base_dir(args.case_base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Case base directory not found: {base_dir}. "
            "Pass --case-base-dir pointing to rendered/DIC outputs."
        )

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (base_dir / "compare")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fea = load_fea_ground_truth(
        simdata_dir=args.fea_dir,
        forced_frame_idx=args.fea_frame,
        w_convention=args.w_convention,
    )

    dic_cases: dict[str, DICCaseData] = {}
    for case_name in args.cases:
        cdir = base_dir / case_name
        deformed_csv = cdir / DIC3D_DIRNAME / "dic3d_rtimage_1_cam0.csv"
        if not deformed_csv.is_file():
            print(f"Skipping '{case_name}': {deformed_csv} not found.")
            continue
        dic_cases[case_name] = load_dic_case(case_name, cdir, fea)

    if not dic_cases:
        raise RuntimeError(
            f"No stereo-DIC results found in {base_dir} for cases {args.cases}. "
            "Run application_4_fea_dic.py first."
        )

    x_grid, y_grid, fea_grid, dic_grids = build_shared_plate_grid(
        fea, dic_cases, step_mm=args.grid_step_mm
    )
    _, dic_vs_fea_rows, intercase_rows, calib_shape_rows = compute_and_export_tables(
        fea, dic_cases, fea_grid, dic_grids, out_dir
    )

    if not args.no_plots:
        plot_displacement_fields(fea, dic_cases, out_dir)
        plot_dic_vs_fea_errors(fea, dic_cases, out_dir)
        plot_intercase_differences(fea, x_grid, y_grid, dic_grids, out_dir)
        plot_line_profiles(fea, dic_cases, out_dir)
        plot_error_summary_and_shape(fea, dic_cases, dic_vs_fea_rows, out_dir)

    print_comparison_report(
        fea, dic_cases, dic_vs_fea_rows, intercase_rows, calib_shape_rows, out_dir
    )


if __name__ == "__main__":
    main()
