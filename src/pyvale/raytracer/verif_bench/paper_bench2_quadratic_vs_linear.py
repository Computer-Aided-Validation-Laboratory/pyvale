#!/usr/bin/env python3
"""Paper figure/table generator for benchmark case 2.

Consumes the outputs written by ``bench2_quadratic_vs_linear.cpp``.  It reads the per-arm
statistics, builds the cost-versus-accuracy curves, and writes the LaTeX
figure and table that are \\input by the manuscript.

"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path as _Path

from paper_const import repo_root

PAPER_DIR = _Path(
    "/home/wiera/Documents/paper-feebee-raytracer/figures/bench2",
).expanduser()

BENCH_DIR = pathlib.Path("verif_bench/bench/bench2")
STATS_PATH = BENCH_DIR / "bench2_stats.csv"
ANALYTIC_PATH = BENCH_DIR / "bench2_vs_analytic.csv"
CAMERA_PATH = BENCH_DIR / "bench2_camera.csv"
VIEW_PPM = BENCH_DIR / "bench2_camera_view_{geometry}.ppm"
DATA_DIR = pathlib.Path("verif_bench/data/curved")
OUT_FIGS_TEX_PATH = BENCH_DIR / "figs_bench2.tex"
OUT_TABS_TEX_PATH = BENCH_DIR / "tabs_bench2.tex"

# Camera-scene wireframe: the coarse quadratic family, drawn for legibility. The
# rendered camera views use the fine mesh of the same family.
VIEW_WIREFRAME = {
    "sphere": DATA_DIR / "curved_sph_tri_coarse.vtk",
    "cylinder": DATA_DIR / "curved_cyl_quad_coarse.vtk",
}

PLOT_DPI = 300.0

# Arms and how they are drawn.  The quadratic arms are single points (a mesh is
# either quadratic or it is not); the linear arms form a ladder in k.
QUADRATIC_ARMS = ("quadratic_tri", "quadratic_quad")
LINEAR_LADDER_ARM = "linear_subdiv"
LINEAR_NATIVE_ARM = "linear_native"

ARM_LABEL = {
    "quadratic_tri": "TRI6 (Newton)",
    "quadratic_quad": "QUAD9 (Newton)",
    "linear_subdiv": "linear sub-tessellation",
    "linear_native": "linear mesh (Gmsh)",
}

ARM_STYLE = {
    "quadratic_tri": {"color": "#3b6ea5", "marker": "o"},
    "quadratic_quad": {"color": "#3b6ea5", "marker": "s"},
    "linear_subdiv": {"color": "#b0553b", "marker": "^"},
    "linear_native": {"color": "#000000", "marker": "x"},
}

FIG_CAPTION = (
    "Radial RMS error for benchmark case 2. (a,b) Radial RMS error at the "
    "hit against single-threaded ray per second throughput, one curve per arm, for the sphere "
    "and the cylinder. The linear curve is labelled by the sub-tessellation factor $k$. "
    "(c,d) The same data against primitive count. Ray bundles are frozen and "
    "shared across every arm."
)

NORMAL_FIG_CAPTION = (
    "Surface-normal error for benchmark case 2. The panels mirror "
    "Fig.~\\ref{fig:bench2_pareto} but the vertical axis is the RMS angle in "
    "degrees between the normal returned at the hit and the analytic surface "
    "normal."
)

CAMERA_FIG_CAPTION = (
    "The camera used in benchmark case 2 and the view it sees. Top row: the "
    "frozen pinhole camera at $(0,0,2.5)$ aimed at the geometry, with the four "
    "corner rays that bound the $96\\times96$ camera bundle (the grey rectangle is "
    "the screen plane). Bottom row: the "
    "same camera rendered by tracing its rays through the fine quadratic mesh "
    "with the Newton solver (\\texttt{tri6} for the sphere, \\texttt{quad9} for "
    "the cylinder). "
    "Position, orientation and field of view are exactly those of the timed "
    "camera bundle, only the sampling density is higher for the displayed image. "
    "The cylinder sits high in its frame because the "
    "camera is level with its lower cap and tilts upwards to aim at the centroid."
)

TABLE_CAPTION = (
    "Cost--accuracy comparison in benchmark case 2. "
    "For each geometry the quadratic arm is compared with the best linear "
    "sub-tessellation available on the ladder. "
    "``Quad \\gls{rms} err'' is the radial \\gls{rms} error at the hit for "
    "the quadratic Newton solve and ``Quad nrm err'' the \\gls{rms} "
    "surface-normal error in degrees. "
    "``Lin err / quad'' and ``Lin nrm / quad'' are the ratios of the "
    "linear model's \\gls{rms} errors to the quadratic arm's, so a value "
    "above one means the linear model did not reach the quadratic accuracy "
    "even at the finest sub-tessellation. "
    "``Prim ratio'' is linear-to-quadratic primitive count and "
    "``Speed ratio'' is quadratic-to-linear throughput where values "
    "above one favour the quadratic arm. "
    "Throughput is single threaded."
)



def run_binary_if_needed() -> None:
    if (repo_root() / STATS_PATH).exists():
        return
    binary = repo_root() / "verif_bench/bench2_quadratic_vs_linear"
    if not binary.exists():
        print(
            "Missing bench2 outputs. Build the case first (from verif_bench/):\n"
            "  g++ -O2 -std=c++20 -fopenmp -o bench2_quadratic_vs_linear \\\n"
            "      bench2_quadratic_vs_linear.cpp cpp/rtrayintersection_modes.cpp\n"
            "  ./bench2_quadratic_vs_linear",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("Missing bench2 outputs, running bench2_quadratic_vs_linear...")
    subprocess.run([str(binary)], cwd=repo_root() / "verif_bench", check=True)


def load_stats() -> list[dict]:
    rows = []
    with open(repo_root() / STATS_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "geometry": row["geometry"],
                    "arm": row["arm"],
                    "level": row["level"],
                    "subdiv_k": int(row["subdiv_k"]),
                    "elements": int(row["elements"]),
                    "primitives": int(row["primitives"]),
                    "bundle": row["bundle"],
                    "rays": int(row["rays"]),
                    "hits": int(row["hits"]),
                    "missed": int(row["missed"]),
                    "coverage": float(row["coverage"]),
                    "mean_err": float(row["mean_radial_err"]),
                    "rms_err": float(row["rms_radial_err"]),
                    "trimmed_max_err": float(row["trimmed_max_radial_err"]),
                    "max_err": float(row["max_radial_err"]),
                    "mean_normal_err": float(row["mean_normal_err_deg"]),
                    "rms_normal_err": float(row["rms_normal_err_deg"]),
                    "trimmed_max_normal_err": float(
                        row["trimmed_max_normal_err_deg"]
                    ),
                    "max_normal_err": float(row["max_normal_err_deg"]),
                    "mean_tests": float(row["mean_primitive_tests"]),
                    "solver_runs": int(row["solver_runs"]),
                    "mean_iters": float(row["mean_iters_per_run"]),
                    "wall_s": float(row["wall_time_s"]),
                    "rays_per_s": float(row["rays_per_s"]),
                }
            )
    return rows


def load_analytic() -> list[dict]:
    rows = []
    with open(repo_root() / ANALYTIC_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "geometry": row["geometry"],
                    "mesh": row["mesh"],
                    "cells": int(row["cells"]),
                    "mean_err": float(row["mean_radial_err"]),
                    "max_err": float(row["max_radial_err"]),
                }
            )
    return rows


def load_camera() -> list[dict]:
    """Camera basis exported by the C++ harness (the exact one used)."""
    rows = []
    with open(repo_root() / CAMERA_PATH, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "geometry": row["geometry"],
                    "origin": np.array(
                        [float(row["origin_x"]), float(row["origin_y"]),
                         float(row["origin_z"])]
                    ),
                    "centre": np.array(
                        [float(row["centre_x"]), float(row["centre_y"]),
                         float(row["centre_z"])]
                    ),
                    "view": np.array(
                        [float(row["view_x"]), float(row["view_y"]),
                         float(row["view_z"])]
                    ),
                    "right": np.array(
                        [float(row["right_x"]), float(row["right_y"]),
                         float(row["right_z"])]
                    ),
                    "up": np.array(
                        [float(row["up_x"]), float(row["up_y"]),
                         float(row["up_z"])]
                    ),
                    "extent": float(row["extent"]),
                    "grid": int(row["grid"]),
                }
            )
    return rows


def load_wireframe(path):
    """Element-boundary polylines of a legacy VTK mesh.

    Only the node order matters here: corners first, then edge midpoints, so a
    TRI6 boundary is traversed corner-midpoint-corner to draw the curved edge.
    """
    lines = path.read_text().splitlines()
    points = np.empty((0, 3))
    cells = []
    types = []
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if not parts:
            i += 1
            continue
        key = parts[0].upper()
        if key == "POINTS":
            count = int(parts[1])
            values = []
            i += 1
            while len(values) < count * 3:
                values.extend(float(v) for v in lines[i].split())
                i += 1
            points = np.array(values[: count * 3]).reshape(count, 3)
            continue
        if key == "CELLS":
            count = int(parts[1])
            i += 1
            for _ in range(count):
                ids = [int(v) for v in lines[i].split()]
                cells.append(ids[1:])
                i += 1
            continue
        if key == "CELL_TYPES":
            count = int(parts[1])
            i += 1
            while len(types) < count:
                types.extend(int(v) for v in lines[i].split())
                i += 1
            continue
        i += 1

    edge_order = {
        5: [(0, 1), (1, 2), (2, 0)],                    # TRI3
        22: [(0, 3, 1), (1, 4, 2), (2, 5, 0)],          # TRI6
        9: [(0, 1), (1, 2), (2, 3), (3, 0)],            # QUAD4
        24: [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],  # QUAD9
        28: [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)],  # QUAD9
    }

    polylines = []
    for ids, code in zip(cells, types):
        order = edge_order.get(code)
        if order is None:
            continue
        for edge in order:
            polylines.append(points[[ids[k] for k in edge]])
    return polylines


def pick(rows, geometry, arm, bundle="camera", level=None, subdiv_k=None):
    for row in rows:
        if row["geometry"] != geometry or row["arm"] != arm:
            continue
        if row["bundle"] != bundle:
            continue
        if level is not None and row["level"] != level:
            continue
        if subdiv_k is not None and row["subdiv_k"] != subdiv_k:
            continue
        return row
    return None


def ladder(rows, geometry, bundle="camera", prefix=""):
    """Return the linear_subdiv ladder (k ascending) for a geometry."""
    out = [
        row
        for row in rows
        if row["geometry"] == geometry
        and row["arm"] == LINEAR_LADDER_ARM
        and row["bundle"] == bundle
        and row["level"].startswith(prefix)
    ]
    out.sort(key=lambda r: r["subdiv_k"])
    return out


def _pareto_chart(stats, error_key, ylabel, out_stem, fig_caption,
                  fig_label) -> str:
    """Draw a 2x2 cost-accuracy chart for one error metric.

    The four panels are (sphere|cylinder) x (throughput|primitives).
    """
    geometries = ("sphere", "cylinder")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))

    for col, geometry in enumerate(geometries):
        ax_cost = axes[0][col]
        ax_prim = axes[1][col]

        # Quadratic arms: one marker each.
        for arm in QUADRATIC_ARMS:
            row = pick(stats, geometry, arm)
            if row is None:
                continue
            style = ARM_STYLE[arm]
            for ax, xval in (
                (ax_cost, row["rays_per_s"]),
                (ax_prim, row["primitives"]),
            ):
                ax.scatter(
                    [xval],
                    [row[error_key]],
                    marker=style["marker"],
                    color=style["color"],
                    zorder=5,
                    label=ARM_LABEL[arm],
                )

        # Linear-native cross-check.
        native = pick(stats, geometry, LINEAR_NATIVE_ARM)
        if native is not None:
            style = ARM_STYLE[LINEAR_NATIVE_ARM]
            for ax, xval in (
                (ax_cost, native["rays_per_s"]),
                (ax_prim, native["primitives"]),
            ):
                ax.scatter(
                    [xval],
                    [native[error_key]],
                    marker=style["marker"],
                    color=style["color"],
                    zorder=5,
                    label=ARM_LABEL[LINEAR_NATIVE_ARM],
                )

        # Linear sub-tessellation ladders.  The fine (TRI6) and quad (QUAD9)
        # families are drawn as one curve each, since they share the same surface.
        for prefix, fam_label in (("tri6_", "linear sub-tessellation (from TRI6)"),
                                  ("quad9_", "linear sub-tessellation (from QUAD9)")):
            rows = ladder(stats, geometry, prefix=prefix)
            if not rows:
                continue
            style = ARM_STYLE[LINEAR_LADDER_ARM]
            ks = [r["subdiv_k"] for r in rows]
            for ax, key in ((ax_cost, "rays_per_s"), (ax_prim, "primitives")):
                xs = [r[key] for r in rows]
                ys = [r[error_key] for r in rows]
                ax.plot(
                    xs,
                    ys,
                    marker=style["marker"],
                    color=style["color"],
                    linewidth=1.2,
                    label=fam_label,
                )
                for xv, yv, k in zip(xs, ys, ks):
                    ax.annotate(
                        f"k={k}",
                        (xv, yv),
                        textcoords="offset points",
                        xytext=(3, 4),
                        fontsize=6.0,
                    )

        for ax, xlabel, title in (
            (ax_cost, "throughput (rays/s, single thread)", "(a) cost vs accuracy"),
            (ax_prim, "primitive count", "(c) primitives vs accuracy"),
        ):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel, fontsize=8.0)
            ax.set_ylabel(ylabel, fontsize=8.0)
            ax.tick_params(axis="both", labelsize=7.0)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)

        ax_cost.set_title(f"{geometry}: throughput", fontsize=9.0)
        ax_prim.set_title(f"{geometry}: primitives", fontsize=9.0)

        handles, labels = ax_cost.get_legend_handles_labels()
        # De-duplicate by label, keeping first occurrence.
        seen = {}
        for handle, label in zip(handles, labels):
            seen.setdefault(label, handle)
        ax_cost.legend(
            seen.values(), seen.keys(), fontsize=6.5, loc="best"
        )

    fig.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PAPER_DIR / out_stem
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.96\\textwidth]{{{out_stem}}}\n"
        f"\\caption{{{fig_caption}}}\n"
        f"\\label{{{fig_label}}}\n"
        "\\end{figure}\n"
    )


def build_figure(stats) -> str:
    return _pareto_chart(
        stats,
        error_key="rms_err",
        ylabel="radial RMS error",
        out_stem="fig_bench2_pareto.png",
        fig_caption=FIG_CAPTION,
        fig_label="fig:bench2_pareto",
    )


def build_normal_figure(stats) -> str:
    return _pareto_chart(
        stats,
        error_key="rms_normal_err",
        ylabel="surface-normal RMS error (deg)",
        out_stem="fig_bench2_normal.png",
        fig_caption=NORMAL_FIG_CAPTION,
        fig_label="fig:bench2_normal",
    )


def _analytic_surface(geometry, radius, y_min, y_max):
    """A light surface mesh for the 3D scene panel."""
    if geometry == "sphere":
        theta = np.linspace(0.0, np.pi, 48)
        phi = np.linspace(0.0, 2.0 * np.pi, 72)
        theta, phi = np.meshgrid(theta, phi)
        return (
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        )
    theta = np.linspace(0.0, 2.0 * np.pi, 96)
    y = np.linspace(y_min, y_max, 12)
    theta, y = np.meshgrid(theta, y)
    return (
        radius * np.cos(theta),
        y,
        radius * np.sin(theta),
    )


def build_camera_figure(cameras) -> str:
    """Camera scene (top) and the view it sees (bottom), per geometry."""
    from PIL import Image

    geometries = ("sphere", "cylinder")
    by_geo = {row["geometry"]: row for row in cameras}
    if not by_geo:
        return ""

    fig = plt.figure(figsize=(7.4, 6.4))

    for col, geometry in enumerate(geometries):
        cam = by_geo.get(geometry)
        if cam is None:
            continue
        radius = 0.5
        y_min, y_max = (0.0, 1.0) if geometry == "cylinder" else (-radius, radius)
        origin, centre = cam["origin"], cam["centre"]
        right, up = cam["right"], cam["up"]
        extent = cam["extent"]

        # ---- Top: the scene, with the camera and its frustum.
        ax = fig.add_subplot(2, 2, 1 + col, projection="3d")
        sx, sy, sz = _analytic_surface(geometry, radius, y_min, y_max)
        ax.plot_surface(sx, sy, sz, color="#d8d8d8", alpha=0.55,
                        linewidth=0, shade=False, antialiased=True)
        if geometry == "cylinder":
            # Flat end caps, added as triangle fans so the solid reads as a
            # closed cylinder rather than an open tube.
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            theta = np.linspace(0.0, 2.0 * np.pi, 96)
            fan = []
            for cap_y in (y_min, y_max):
                for t0, t1 in zip(theta[:-1], theta[1:]):
                    fan.append([
                        (0.0, cap_y, 0.0),
                        (radius * np.cos(t0), cap_y, radius * np.sin(t0)),
                        (radius * np.cos(t1), cap_y, radius * np.sin(t1)),
                    ])
            ax.add_collection3d(
                Poly3DCollection(fan, facecolor="#d8d8d8", alpha=0.55,
                                 edgecolor="none")
            )

        wire_path = repo_root() / VIEW_WIREFRAME[geometry]
        wire = load_wireframe(wire_path) if wire_path.exists() else []
        path = repo_root() / str(VIEW_PPM).format(geometry=geometry)
        if wire:
            xs, ys, zs = [], [], []
            for poly in wire:
                xs.extend(list(poly[:, 0]) + [np.nan])
                ys.extend(list(poly[:, 1]) + [np.nan])
                zs.extend(list(poly[:, 2]) + [np.nan])
            ax.plot(xs, ys, zs, color="#3b6ea5", linewidth=0.4, alpha=0.85)

        # Frustum: corner rays plus the screen rectangle on the plane at centre.
        corners = [
            centre + extent * sr * right + extent * su * up
            for sr in (-1.0, 1.0)
            for su in (-1.0, 1.0)
        ]
        order = [corners[0], corners[1], corners[3], corners[2], corners[0]]
        rect = np.array(order)
        ax.plot(rect[:, 0], rect[:, 1], rect[:, 2], color="#555555",
                linewidth=0.7, alpha=0.8)
        for corner in corners:
            seg = np.vstack([origin, corner])
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="#888888",
                    linewidth=0.5, alpha=0.7)
        axis = np.vstack([origin, centre])
        ax.plot(axis[:, 0], axis[:, 1], axis[:, 2], color="#c0392b",
                linewidth=0.8, linestyle="--", alpha=0.9)

        ax.scatter([origin[0]], [origin[1]], [origin[2]], color="#111111",
                   s=26, depthshade=False, zorder=10)
        ax.text(origin[0], origin[1], origin[2] + 0.12, "camera",
                fontsize=7.0, ha="center")

        # Equal data scale on all three axes, spanning the object *and* the
        # camera so neither is clipped.
        scene = np.vstack([np.array([[origin[0], origin[1], origin[2]]]),
                           rect,
                           np.array([[sx.min(), sy.min(), sz.min()],
                                     [sx.max(), sy.max(), sz.max()]])])
        lo, hi = scene.min(axis=0), scene.max(axis=0)
        mid = 0.5 * (lo + hi)
        span = float((hi - lo).max())
        ax.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
        ax.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
        ax.set_zlim(mid[2] - span / 2, mid[2] + span / 2)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except AttributeError:
            pass
        ax.view_init(elev=16.0, azim=-58.0)
        ax.set_axis_off()
        ax.set_title(f"({'a' if col == 0 else 'b'}) {geometry}: camera",
                     fontsize=9.0, pad=-2)

        # ---- Bottom: what the camera sees, rendered by the tracer.
        ax_view = fig.add_subplot(2, 2, 3 + col)
        if path.exists():
            image = Image.open(path)
            rgb = np.asarray(image.convert("RGB")).reshape(-1, 3)
            colours, counts = np.unique(rgb, axis=0, return_counts=True)
            background = colours[int(np.argmax(counts))]
            hit = float(np.any(rgb != background, axis=1).mean())
            ax_view.imshow(image, interpolation="bilinear")
            ax_view.set_title(
                f"({'c' if col == 0 else 'd'}) {geometry}: camera view "
                f"({image.width}$\\times${image.height}, {hit:.0%} hit)",
                fontsize=9.0,
            )
        else:
            ax_view.text(0.5, 0.5, "camera view not rendered\n(run with "
                         "--view-grid > 0)", ha="center", va="center",
                         transform=ax_view.transAxes, fontsize=7.0)
        ax_view.set_xticks([])
        ax_view.set_yticks([])
        for spine in ax_view.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    fig.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    out_png = PAPER_DIR / "fig_bench2_camera.png"
    fig.savefig(out_png, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        "\\includegraphics[width=0.96\\textwidth]{fig_bench2_camera.png}\n"
        f"\\caption{{{CAMERA_FIG_CAPTION}}}\n"
        "\\label{fig:bench2_camera}\n"
        "\\end{figure}\n"
    )


def best_linear(stats, geometry):
    """The most accurate linear sub-tessellation available for a geometry."""
    candidates = [
        row
        for row in stats
        if row["geometry"] == geometry
        and row["arm"] == LINEAR_LADDER_ARM
        and row["bundle"] == "camera"
    ]
    if not candidates:
        return None
    # Lowest RMS error; ties broken by higher throughput.
    return min(candidates, key=lambda r: (r["rms_err"], -r["rays_per_s"]))


def cheapest_matching_linear(stats, geometry, target_err):
    """Lowest-throughput linear sub-tessellation whose error <= target_err."""
    candidates = [
        row
        for row in stats
        if row["geometry"] == geometry
        and row["arm"] == LINEAR_LADDER_ARM
        and row["bundle"] == "camera"
        and row["rms_err"] <= target_err
    ]
    if not candidates:
        return None
    # Highest throughput among those meeting the accuracy target.
    return max(candidates, key=lambda r: r["rays_per_s"])


def build_table(stats) -> str:
    lines = []
    geometries = ("sphere", "cylinder")
    quad_arms = ("quadratic_tri", "quadratic_quad")

    for geometry in geometries:
        for arm in quad_arms:
            quad = pick(stats, geometry, arm)
            if quad is None:
                continue

            # If a linear model can match the quadratic error, report the
            # cheapest one that does (the Pareto crossing). Otherwise report the
            # most accurate linear model available, which is the closest the
            # ladder ever gets, and state the residual error ratio.
            lin = cheapest_matching_linear(stats, geometry, quad["rms_err"])
            matched = lin is not None
            if lin is None:
                lin = best_linear(stats, geometry)
            if lin is None:
                continue

            err_ratio = lin["rms_err"] / quad["rms_err"] if quad["rms_err"] else 0.0
            prim_ratio = lin["primitives"] / quad["primitives"]
            speed_ratio = quad["rays_per_s"] / lin["rays_per_s"]
            normal_ratio = (
                lin["rms_normal_err"] / quad["rms_normal_err"]
                if quad["rms_normal_err"]
                else 0.0
            )

            if matched:
                note = "match"
            else:
                note = f"{err_ratio:.0f}$\\times$"

            lines.append(
                "\\texttt{{{geo}}} & \\texttt{{{arm_lab}}} & {qerr:.2e} & "
                "{qn:.2e} & {qprim} & {qkps:.0f} & {lk} & {lprim} & {lkps:.0f} & "
                "{pr:.0f}$\\times$ & {sr:.2f}$\\times$ & {note} & "
                "{nr:.0f}$\\times$ \\\\".format(
                    geo=geometry,
                    arm_lab=ARM_LABEL[arm].replace(" (Newton)", ""),
                    qerr=quad["rms_err"],
                    qn=quad["rms_normal_err"],
                    qprim=quad["primitives"],
                    qkps=quad["rays_per_s"],
                    lk=lin["subdiv_k"],
                    lprim=lin["primitives"],
                    lkps=lin["rays_per_s"],
                    pr=prim_ratio,
                    sr=speed_ratio,
                    note=note,
                    nr=normal_ratio,
                )
            )

    body = "\n".join(lines)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{TABLE_CAPTION}}}\n"
        "\\label{tab:bench2_pareto}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llrrrrrrrrrrr}\n"
        "\\hline\n"
        "Geometry & Arm & \\makecell{Quad\\\\RMS err} & "
        "\\makecell{Quad\\\\nrm\\\\err} & "
        "\\makecell{Quad\\\\prims} & \\makecell{Quad\\\\rays/s} & "
        "\\makecell{Lin\\\\$k$} & \\makecell{Linear\\\\prims} & "
        "\\makecell{Linear\\\\rays/s} & "
        "\\makecell{Prim\\\\ratio} & \\makecell{Speed\\\\ratio} & "
        "\\makecell{Lin err\\\\/ quad} & "
        "\\makecell{Lin nrm\\\\/ quad} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )


def write_tex_outputs(figs_tex: str, tabs_tex: str) -> None:
    out_figs_path = repo_root() / OUT_FIGS_TEX_PATH
    out_tabs_path = repo_root() / OUT_TABS_TEX_PATH
    out_figs_path.parent.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    out_figs_path.write_text(figs_tex)
    out_tabs_path.write_text(tabs_tex)
    (PAPER_DIR / OUT_FIGS_TEX_PATH.name).write_text(figs_tex)
    (PAPER_DIR / OUT_TABS_TEX_PATH.name).write_text(tabs_tex)

    print(f"Wrote {out_figs_path}")
    print(f"Wrote {out_tabs_path}")
    print(f"Wrote {PAPER_DIR / OUT_FIGS_TEX_PATH.name}")
    print(f"Wrote {PAPER_DIR / OUT_TABS_TEX_PATH.name}")


def main() -> int:
    run_binary_if_needed()

    stats = load_stats()
    analytic = load_analytic()
    cameras = (
        load_camera() if (repo_root() / CAMERA_PATH).exists() else []
    )
    if not cameras:
        print("No bench2_camera.csv; the camera figure will be skipped.")

    print("Arms present (camera bundle):")
    for row in stats:
        if row["bundle"] != "camera":
            continue
        print(
            f"  {row['geometry']:8s} {row['arm']:14s} {row['level']:16s} "
            f"prims {row['primitives']:>7d}  "
            f"rms {row['rms_err']:.3e}  "
            f"nrm {row['rms_normal_err']:.4f}deg  "
            f"rays/s {row['rays_per_s']:>10.0f}  "
            f"hits {row['hits']}/{row['rays']}"
        )

    print("\nPer-element geometric error vs analytic surface:")
    for row in analytic:
        print(
            f"  {row['geometry']:8s} {row['mesh']:28s} "
            f"cells {row['cells']:>6d}  "
            f"mean {row['mean_err']:.3e}  max {row['max_err']:.3e}"
        )

    print("\nCamera (exported by the harness):")
    for row in cameras:
        origin = ", ".join(f"{v:g}" for v in row["origin"])
        centre = ", ".join(f"{v:g}" for v in row["centre"])
        view = ", ".join(f"{v:g}" for v in row["view"])
        print(
            f"  {row['geometry']:8s} origin ({origin}) -> centre ({centre})  "
            f"view ({view})  extent {row['extent']:.3f}  grid {row['grid']}"
        )

    print("\nHeadline: quadratic error vs cheapest matching linear model:")
    for geometry in ("sphere", "cylinder"):
        for arm in QUADRATIC_ARMS:
            quad = pick(stats, geometry, arm)
            if quad is None:
                continue
            lin = cheapest_matching_linear(stats, geometry, quad["rms_err"])
            if lin is None:
                lin = best_linear(stats, geometry)
                print(
                    f"  {geometry:8s} {arm:14s}: quad {quad['primitives']:>6d} prims "
                    f"@ {quad['rays_per_s']:>9.0f} rays/s rms {quad['rms_err']:.2e} "
                    f"nrm {quad['rms_normal_err']:.4f}deg  "
                    f"| NO linear model matched; best is k={lin['subdiv_k']} with "
                    f"{lin['primitives']:>6d} prims @ {lin['rays_per_s']:>9.0f} rays/s "
                    f"rms {lin['rms_err']:.2e} nrm {lin['rms_normal_err']:.4f}deg "
                    f"({lin['rms_err'] / quad['rms_err']:.0f}x more radial error, "
                    f"{lin['rms_normal_err'] / quad['rms_normal_err']:.0f}x more "
                    f"normal error at "
                    f"{lin['primitives'] / quad['primitives']:.0f}x prims, "
                    f"{quad['rays_per_s'] / lin['rays_per_s']:.2f}x speed)"
                )
                continue
            print(
                f"  {geometry:8s} {arm:14s}: quad {quad['primitives']:>6d} prims "
                f"@ {quad['rays_per_s']:>9.0f} rays/s  vs  "
                f"linear k={lin['subdiv_k']} {lin['primitives']:>6d} prims "
                f"@ {lin['rays_per_s']:>9.0f} rays/s  "
                f"({lin['primitives'] / quad['primitives']:.1f}x prims, "
                f"{quad['rays_per_s'] / lin['rays_per_s']:.1f}x speed)"
            )

    write_tex_outputs(
        build_figure(stats)
        + build_normal_figure(stats)
        + build_camera_figure(cameras),
        build_table(stats),
    )
    print(f"\nSaved figure assets to {PAPER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
