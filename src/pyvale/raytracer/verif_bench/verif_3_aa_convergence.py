#!/usr/bin/env python3
"""Verification case 3: anti-aliasing convergence.

Convergence study using ``antialiasing_samples`` parameter (the number of Sobol
samples cast per pixel). For every bunny mesh, surface shader, and
image resolution the scene is rendered at an ascending sequence of
anti-aliasing sample counts (doubling each iteration starting from one sample). 
Each render is differenced
against the previous (half as many samples) render and the resulting
per-pixel errors are logged. Rendering continues until the maximum absolute
per-pixel error drops to or below a configurable threshold on a number of
iterations (runs one extra step "for sureness" after the
first criterion hit and stops on the next hit) or until a configurable maximum
sample count is reached.

The maximum-error criterion is configurable in 8-bit grey levels (GL) via
``--max-abs-err`` or, by default, as a fraction of the full scale (255 GL) via
``--max-abs-err-frac``.

Output layout (self-contained under ``verif_bench/verif/verif_3``)::

    c_<mesh>_<shader>_<resolution>/
        rtimage_subsamples_<aa>.bmp   render at ``aa`` anti-aliasing samples
        convergence_log.csv           per-refinement RMSE/error metrics
        render_time_log.csv           render wall-clock time per step

The corresponding paper figures are produced by ``paper_verif_3.py``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtmain import (
    RenderType,
    ShadingType,
    TextureSampler,
    render_scene,
)
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtscene import Scene
from pyvale.sensorsim.imagetools import ImageTools

import pyvale.sensorsim as sens


VERIF_ROOT = Path(__file__).resolve().parent / "verif" / "verif_3"
DATA_ROOT = Path(__file__).resolve().parent / "data"
TEXTURE_PATH = DATA_ROOT / "texture" / "speckle-simple.tiff"

BUNNY_NAMES = ["tri3", "tri6", "quad4", "quad8", "quad9"]
BUNNY_DIRS = [f"riley_{name}" for name in BUNNY_NAMES]

SHADER_NAMES = ["funcconst", "tex"]

RESOLUTION_CASES = [
    ("lowres", (160, 100)),
    ("highres", (640, 400)),
]

FOV_SCALE = 1.02
ANGLE_VERTICAL_VIEW = 90.0

# 8-bit grey scale
FULL_SCALE = 255.0

# Convergence defaults. The threshold can be given absolutely in GL
# (``--max-abs-err``) or relative to the full scale (``--max-abs-err-frac``).
DEFAULT_MAX_ABS_ERR_FRACTION = 0.01
DEFAULT_START_SSAA = 1
DEFAULT_SSAA_LIMIT = 4096
DEFAULT_NUM_CONVERGED_ITERS = 2

# Number of OpenMP threads passed explicitly to the renderer
DEFAULT_OMP_THREADS = 1

SUMMARY_NAME = "verif_3_summary.csv"
CONVERGENCE_LOG_NAME = "convergence_log.csv"
RENDER_TIME_LOG_NAME = "render_time_log.csv"

CONV_CSV_COLS = [
    "iteration",
    "subsamples",
    "rmse",
    "max_ae",
    "99p_abs_error",
    "identical_px_count",
    "tot_px_roi",
]
TIME_CSV_COLS = ["subsamples", "time (s)"]


def mesh_dir(bunny_dir: str) -> Path:
    return DATA_ROOT / "rabbits" / bunny_dir


def load_bunny(bunny_dir: str) -> RTMesh:
    """Loads a bunny surface mesh (coords/connectivity/uvs) via SimData CSVs."""
    return simdata_csv_to_rtmesh(
        directory=mesh_dir(bunny_dir),
        spatial_dim=sens.EDim.TWOD,
        world_position=np.array([0.0, 0.0, 0.0]),
    )


def build_camera(mesh: RTMesh, pixels_num: tuple[int, int]) -> Camera:
    """Frames the mesh so it fills the sensor with a small margin.

    The camera looks down ``-z`` at the mesh centroid from a distance that
    fits the mesh's bounding radius inside the vertical field of view.
    """
    coords = np.asarray(mesh.node_coords, dtype=float)
    roi_cent_world = coords.mean(axis=0)
    radius = float(np.linalg.norm(coords - roi_cent_world, axis=1).max())
    distance = FOV_SCALE * radius
    pos_world = roi_cent_world + np.array([0.0, 0.0, distance])

    return Camera(
        pixels_num[0],
        pixels_num[1],
        pos_world,
        roi_cent_world,
        ANGLE_VERTICAL_VIEW,
    )


def build_scene(mesh: RTMesh,
                camera: Camera,
                shader_name: str,
                texture: np.ndarray | None) -> Scene:
    """Builds a single-mesh scene with the requested shader applied."""
    if shader_name == "funcconst":
        mesh.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            surface_fill=np.ones(3) * 1.0,
            material_type=MaterialType.UNLIT,
        )
    elif shader_name == "tex":
        if texture is None:
            raise ValueError("Texture must be provided for the tex shader.")
        mesh.set_surface(
            surface_type=SurfType.TEXTURE,
            surface_fill=texture,
            material_type=MaterialType.UNLIT,
        )
    else:
        raise ValueError(f"Unsupported shader name: {shader_name}")

    scene = Scene()
    scene.add_camera(camera)
    scene.add_rtmesh(mesh)
    scene.set_background(np.ones(3) * 0.0)
    return scene


def render_at_subsamples(scene: Scene,
                         pixels_num: tuple[int, int],
                         subsamples: int,
                         out_dir: Path,
                         omp_threads: int) -> tuple[Path, float]:
    """Renders the scene at ``subsamples`` AA samples.

    The ray tracer writes a fixed ``rtimage_0_cam0.bmp`` file; this is renamed
    to ``rtimage_subsamples_<subsamples>.bmp`` so every refinement step is
    kept. The renderer is pinned to ``omp_threads`` OpenMP threads. Returns the
    final path and the render wall-clock time in seconds.
    """
    start = time.perf_counter()
    render_scene(
        pixels_num[1],
        pixels_num[0],
        scene,
        subsamples,
        out_dir,
        RenderType.STATIC,
        texture_sampler=TextureSampler.CATMULL_ROM,
        shading_type=ShadingType.FLAT,
        omp_thread_count=omp_threads,
    )
    elapsed = time.perf_counter() - start

    fresh_path = out_dir / "rtimage_0_cam0.bmp"
    target_path = out_dir / f"rtimage_subsamples_{subsamples}.bmp"
    if target_path.exists():
        target_path.unlink()
    fresh_path.replace(target_path)
    return target_path, elapsed


def load_uint8(path: Path) -> np.ndarray:
    """Loads an 8-bit grey render as an integer code array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def compare_images(new_path: Path, prev_path: Path) -> dict:
    """Differences two subsequent renders.

    The new render (higher sample count) is compared to the previous one and the
    per-pixel difference is summarised. All values are in 8-bit grey levels.
    """
    new_img = load_uint8(new_path)
    prev_img = load_uint8(prev_path)
    if new_img.shape != prev_img.shape:
        raise ValueError(
            f"Image shapes differ: {new_img.shape} vs {prev_img.shape}",
        )

    diff = new_img.astype(np.int32) - prev_img.astype(np.int32)
    total_pixels = int(diff.size)
    identical_count = int(np.count_nonzero(diff == 0))

    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    max_ae = float(np.max(np.abs(diff)))
    percentile_diff = float(np.percentile(np.abs(diff), 99.9))

    return {
        "rmse": rmse,
        "max_ae": max_ae,
        "99p_abs_error": percentile_diff,
        "identical_px_count": identical_count,
        "tot_px_roi": total_pixels,
    }


def run_case(bunny_name: str,
             bunny_dir: str,
             shader_name: str,
             resolution_tag: str,
             pixels_num: tuple[int, int],
             texture: np.ndarray | None,
             start_ssaa: int,
             ssaa_limit: int,
             max_abs_err: float,
             num_converged_iters: int,
             omp_threads: int) -> tuple[list[dict], list[dict]]:
    """Runs one mesh/shader/resolution convergence study.

    Returns the convergence log rows and the render-time log rows.
    """
    out_dir = VERIF_ROOT / f"c_{bunny_name}_{shader_name}_{resolution_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous outputs so a rerun does not leave stale render files
    # (including the older ssaa/diff_ssaa layout) behind.
    stale_patterns = (
        "ssaa*.bmp",
        "ssaa*.csv",
        "diff_ssaa*.bmp",
        "diff_ssaa*.csv",
        "rtimage_*.bmp",
        CONVERGENCE_LOG_NAME,
        RENDER_TIME_LOG_NAME,
    )
    for pattern in stale_patterns:
        for stale_path in out_dir.glob(pattern):
            stale_path.unlink()

    print(
        f"  Mesh: {bunny_name}, shader: {shader_name}, "
        f"resolution: {resolution_tag} {pixels_num}",
    )

    mesh = load_bunny(bunny_dir)
    camera = build_camera(mesh, pixels_num)

    convergence_rows: list[dict] = []
    time_rows: list[dict] = []

    # Baseline render at the starting sample count (nothing to compare to).
    subsamples = start_ssaa
    scene = build_scene(mesh, camera, shader_name, texture)
    prev_path, elapsed = render_at_subsamples(
        scene, pixels_num, subsamples, out_dir, omp_threads,
    )
    time_rows.append({"subsamples": subsamples, "time (s)": elapsed})
    print(f"    SS {subsamples:>6d}  baseline render      {elapsed:6.2f}s")

    iteration = 0
    converged_hits = 0

    while True:
        subsamples *= 2
        iteration += 1

        scene = build_scene(mesh, camera, shader_name, texture)
        new_path, elapsed = render_at_subsamples(
            scene, pixels_num, subsamples, out_dir, omp_threads,
        )
        time_rows.append({"subsamples": subsamples, "time (s)": elapsed})

        metrics = compare_images(new_path, prev_path)
        convergence_rows.append(
            {"iteration": iteration, "subsamples": subsamples, **metrics},
        )
        print(
            f"    SS {subsamples:>6d}  "
            f"RMSE {metrics['rmse']:9.4f} GL  "
            f"max|err| {metrics['max_ae']:7.2f} GL  "
            f"99.9p {metrics['99p_abs_error']:7.2f} GL  "
            f"{elapsed:6.2f}s",
        )

        # Advance the comparison baseline for the next refinement step.
        prev_path = new_path

        # Convergence: the first time the criterion is met one more refinement
        # step is run "for sureness"; when the criterion is met again the study
        # terminates
        if metrics["max_ae"] <= max_abs_err:
            converged_hits += 1
            if converged_hits >= num_converged_iters:
                print(
                    f"    converged: max|err| <= {max_abs_err:.4g} GL "
                    f"({num_converged_iters} criterion hits).",
                )
                break
            continue

        # Fallback: stop when the sample count becomes unreasonable.
        if subsamples >= ssaa_limit:
            print(
                f"    reached the maximum sample limit of {ssaa_limit} "
                "without meeting the convergence criterion.",
            )
            break

    with (out_dir / CONVERGENCE_LOG_NAME).open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CONV_CSV_COLS)
        writer.writeheader()
        writer.writerows(convergence_rows)

    with (out_dir / RENDER_TIME_LOG_NAME).open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TIME_CSV_COLS)
        writer.writeheader()
        writer.writerows(time_rows)

    return convergence_rows, time_rows


def write_summary(rows: list[dict]) -> None:
    summary_path = VERIF_ROOT / SUMMARY_NAME
    fieldnames = [
        "mesh_name",
        "shader_name",
        "resolution",
        "pixels_x",
        "pixels_y",
        *CONV_CSV_COLS,
    ]
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=BUNNY_NAMES,
        choices=BUNNY_NAMES,
        help="Bunny meshes to render (default: all).",
    )
    parser.add_argument(
        "--shaders",
        nargs="+",
        default=SHADER_NAMES,
        choices=SHADER_NAMES,
        help="Surface shaders to render (default: all).",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=[tag for tag, _ in RESOLUTION_CASES],
        choices=[tag for tag, _ in RESOLUTION_CASES],
        help="Resolution tags to render (default: all).",
    )
    parser.add_argument(
        "--start-ssaa",
        type=int,
        default=DEFAULT_START_SSAA,
        help="Starting anti-aliasing sample count (default: 1).",
    )
    parser.add_argument(
        "--ssaa-limit",
        type=int,
        default=DEFAULT_SSAA_LIMIT,
        help=(
            "Maximum anti-aliasing sample count before giving up "
            f"(default: {DEFAULT_SSAA_LIMIT})."
        ),
    )
    parser.add_argument(
        "--max-abs-err",
        type=float,
        default=None,
        help=(
            "Absolute convergence threshold in 8-bit grey levels. If omitted, "
            "--max-abs-err-frac is used instead."
        ),
    )
    parser.add_argument(
        "--max-abs-err-frac",
        type=float,
        default=DEFAULT_MAX_ABS_ERR_FRACTION,
        help=(
            "Convergence threshold as a fraction of the full 8-bit range "
            f"(default: {DEFAULT_MAX_ABS_ERR_FRACTION})."
        ),
    )
    parser.add_argument(
        "--num-converged-iters",
        type=int,
        default=DEFAULT_NUM_CONVERGED_ITERS,
        help=(
            "Number of criterion hits (not necessarily consecutive) required "
            "before stopping; after the criterion is first met, one further "
            "refinement step is run for sureness "
            f"(default: {DEFAULT_NUM_CONVERGED_ITERS})."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_OMP_THREADS,
        help=(
            "Number of OpenMP threads the renderer is pinned to. Defaults to 1 "
            f"(default: {DEFAULT_OMP_THREADS})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.start_ssaa < 1:
        raise ValueError("--start-ssaa must be >= 1")
    if args.ssaa_limit < args.start_ssaa:
        raise ValueError("--ssaa-limit must be >= --start-ssaa")
    if args.num_converged_iters < 1:
        raise ValueError("--num-converged-iters must be >= 1")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")

    max_abs_err = (
        args.max_abs_err
        if args.max_abs_err is not None
        else args.max_abs_err_frac * FULL_SCALE
    )

    VERIF_ROOT.mkdir(parents=True, exist_ok=True)

    print(
        f"Convergence criterion: max|err| <= {max_abs_err:.4g} GL "
        f"({max_abs_err / FULL_SCALE * 100.0:.4g}% of full scale), "
        f"terminating after {args.num_converged_iters} criterion hits",
    )
    print(
        f"Anti-aliasing samples: start at {args.start_ssaa}, doubling up to a "
        f"maximum of {args.ssaa_limit}",
    )
    print(f"Renderer pinned to {args.threads} OpenMP thread(s)")

    print("Loading speckle texture...")
    texture = ImageTools.load_image_greyscale(TEXTURE_PATH)

    resolution_map = dict(RESOLUTION_CASES)
    summary_rows: list[dict] = []
    for bunny_name, bunny_dir in zip(BUNNY_NAMES, BUNNY_DIRS):
        if bunny_name not in args.meshes:
            continue
        print(f"Processing bunny mesh: {bunny_name}")
        for shader_name in args.shaders:
            for resolution_tag in args.resolutions:
                pixels_num = resolution_map[resolution_tag]
                convergence_rows, _ = run_case(
                    bunny_name,
                    bunny_dir,
                    shader_name,
                    resolution_tag,
                    pixels_num,
                    texture,
                    args.start_ssaa,
                    args.ssaa_limit,
                    max_abs_err,
                    args.num_converged_iters,
                    args.threads,
                )
                for row in convergence_rows:
                    summary_rows.append(
                        {
                            "mesh_name": bunny_name,
                            "shader_name": shader_name,
                            "resolution": resolution_tag,
                            "pixels_x": pixels_num[0],
                            "pixels_y": pixels_num[1],
                            **row,
                        },
                    )

    if summary_rows:
        write_summary(summary_rows)

    print("Convergence study complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
