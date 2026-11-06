#!/usr/bin/env python3
"""Verification case 4: depth testing / occlusion (visibility).

Two surface meshes are placed in the same scene: a front mesh drawn with a constant grey value of
1.0 (255 GL) and a back mesh drawn with a constant grey value of 0.5
(≈128 GL). The meshes are offset so they only partially overlap in the image
plane and the back mesh is pushed behind the front mesh along the camera axis.
Three images are produced for each case:

* ``both.bmp``      - both meshes present;
* ``frontonly.bmp`` - the front mesh on its own;
* ``diff.bmp``      - per-pixel ``both - frontonly``.

The verification quantity is the number of pixels rendered at the front mesh's
constant grey level. If the back mesh is correctly occluded by the front mesh,
adding it must not change the front mesh's visible footprint, so
``count(both) - count(frontonly) == 0``. A negative difference indicates the
back mesh is incorrectly drawn in front of the occluding front mesh.

Output layout (self-contained under ``verif_bench/verif/verif_4``)::

    d_<mesh>_<case>/
        both.bmp        both meshes
        frontonly.bmp   front mesh only
        diff.bmp        signed difference (zero maps to mid-grey, 128)
    verif_4_summary.csv per-case occlusion counts

The corresponding paper figures are produced by ``paper_verif_4.py``.
"""

from __future__ import annotations

import argparse
import csv
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

import pyvale.sensorsim as sens


VERIF_ROOT = Path(__file__).resolve().parent / "verif" / "verif_4"
DATA_ROOT = Path(__file__).resolve().parent / "data"

# Geometry and rendering settings
OVERLAP_X_DEFAULT = 0.85
OVERLAP_Y_DEFAULT = 0.8
BEHIND_FACT_DEFAULT = 1.05
FOV_SCALE = 1.01
ANGLE_VERTICAL_VIEW = 90.0
PIXELS_NUM = (1200, 600)
AA_SAMPLES = 1

# Constant shader values and the matching 8-bit grey levels.
FRONT_VALUE = 1.0
BACK_VALUE = 0.5
BACKGROUND = 0.0
FRONT_GREY = 255
BACK_GREY = 128

CASE_NAMES = ["rabbit", "sphere"]
RABBIT_MESH_NAMES = ["tri3", "tri6", "quad4", "quad8", "quad9"]

# Default pairing between the front element type and the back element type
PAIRED_BACK = {
    "tri3": "tri6",
    "tri6": "tri3",
    "quad4": "quad8",
    "quad8": "quad4",
    "quad9": "quad4",
}

SUMMARY_NAME = "verif_4_summary.csv"
SUMMARY_COLS = [
    "case",
    "mesh_name",
    "pixels_x",
    "pixels_y",
    "count_front",
    "count_both",
    "diff",
]


def paired_back(mesh_name: str) -> str:
    return PAIRED_BACK[mesh_name]


def case_dirs(case_name: str, mesh_name: str) -> tuple[Path, Path, object]:
    """Returns the front/back data directories and the rabbit flip rotation."""
    back_name = paired_back(mesh_name)
    if case_name == "rabbit":
        front_dir = DATA_ROOT / "rabbits" / f"riley_{mesh_name}"
        back_dir = DATA_ROOT / "rabbits" / f"feebs_{back_name}"
        # Both rabbit meshes are rotated 180 degrees about y.
        rotation = make_axis_rotation(np.array([0.0, 1.0, 0.0]), 180.0)
    elif case_name == "sphere":
        front_dir = DATA_ROOT / "bench" / f"{sphere_dir_name(mesh_name)}_sphere200"
        back_dir = DATA_ROOT / "bench" / f"{sphere_dir_name(back_name)}_sphere200"
        rotation = None
    else:
        raise ValueError(f"Unsupported case name: {case_name}")
    return front_dir, back_dir, rotation


def sphere_dir_name(mesh_name: str) -> str:
    return "quad4ibi" if mesh_name == "quad4" else mesh_name


def load_mesh(directory: Path) -> RTMesh:
    """Loads a static surface mesh (coords/connectivity/uvs) from SimData CSVs."""
    return simdata_csv_to_rtmesh(
        directory=directory,
        spatial_dim=sens.EDim.TWOD,
        world_position=np.array([0.0, 0.0, 0.0]),
    )


def load_grey(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_diff_bmp(path: Path, diff: np.ndarray) -> None:
    """Saves the signed difference so that zero is mid-grey (128)."""
    shifted = np.clip(diff.astype(np.int32) + BACK_GREY, 0, 255).astype(np.uint8)
    Image.fromarray(shifted).save(path, format="BMP")


def render(camera: Camera,
           meshes: list[tuple[RTMesh, float]],
           pixels_num: tuple[int, int],
           out_dir: Path,
           stem: str) -> np.ndarray:
    """Renders the meshes at a constant grey value and returns the image."""
    for mesh, value in meshes:
        mesh.set_surface(
            surface_type=SurfType.FIELD_COLOR,
            surface_fill=np.ones(3) * value,
            material_type=MaterialType.UNLIT,
        )

    scene = Scene()
    scene.add_camera(camera)
    scene.add_rtmesh([mesh for mesh, _ in meshes])
    scene.set_background(np.ones(3) * BACKGROUND)

    render_scene(
        pixels_num[1],
        pixels_num[0],
        scene,
        AA_SAMPLES,
        out_dir,
        RenderType.STATIC,
        texture_sampler=TextureSampler.CATMULL_ROM,
        shading_type=ShadingType.FLAT,
    )

    fresh_path = out_dir / "rtimage_0_cam0.bmp"
    target_path = out_dir / f"{stem}.bmp"
    if target_path.exists():
        target_path.unlink()
    fresh_path.replace(target_path)
    return load_grey(target_path)


def run_case(case_name: str,
             mesh_name: str,
             pixels_num: tuple[int, int],
             overlap_x: float,
             overlap_y: float,
             behind_fact: float) -> dict:
    out_dir = VERIF_ROOT / f"d_{mesh_name}_{case_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in out_dir.glob("*.bmp"):
        stale_path.unlink()

    front_dir, back_dir, rotation = case_dirs(case_name, mesh_name)
    print(f"  Case: {case_name}, mesh: {mesh_name}")
    print(f"    front: {front_dir.name}, back: {back_dir.name}")

    front = load_mesh(front_dir)
    back = load_mesh(back_dir)

    if rotation is not None:
        front.rotate(rotation)
        back.rotate(rotation)

    # Offset the meshes so they only partially overlap in the image plane.
    extent = front.node_coords.max(axis=0) - front.node_coords.min(axis=0)
    x_sep = float(extent[0]) * (1.0 - overlap_x)
    y_sep = float(extent[1]) * (1.0 - overlap_y)
    front.translate(np.array([0.5 * x_sep, -0.5 * y_sep, 0.0]))
    back.translate(np.array([-0.5 * x_sep, 0.5 * y_sep, 0.0]))

    # Frame both meshes together: place the camera along +z at a distance of
    # FOV_SCALE times the radius of the combined point cloud about its centroid.
    both_coords = np.vstack([front.node_coords, back.node_coords])
    roi_cent_world = both_coords.mean(axis=0)
    radius = float(np.linalg.norm(both_coords - roi_cent_world, axis=1).max())
    pos_world = roi_cent_world + np.array([0.0, 0.0, FOV_SCALE * radius])
    camera = Camera(
        pixels_num[0],
        pixels_num[1],
        pos_world,
        roi_cent_world,
        ANGLE_VERTICAL_VIEW,
    )

    # Push the back mesh behind the front mesh along the camera axis by a
    # fraction of the camera-to-front distance.
    cam_axis = pos_world - roi_cent_world
    cam_axis = cam_axis / np.linalg.norm(cam_axis)
    front_centroid = front.node_coords.mean(axis=0)
    front_dist = float(np.dot(pos_world - front_centroid, cam_axis))
    behind_extra = (behind_fact - 1.0) * front_dist
    back.translate(-cam_axis * behind_extra)

    both = render(
        camera,
        [(front, FRONT_VALUE), (back, BACK_VALUE)],
        pixels_num,
        out_dir,
        "both",
    )
    frontonly = render(
        camera,
        [(front, FRONT_VALUE)],
        pixels_num,
        out_dir,
        "frontonly",
    )

    diff = both.astype(np.int32) - frontonly.astype(np.int32)
    save_diff_bmp(out_dir / "diff.bmp", diff)

    count_front = int(np.count_nonzero(frontonly == FRONT_GREY))
    count_both = int(np.count_nonzero(both == FRONT_GREY))
    print(
        f"    front pixels: {count_front}, both pixels: {count_both}, "
        f"diff: {count_both - count_front}",
    )

    return {
        "case": case_name,
        "mesh_name": mesh_name,
        "pixels_x": pixels_num[0],
        "pixels_y": pixels_num[1],
        "count_front": count_front,
        "count_both": count_both,
        "diff": count_both - count_front,
    }


def write_summary(rows: list[dict]) -> None:
    summary_path = VERIF_ROOT / SUMMARY_NAME
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=CASE_NAMES,
        choices=CASE_NAMES,
        help="Geometry cases to render (default: rabbit sphere).",
    )
    parser.add_argument(
        "--meshes",
        nargs="+",
        default=RABBIT_MESH_NAMES,
        choices=RABBIT_MESH_NAMES,
        help="Front-mesh element types to render (default: all).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=PIXELS_NUM[0],
        help=f"Image width in pixels (default: {PIXELS_NUM[0]}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=PIXELS_NUM[1],
        help=f"Image height in pixels (default: {PIXELS_NUM[1]}).",
    )
    parser.add_argument(
        "--overlap-x",
        type=float,
        default=OVERLAP_X_DEFAULT,
        help=f"In-plane overlap fraction along x (default: {OVERLAP_X_DEFAULT}).",
    )
    parser.add_argument(
        "--overlap-y",
        type=float,
        default=OVERLAP_Y_DEFAULT,
        help=f"In-plane overlap fraction along y (default: {OVERLAP_Y_DEFAULT}).",
    )
    parser.add_argument(
        "--behind-fact",
        type=float,
        default=BEHIND_FACT_DEFAULT,
        help=(
            "Factor by which the camera-to-front distance is scaled to push the "
            f"back mesh behind the front mesh (default: {BEHIND_FACT_DEFAULT})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.width < 1 or args.height < 1:
        raise ValueError("Image dimensions must be >= 1.")
    if args.behind_fact < 1.0:
        raise ValueError("--behind-fact must be >= 1.0.")

    pixels_num = (args.width, args.height)
    VERIF_ROOT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for case_name in args.cases:
        for mesh_name in args.meshes:
            rows.append(
                run_case(
                    case_name,
                    mesh_name,
                    pixels_num,
                    args.overlap_x,
                    args.overlap_y,
                    args.behind_fact,
                )
            )

    if rows:
        write_summary(rows)

    print("Depth/occlusion study complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
