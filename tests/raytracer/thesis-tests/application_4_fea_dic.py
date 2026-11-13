#!/usr/bin/env python3
"""
Application test 4: stereo calibration and 3D DIC analysis.

This script consumes the calibration and stereo-DIC images rendered by
``application_4_fea.py``, calibrates the stereo rig from the rendered dot
targets, builds a specimen ROI, and runs Pyvale's stereo (3D) DIC engine.

The output for each case is written to::

    thesis-output/app4_fea/<case>/dic3d/
        stereo_calibration.txt      calibration parameters (pyvale text format)
        calibration_summary.csv     intrinsics, baseline, reprojection errors
        calibration_residuals.png   per-pose reprojection RMSE
        dots_cam0_*.txt             matched target points per accepted pair
        dots_cam1_*.txt
        grid_*.txt                  target object coordinates per accepted pair
        roi_mask.txt                DIC ROI mask (0/1, one row per image row)
        roi_overlay.png             reference image with ROI, hole and seed
        seed.txt                    region-growing seed
        dic3d_rtimage_{0,1}_cam0.csv pyvale stereo DIC results
        dic3d_summary.csv           DIC summary for the deformed frame
        dic3d_displacement.png      u, v, w and |d| maps

Usage::

    /home/wiera/anaconda3/envs/pyvale/bin/python application_4_fea_dic.py
    ... application_4_fea_dic.py --cases air_diffuse pipe
    ... application_4_fea_dic.py --cases water --force

The dot detector is relaxed relative to the pyvale default because the
rendered targets are small (the dots are about 15--27 px across).  The light
(hollow) dots in the water case are additionally restricted to the dark-dot
target region, which removes bright reflections near the image corners (this is a concern for lower SSAA numbers).
The amended dot detector follows the following workflow:
1. Original calibration image
2. Detect dark calibration dots
    - Are there >= 10 dots?
    - No: Detect light dots in entire image
    - Yes: 
        - Extract dark-dot coordinates
        - Calculate bounding box
        - Add 5% padding
        - Generate binary mask
        - Detect light dots inside the mask
        - Return detected keypoints
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyvale import calib, dic
import pyvale.calib.calibdotdetect as calibdotdetect

# ================================================================================
# Layout and plotting constants
#
# These deliberately duplicate the small subset of global_utils that this
# analysis needs.  global_utils imports pyvale.raytracer, which is not part of
# newer Pyvale version used by this script (TODO: merge newer Pyvale into raytarcer).
# ================================================================================

BASE_TEST_DIR = Path(__file__).resolve().parent.parent / "thesis-output"
CASE_NAMES = ("air_diffuse", "pipe", "water")
CASE_BASE_DIR = "app4_fea/ssaa_2_8"
THESIS_DATA_DIR = Path(__file__).resolve().parent.parent / "thesis-data"
CAL_TARGET_ACCESS = THESIS_DATA_DIR / "texture/cal_target-simple.tiff"
PLATE_COORDS_ACCESS = (
    THESIS_DATA_DIR / "plate_hole/platehole3d_2mr_63f/coords.csv"
)
CMAP_DIC_FEA = "coolwarm"
FIGURE_SIZE_DOUBLE_CMAP = (15, 10)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "text.usetex": True,
    "font.family": "serif",
})


def spatial_scale(fov_size: float, image_size: int) -> float:
    """Return the physical size of one image pixel [mm/px]."""
    return fov_size / image_size


def test_dir(base_dir: Path, name: str) -> Path:
    """Create and return a named output directory."""
    path = Path(base_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def measure_cal_target_geometry(texture_path: Path) -> tuple[float, float]:
    """
    Measure the calibration target width and dot spacing [px] from its texture.

    The texture is the 12 x 9 dark-dot target with three hollow dots.  The
    dark dots are found as connected components and the spacing is fitted to
    their centroids.
    """
    image = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(
            f"Calibration target texture not found: {texture_path}"
        )

    _, mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    points = np.array([
        centroids[i] for i in range(1, count)
        if 100 <= stats[i, cv2.CC_STAT_AREA] <= 10000
    ])
    expected = GRID_WIDTH * GRID_HEIGHT
    if len(points) != expected:
        raise RuntimeError(
            f"Expected {expected} calibration dots in {texture_path}, "
            f"found {len(points)}"
        )

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    x_step = (x_max - x_min) / (GRID_WIDTH - 1)
    y_step = (y_max - y_min) / (GRID_HEIGHT - 1)
    columns = np.rint((points[:, 0] - x_min) / x_step).astype(int)
    rows = np.rint((points[:, 1] - y_min) / y_step).astype(int)
    if (np.unique(columns).size != GRID_WIDTH
            or np.unique(rows).size != GRID_HEIGHT):
        raise RuntimeError(
            f"Could not resolve the {GRID_WIDTH}x{GRID_HEIGHT} dot grid "
            f"in {texture_path}"
        )

    spacing_x = float(np.polyfit(columns, points[:, 0], 1)[0])
    spacing_y = float(np.polyfit(rows, points[:, 1], 1)[0])
    if abs(spacing_x - spacing_y) > 1.0:
        raise RuntimeError(
            f"Inconsistent dot spacing in {texture_path}: "
            f"{spacing_x:.3f} px vs {spacing_y:.3f} px"
        )
    return float(image.shape[1]), 0.5 * (spacing_x + spacing_y)


def measure_plate_geometry(coords_path: Path,
                           plate_height_mm: float,
                           plate_base_y_mm: float
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Measure the plate half-size, front-face centre and hole radius.

    The SimData coordinates are in metres.  They are scaled so that the plate
    height matches application_4_fea.py.  The plate is then placed with
    Anchor.BASE at (0, plate_base_y_mm, 0): x and z are centred on the target,
    and the minimum y sits on the target.
    """
    coords = np.loadtxt(coords_path, delimiter=",")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(
            f"Expected N x 3 coordinates in {coords_path}, got {coords.shape}"
        )

    y_span_m = coords[:, 1].max() - coords[:, 1].min()
    if y_span_m <= 0.0:
        raise RuntimeError(f"Degenerate plate y extent in {coords_path}")
    coords_mm = coords * (plate_height_mm / y_span_m)

    xy_min = coords_mm[:, :2].min(axis=0)
    xy_max = coords_mm[:, :2].max(axis=0)
    half_size = np.round(0.5 * (xy_max - xy_min), 9)
    centre_xy = 0.5 * (xy_min + xy_max)
    hole_radius = float(np.round(
        np.linalg.norm(coords_mm[:, :2] - centre_xy, axis=1).min(), 9
    ))

    z_span = coords_mm[:, 2].max() - coords_mm[:, 2].min()
    # Anchor.BASE centres x/z on the placement target and puts y_min on it.
    placement_target = np.array([0.0, plate_base_y_mm, 0.0])
    front_centre = np.round(placement_target + np.array([
        0.0,
        half_size[1],
        0.5 * z_span,
    ]), 9)
    return half_size, front_centre, hole_radius


STEREOCAL_DIRNAME = "stereocal"
STEREODIC_DIRNAME = "stereodic"
DIC3D_DIRNAME = "dic3d"

# The rendered calibration target is the 12 x 9 dot target with three hollow
# dots.  The (x, y) indices follow the target texture: y increases down the
# texture image, so (2, 6) is bottom-left and (2, 2), (9, 2) are on the upper
# row.  This is the pattern detected in the images rendered by
# application_4_fea.py.
GRID_HEIGHT = 9
GRID_WIDTH = 12
HOLLOW_DOTS = [(2, 6), (2, 2), (9, 2)]

# The target rectangle spans the full calibration texture.  Both the texture
# width and the dot spacing are measured from that texture, so the physical
# dot spacing follows from the target length in calplate/coords.csv.
CAL_TEXTURE_WIDTH_PX, CAL_DOT_SPACING_PX = measure_cal_target_geometry(
    CAL_TARGET_ACCESS)

# Nominal camera model used by application_4_fea.py: 50 mm lens, 20.48 mm
# sensor and a 1024 x 1024 full frame.  The rendered images are crops of that
# frame, so the principal point stays at (512, 512) and fx = fy = 2500 px.
NOMINAL_FX = 50.0 / (20.48 / 1024.0) # f_x = f/s_x
NOMINAL_CX = 512.0
NOMINAL_CY = 512.0
NOMINAL_BASELINE_MM = 8.475792207426851
MM_PER_PX = spatial_scale(40.0, 1024)          # 0.0390625 mm/px

# Specimen geometry.  application_4_fea.py places the 25 x 35 x 1 mm plate
# with Anchor.BASE at (0, pipe_bottom_inner_y, 0), where
# pipe_bottom_inner_y = -23 + 4 mm.  The half-size, front-face centre and hole
# radius are measured from the plate mesh coordinates.
PLATE_HEIGHT_MM = 35.0
PLATE_BASE_Y_MM = -19.0
PLATE_HALF_SIZE_MM, PLATE_CENTRE_WORLD_MM, HOLE_RADIUS_MM = (
    measure_plate_geometry(PLATE_COORDS_ACCESS, PLATE_HEIGHT_MM,
                           PLATE_BASE_Y_MM)
)
# z is the camera-facing front surface used for the nominal projection.
PLATE_CENTRE_Z_MM = float(PLATE_CENTRE_WORLD_MM[2])

PLATE_INNER_MARGIN_PX = 12.0   # keep subsets away from the plate edge
IMAGE_EDGE_MARGIN_PX = 12.0    # keep subsets away from the image edge
HOLE_EXCLUSION_FACTOR = 1.5    # hole radius multiplier before subset margin
EPI_MARGIN_PX = 20             # extra epipolar search beyond the disparity
SEED_ERODE_EXTRA_PX = 5        # extra erosion used to choose a valid seed

# Calibration acceptance thresholds.  The rendered AIR calibration is at
# about 0.04 px; the through-pipe calibration is at about 0.1--0.3 px.  The
# through-water calibration cannot be represented by a pinhole model and is
# rejected by these checks.
MAX_REPROJ_MEAN_PX = 0.5
FX_RATIO_RANGE = (0.5, 1.5)
BASELINE_RATIO_RANGE = (0.5, 1.5)
MAX_STEREO_ANGLE_DEG = 15.0


# ================================================================================
# Data containers
# ================================================================================

@dataclass
class CalibrationInfo:
    calibration: calib.Calib
    dots_cam0: list[np.ndarray]
    dots_cam1: list[np.ndarray]
    grid: list[np.ndarray]
    filenames_cam0: list[str]
    filenames_cam1: list[str]
    errors_cam0: np.ndarray
    errors_cam1: np.ndarray
    grid_spacing_mm: float
    image_dims: tuple[int, int]
    passed: bool
    notes: list[str]

    @property
    def dot_counts(self) -> list[int]:
        return [len(dots) for dots in self.dots_cam0]


# ================================================================================
# Dot detection
# ================================================================================

def _make_blob_detector(light: bool,
                        min_area: float = 18.0,
                        max_area: float = 10000.0,
                        min_circularity: float = 0.6,
                        min_inertia: float = 0.05,
                        min_distance: float = 10.0) -> cv2.SimpleBlobDetector:
    """Return a blob detector tuned for the small rendered calibration dots."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = True
    params.minCircularity = min_circularity
    params.filterByInertia = True
    params.minInertiaRatio = min_inertia
    params.filterByColor = True
    params.blobColor = 255 if light else 0
    params.minDistBetweenBlobs = min_distance
    return cv2.SimpleBlobDetector_create(params)


class _MaskedLightDetector:
    """
    Light-dot detector restricted to the region occupied by the dot target.

    Bright reflections on the water tank and pipe walls can produce extra
    light blobs near the corners of the image.  The dark dots define the
    target region unambiguously, so the light detector is evaluated with a
    rectangular mask around their robust bounding box.  This keeps the three
    hollow dots and rejects the corner reflections.
    """

    def __init__(self) -> None:
        self._light = _make_blob_detector(light=True)
        self._dark = _make_blob_detector(light=False)

    def detect(self, image: np.ndarray):
        dark_keypoints = self._dark.detect(image)

        # If the dark detector fails, do not risk throwing away valid dots.
        if len(dark_keypoints) < 10:
            return self._light.detect(image)

        points = np.asarray([kp.pt for kp in dark_keypoints], dtype=np.float64)
        x0, x1 = np.percentile(points[:, 0], [1.0, 99.0])
        y0, y1 = np.percentile(points[:, 1], [1.0, 99.0])

        pad = 0.05 * max(x1 - x0, y1 - y0)
        x0 = max(0, int(np.floor(x0 - pad)))
        x1 = min(image.shape[1], int(np.ceil(x1 + pad)))
        y0 = max(0, int(np.floor(y0 - pad)))
        y1 = min(image.shape[0], int(np.ceil(y1 + pad)))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        return self._light.detect(image, mask)


def _install_relaxed_blob_detectors() -> None:
    """Replace the package detector with the relaxed, target-aware version."""
    def factory(light: bool):
        return _MaskedLightDetector() if light else _make_blob_detector(False)

    calibdotdetect.create_blob_detector = factory


# ================================================================================
# Calibration
# ================================================================================

def calibration_grid_spacing(case_dir: Path) -> float:
    """Return the physical dot spacing [mm] for this rendered target."""
    coords_path = case_dir / "calplate" / "coords.csv"
    if not coords_path.is_file():
        raise FileNotFoundError(
            f"Calibration target coordinates not found: {coords_path}. "
            "Run application_4_fea.py with render_cal=True first."
        )

    coords = np.loadtxt(coords_path, delimiter=",")
    target_length_mm = float(coords[:, 0].max() - coords[:, 0].min())
    return target_length_mm * CAL_DOT_SPACING_PX / CAL_TEXTURE_WIDTH_PX


def assess_calibration(calib_params: calib.Calib,
                       errors_cam0: np.ndarray,
                       errors_cam1: np.ndarray,
                       image_dims: tuple[int, int]) -> tuple[bool, list[str]]:
    """Check the calibration against nominal rig and reprojection limits."""
    width, height = image_dims
    notes: list[str] = []
    checks: dict[str, bool] = {}

    if errors_cam0.size and errors_cam1.size:
        mean0 = float(np.mean(errors_cam0))
        mean1 = float(np.mean(errors_cam1))
        checks["reprojection"] = (
            mean0 <= MAX_REPROJ_MEAN_PX and mean1 <= MAX_REPROJ_MEAN_PX
        )
        if not checks["reprojection"]:
            notes.append(
                f"mean reprojection RMSE {mean0:.3f}/{mean1:.3f} px "
                f"exceeds {MAX_REPROJ_MEAN_PX:.1f} px"
            )
    else:
        checks["reprojection"] = True

    intr = (calib_params.cam0, calib_params.cam1)
    checks["intrinsics"] = all(
        0.0 <= cam.cx <= width
        and 0.0 <= cam.cy <= height
        and FX_RATIO_RANGE[0] * NOMINAL_FX <= cam.fx <= FX_RATIO_RANGE[1] * NOMINAL_FX
        for cam in intr
    )
    if not checks["intrinsics"]:
        notes.append(
            "implausible intrinsics: "
            f"fx=({calib_params.cam0.fx:.1f}, {calib_params.cam1.fx:.1f}) px, "
            f"cx=({calib_params.cam0.cx:.1f}, {calib_params.cam1.cx:.1f}) px, "
            f"cy=({calib_params.cam0.cy:.1f}, {calib_params.cam1.cy:.1f}) px "
            f"for a {width}x{height} image"
        )

    baseline = float(np.linalg.norm(calib_params.translation))
    baseline_ratio = baseline / NOMINAL_BASELINE_MM
    checks["baseline"] = (
        BASELINE_RATIO_RANGE[0] <= baseline_ratio <= BASELINE_RATIO_RANGE[1]
    )
    if not checks["baseline"]:
        notes.append(
            f"baseline {baseline:.2f} mm is {baseline_ratio:.2f}x the nominal "
            f"{NOMINAL_BASELINE_MM:.2f} mm"
        )

    stereo_angle = float(calib_params.rotation[1])
    checks["stereo_angle"] = abs(stereo_angle) <= MAX_STEREO_ANGLE_DEG
    if not checks["stereo_angle"]:
        notes.append(
            f"stereo angle {stereo_angle:.2f} deg is outside "
            f"+/-{MAX_STEREO_ANGLE_DEG:.1f} deg"
        )

    return all(checks.values()), notes


def calibrate_case(case_dir: Path,
                   out_dir: Path,
                   *,
                   optimize_distortion: bool,
                   calibration_path: Path | None,
                   num_threads: int | None) -> CalibrationInfo:
    """Detect the target dots and calibrate the stereo rig for one case."""
    cal_dir = case_dir / STEREOCAL_DIRNAME
    if not cal_dir.is_dir():
        raise FileNotFoundError(
            f"Calibration image directory not found: {cal_dir}. "
            "Run application_4_fea.py with render_cal=True first."
        )

    sample_path = cal_dir / "rtimage_0_cam0.bmp"
    sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    if sample is None:
        raise FileNotFoundError(
            f"Could not read calibration image: {sample_path}"
        )
    image_height, image_width = sample.shape
    image_dims = (image_width, image_height)
    grid_spacing_mm = calibration_grid_spacing(case_dir)

    if calibration_path is not None:
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"Supplied calibration file not found: {calibration_path}"
            )
        calib_params = calib.loadtxt(calibration_path, delimiter=",")
        calib.savetxt(calib=calib_params,
                      path=out_dir / "stereo_calibration.txt",
                      delimiter=",")
        return CalibrationInfo(
            calibration=calib_params,
            dots_cam0=[],
            dots_cam1=[],
            grid=[],
            filenames_cam0=[],
            filenames_cam1=[],
            errors_cam0=np.array([], dtype=float),
            errors_cam1=np.array([], dtype=float),
            grid_spacing_mm=grid_spacing_mm,
            image_dims=image_dims,
            passed=True,
            notes=[
                f"loaded calibration from {calibration_path}; "
                "reprojection errors were not re-evaluated"
            ],
        )

    _install_relaxed_blob_detectors()

    print(f"Detecting calibration dots in {cal_dir} ...")
    dots_cam0, dots_cam1, grid, filenames0, filenames1 = calib.detect_dots(
        cam0=str(cal_dir / "rtimage_*_cam0.bmp"),
        cam1=str(cal_dir / "rtimage_*_cam1.bmp"),
        grid_height=GRID_HEIGHT,
        grid_width=GRID_WIDTH,
        grid_spacing=grid_spacing_mm,
        hollow_dots=HOLLOW_DOTS,
        min_dot_fraction=0.5,
        visualisationCV2=False,
        visualisationPLT=False,
    )

    if not dots_cam0:
        raise RuntimeError(
            f"No calibration image pairs were accepted for {case_dir.name}. "
            "Check the rendered target and the dot detector."
        )

    print(f"Accepted {len(dots_cam0)} calibration pairs; "
          f"dots per pair = {min(len(d) for d in dots_cam0)}.."
          f"{max(len(d) for d in dots_cam0)}")

    calib_params, errors_cam0, errors_cam1 = calib.calibrate_stereo(
        dots_cam0=dots_cam0,
        dots_cam1=dots_cam1,
        grid=grid,
        img_dims=[image_width, image_height],
        filenames=filenames0,
        optimize_distortion=optimize_distortion,
        num_threads=num_threads,
        error_formulation="RMSE",
    )

    calib.savetxt(calib=calib_params,
                  path=out_dir / "stereo_calibration.txt",
                  delimiter=",")

    for points0, points1, grid_points, filename0, filename1 in zip(
            dots_cam0, dots_cam1, grid, filenames0, filenames1):
        stem0 = Path(filename0).stem
        stem1 = Path(filename1).stem
        np.savetxt(out_dir / f"dots_cam0_{stem0}.txt", points0)
        np.savetxt(out_dir / f"dots_cam1_{stem1}.txt", points1)
        np.savetxt(out_dir / f"grid_{stem0}.txt", grid_points)

    passed, notes = assess_calibration(calib_params, errors_cam0,
                                       errors_cam1, image_dims)

    return CalibrationInfo(
        calibration=calib_params,
        dots_cam0=dots_cam0,
        dots_cam1=dots_cam1,
        grid=grid,
        filenames_cam0=filenames0,
        filenames_cam1=filenames1,
        errors_cam0=errors_cam0,
        errors_cam1=errors_cam1,
        grid_spacing_mm=grid_spacing_mm,
        image_dims=image_dims,
        passed=passed,
        notes=notes,
    )


def print_calibration(info: CalibrationInfo) -> None:
    cal = info.calibration
    print(f"  cam0: fx={cal.cam0.fx:.2f} fy={cal.cam0.fy:.2f} "
          f"cx={cal.cam0.cx:.2f} cy={cal.cam0.cy:.2f}")
    print(f"  cam1: fx={cal.cam1.fx:.2f} fy={cal.cam1.fy:.2f} "
          f"cx={cal.cam1.cx:.2f} cy={cal.cam1.cy:.2f}")
    print(f"  baseline: {np.linalg.norm(cal.translation):.3f} mm; "
          f"translation: {np.array2string(cal.translation, precision=3)} mm")
    print(f"  rotation (cam0 -> cam1): "
          f"{np.array2string(cal.rotation, precision=3)} deg")

    if info.errors_cam0.size:
        print(f"  reprojection RMSE: "
              f"cam0 mean={np.mean(info.errors_cam0):.4f} "
              f"max={np.max(info.errors_cam0):.4f} px; "
              f"cam1 mean={np.mean(info.errors_cam1):.4f} "
              f"max={np.max(info.errors_cam1):.4f} px")
    else:
        print("  reprojection RMSE: not available")

    print(f"  quality: {'PASS' if info.passed else 'FAIL'}")
    for note in info.notes:
        print(f"    - {note}")


# ================================================================================
# ROI and seed
# ================================================================================

def load_camera_geometry(case_dir: Path) -> tuple[np.ndarray, float]:
    """Read the nominal camera-0 position and baseline from stereo_geometry.csv."""
    geometry_path = case_dir / "stereo_geometry.csv"
    if not geometry_path.is_file():
        raise FileNotFoundError(
            f"Nominal stereo geometry not found: {geometry_path}. "
            "Run application_4_fea.py first."
        )

    geometry: dict[str, float] = {}
    with open(geometry_path, newline="") as file:
        for row in csv.reader(file):
            if len(row) >= 2:
                geometry[row[0].strip()] = float(row[1])

    cam0 = np.array([
        geometry["cam0_x_mm"],
        geometry["cam0_y_mm"],
        geometry["cam0_z_mm"],
    ], dtype=float)
    baseline = float(geometry.get("baseline_len_mm", NOMINAL_BASELINE_MM))
    return cam0, baseline


def build_roi(ref_image: np.ndarray,
              case: str,
              case_dir: Path,
              info: CalibrationInfo,
              subset_size: int) -> tuple[np.ndarray, list[int], dict[str, float | int]]:
    """
    Build the specimen ROI and a valid reliability-guided seed.

    The specimen is projected using the nominal camera model rather than the
    calibration intrinsics.
    """
    height, width = ref_image.shape
    cam0, baseline = load_camera_geometry(case_dir)

    depth = cam0[2] - PLATE_CENTRE_Z_MM
    if depth <= 0.0:
        raise RuntimeError(f"Invalid camera depth: {depth:.3f} mm")

    # Nominal image projection of the specimen centre and size.
    centre_x = NOMINAL_CX + NOMINAL_FX * (
        PLATE_CENTRE_WORLD_MM[0] - cam0[0]) / depth
    centre_y = NOMINAL_CY + NOMINAL_FX * (
        -(PLATE_CENTRE_WORLD_MM[1] - cam0[1])) / depth
    half_width = NOMINAL_FX * PLATE_HALF_SIZE_MM[0] / depth
    half_height = NOMINAL_FX * PLATE_HALF_SIZE_MM[1] / depth
    hole_radius_px = NOMINAL_FX * HOLE_RADIUS_MM / depth

    if not (0.0 <= centre_x < width and 0.0 <= centre_y < height):
        raise RuntimeError(
            "Nominal specimen projection lies outside the image: "
            f"({centre_x:.1f}, {centre_y:.1f}) for {width}x{height}"
        )

    # Stereo overlap: a cam0 point at x has its cam1 correspondence near
    # x - disparity.  Use the calibrated disparity when the calibration is
    # trustworthy, otherwise use the nominal rig.
    nominal_disparity = NOMINAL_FX * baseline / depth
    if info.passed:
        calibrated_disparity = (
            info.calibration.cam0.fx
            * float(np.linalg.norm(info.calibration.translation))
            / depth
        )
    else:
        calibrated_disparity = nominal_disparity

    if 0.0 < calibrated_disparity < 0.9 * width:
        disparity = calibrated_disparity
    else:
        disparity = nominal_disparity

    subset_half = subset_size // 2

    x_min_pad = 0
    match case:
        case "air_diffuse":
            x_min_pad = 180
        case "pipe":
            x_min_pad = 100
        case "water":
            x_min_pad = 50
        case _:
            x_min_pad = 0




    x_min = int(np.ceil(max(
        centre_x - half_width + PLATE_INNER_MARGIN_PX,
        disparity + EPI_MARGIN_PX,
        IMAGE_EDGE_MARGIN_PX,
    ))) - x_min_pad # move left x boundary more to the left
    x_max = int(np.floor(min(
        centre_x + half_width - PLATE_INNER_MARGIN_PX,
        width - 1 - IMAGE_EDGE_MARGIN_PX,
    )))
    y_min = int(np.ceil(max(
        centre_y - half_height + PLATE_INNER_MARGIN_PX,
        IMAGE_EDGE_MARGIN_PX,
    )))
    y_max = int(np.floor(min(
        centre_y + half_height - PLATE_INNER_MARGIN_PX,
        height - 1 - IMAGE_EDGE_MARGIN_PX,
    )))

    if x_min >= x_max or y_min >= y_max:
        raise RuntimeError(
            "The specimen ROI is empty after applying the stereo overlap "
            f"and image margins: x={x_min}..{x_max}, y={y_min}..{y_max}"
        )

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max + 1, x_min:x_max + 1] = 1

    # Exclude the circular hole.  The multiplier and subset margin keep the
    # correlation windows away from the hole edge and its refraction blur.
    hole_exclusion_px = (
        HOLE_EXCLUSION_FACTOR * hole_radius_px + subset_half + 5.0
    )
    cv2.circle(mask,
               (int(round(centre_x)), int(round(centre_y))),
               int(np.ceil(hole_exclusion_px)),
               0, -1)

    # The engine requires full subsets inside the mask, so erode the ROI by
    # half a subset plus a small margin.
    erode_radius = subset_half + 3
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1))
    mask = cv2.erode(mask, kernel)

    if not np.any(mask):
        raise RuntimeError(
            "The ROI mask is empty after erosion.  The hole exclusion or "
            "subset size is too large for the specimen."
        )

    # Choose a high-contrast seed from a further-eroded mask.  The extra
    # erosion guarantees that the seed's full subset lies inside the mask
    # passed to the DIC engine (the engine also snaps the seed to the step
    # grid, so leave a few pixels of margin).
    seed_erode_radius = subset_half + SEED_ERODE_EXTRA_PX
    seed_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * seed_erode_radius + 1, 2 * seed_erode_radius + 1))
    valid_seed_region = cv2.erode(mask, seed_kernel)
    if not np.any(valid_seed_region):
        raise RuntimeError(
            "No valid seed region remains after eroding the ROI.  Reduce the "
            "hole exclusion or the subset size."
        )

    image_float = ref_image.astype(np.float32)
    local_mean = cv2.blur(image_float, (15, 15))
    local_std = np.sqrt(np.maximum(
        cv2.blur(image_float * image_float, (15, 15))
        - local_mean * local_mean,
        0.0,
    ))

    ys, xs = np.where(valid_seed_region)
    values = local_std[ys, xs]
    candidates = values >= np.percentile(values, 60.0)
    centroid_x = xs.mean()
    centroid_y = ys.mean()
    distance_sq = (
        (xs[candidates] - centroid_x) ** 2
        + (ys[candidates] - centroid_y) ** 2
    )
    best = int(np.argmin(distance_sq))
    seed = [int(xs[candidates][best]), int(ys[candidates][best])]

    roi: dict[str, float | int] = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "roi_pixels": int(np.count_nonzero(mask)),
        "plate_centre_x_px": float(centre_x),
        "plate_centre_y_px": float(centre_y),
        "hole_radius_px": float(hole_radius_px),
        "hole_exclusion_px": float(hole_exclusion_px),
        "disparity_px": float(disparity),
        "seed_x": seed[0],
        "seed_y": seed[1],
    }
    return mask, seed, roi


def save_roi(out_dir: Path,
             ref_image: np.ndarray,
             mask: np.ndarray,
             seed: list[int],
             roi: dict[str, float | int]) -> None:
    np.savetxt(out_dir / "roi_mask.txt", mask, fmt="%d", delimiter=" ")
    np.savetxt(out_dir / "seed.txt", np.asarray(seed), fmt="%d")

    overlay = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay,
                  (int(roi["x_min"]), int(roi["y_min"])),
                  (int(roi["x_max"]), int(roi["y_max"])),
                  (0, 255, 0), 2)
    cv2.circle(overlay,
               (int(round(roi["plate_centre_x_px"])),
                int(round(roi["plate_centre_y_px"]))),
               int(np.ceil(roi["hole_radius_px"])),
               (0, 255, 255), 2)
    cv2.circle(overlay,
               (int(round(roi["plate_centre_x_px"])),
                int(round(roi["plate_centre_y_px"]))),
               int(np.ceil(roi["hole_exclusion_px"])),
               (0, 0, 255), 2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
    cv2.circle(overlay, tuple(seed), 5, (255, 0, 0), -1)
    cv2.imwrite(str(out_dir / "roi_overlay.png"), overlay)


def print_roi(roi: dict[str, float | int]) -> None:
    print(f"  ROI: x={roi['x_min']}..{roi['x_max']}, "
          f"y={roi['y_min']}..{roi['y_max']}, "
          f"{roi['roi_pixels']} subset-centre pixels")
    print(f"  stereo disparity estimate: {roi['disparity_px']:.1f} px; "
          f"hole exclusion radius: {roi['hole_exclusion_px']:.1f} px")
    print(f"  seed: ({roi['seed_x']}, {roi['seed_y']})")


# ================================================================================
# DIC
# ================================================================================

def _dic_result_files(out_dir: Path) -> list[Path]:
    """Return only the pyvale DIC result files, not the summary CSV."""
    return sorted(out_dir.glob("dic3d_rtimage_*_cam0.csv"))


def run_dic(case_dir: Path,
            out_dir: Path,
            info: CalibrationInfo,
            mask: np.ndarray,
            seed: list[int],
            roi: dict[str, float | int],
            *,
            subset_size: int,
            subset_step: int,
            max_displacement: int,
            epi_search_distance: int | None,
            epi_distance_threshold: int,
            num_threads: int | None,
            print_level: int) -> tuple[int, list[Path]]:
    dic_dir = case_dir / STEREODIC_DIRNAME
    paths = {
        "ref0": dic_dir / "rtimage_0_cam0.bmp",
        "ref1": dic_dir / "rtimage_0_cam1.bmp",
        "def0": dic_dir / "rtimage_1_cam0.bmp",
        "def1": dic_dir / "rtimage_1_cam1.bmp",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Stereo DIC image not found ({label}): {path}. "
                "Run application_4_fea.py with render_dic=True first."
            )

    # Remove results from a previous run so import_3d cannot pick up stale
    # frames with a different ROI or frame count.
    for old_file in out_dir.glob("dic3d_*.csv"):
        old_file.unlink()

    if epi_search_distance is None:
        epi_search_distance = int(np.ceil(roi["disparity_px"] + EPI_MARGIN_PX))
    epi_search_distance = int(np.clip(
        epi_search_distance, 20, max(mask.shape[1] - 1, 20)))

    print(f"Running stereo DIC: subset={subset_size}, step={subset_step}, "
          f"max_disp={max_displacement} px, epi_search={epi_search_distance} px")

    dic.calculate_3d(
        reference=[paths["ref0"], paths["ref1"]],
        deformed=[paths["def0"], paths["def1"]],
        calibration=info.calibration,
        roi_mask=mask.astype(bool),
        seed=seed,
        subset_size=subset_size,
        subset_step=subset_step,
        max_displacement=max_displacement,
        epi_search_distance=epi_search_distance,
        epi_distance_threshold=int(epi_distance_threshold),
        method="MULTIWINDOW_RG",
        output_basepath=out_dir,
        output_prefix="dic3d_",
        num_threads=num_threads,
        print_level=print_level,
    )

    result_files = _dic_result_files(out_dir)
    if not result_files:
        raise RuntimeError(
            f"Stereo DIC finished but no result files were written to {out_dir}"
        )
    return epi_search_distance, result_files


def summarise_dic(out_dir: Path) -> dict[str, float | int | str]:
    files = _dic_result_files(out_dir)
    if not files:
        raise RuntimeError(f"No stereo DIC results found in {out_dir}")

    deformed_file = next(
        (path for path in files if "rtimage_1_cam0" in path.name),
        files[-1],
    )
    reference_file = next(
        (path for path in files if "rtimage_0_cam0" in path.name),
        files[0],
    )

    data = np.genfromtxt(deformed_file, delimiter=",", names=True)
    if data.dtype.names is None:
        raise RuntimeError(f"Could not read DIC result columns: {deformed_file}")

    def column(name: str) -> np.ndarray:
        return np.asarray(data[name], dtype=np.float64)

    u = column("stereo_disp_u_mm")
    v = column("stereo_disp_v_mm")
    w = column("stereo_disp_w_mm")
    x = column("stereo_x_mm")
    y = column("stereo_y_mm")
    z = column("stereo_z_mm")

    converged = (
        (column("converged").astype(int) == 1)
        & (column("stereo_converged").astype(int) == 1)
    )
    finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    valid = converged & finite
    if not np.any(valid):
        valid = finite
    if not np.any(valid):
        raise RuntimeError(f"No finite DIC displacements in {deformed_file}")

    magnitude = np.sqrt(u * u + v * v + w * w)

    def median_or_nan(values: np.ndarray) -> float:
        return float(np.nanmedian(values[valid])) if np.any(valid) else np.nan

    def min_or_nan(values: np.ndarray) -> float:
        return float(np.nanmin(values[valid])) if np.any(valid) else np.nan

    def max_or_nan(values: np.ndarray) -> float:
        return float(np.nanmax(values[valid])) if np.any(valid) else np.nan

    return {
        "subsets": int(len(u)),
        "converged_fraction": float(np.mean(converged)),
        "u_median_mm": median_or_nan(u),
        "u_min_mm": min_or_nan(u),
        "u_max_mm": max_or_nan(u),
        "v_median_mm": median_or_nan(v),
        "v_min_mm": min_or_nan(v),
        "v_max_mm": max_or_nan(v),
        "w_median_mm": median_or_nan(w),
        "w_min_mm": min_or_nan(w),
        "w_max_mm": max_or_nan(w),
        "mag_median_mm": median_or_nan(magnitude),
        "mag_max_mm": max_or_nan(magnitude),
        "max_disp_px": max_or_nan(magnitude) / MM_PER_PX,
        "x_min_mm": min_or_nan(x),
        "x_max_mm": max_or_nan(x),
        "y_min_mm": min_or_nan(y),
        "y_max_mm": max_or_nan(y),
        "z_min_mm": min_or_nan(z),
        "z_max_mm": max_or_nan(z),
        "reference_file": reference_file.name,
        "deformed_file": deformed_file.name,
    }


def print_dic_summary(summary: dict[str, float | int | str]) -> None:
    print(f"  3D DIC: {summary['subsets']} subsets, "
          f"{100.0 * float(summary['converged_fraction']):.2f}% converged")
    print(f"    u: median={summary['u_median_mm']:+.5f} mm "
          f"[{summary['u_min_mm']:+.5f}, {summary['u_max_mm']:+.5f}]")
    print(f"    v: median={summary['v_median_mm']:+.5f} mm "
          f"[{summary['v_min_mm']:+.5f}, {summary['v_max_mm']:+.5f}]")
    print(f"    w: median={summary['w_median_mm']:+.5f} mm "
          f"[{summary['w_min_mm']:+.5f}, {summary['w_max_mm']:+.5f}]")
    print(f"    |d|: median={summary['mag_median_mm']:.5f} mm, "
          f"max={summary['mag_max_mm']:.5f} mm "
          f"({summary['max_disp_px']:.3f} px at {MM_PER_PX:.6f} mm/px)")


# ================================================================================
# CSV and plots
# ================================================================================

def _write_row_csv(path: Path, row: dict[str, object]) -> None:
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def calibration_row(case: str, info: CalibrationInfo) -> dict[str, object]:
    cal = info.calibration
    counts = info.dot_counts
    baseline = float(np.linalg.norm(cal.translation))
    if info.errors_cam0.size:
        reproj_mean0 = float(np.mean(info.errors_cam0))
        reproj_max0 = float(np.max(info.errors_cam0))
        reproj_mean1 = float(np.mean(info.errors_cam1))
        reproj_max1 = float(np.max(info.errors_cam1))
    else:
        reproj_mean0 = reproj_max0 = np.nan
        reproj_mean1 = reproj_max1 = np.nan

    return {
        "case": case,
        "accepted_pairs": len(info.dots_cam0),
        "min_dots_per_pair": int(min(counts)) if counts else 0,
        "max_dots_per_pair": int(max(counts)) if counts else 0,
        "grid_spacing_mm": info.grid_spacing_mm,
        "image_width_px": info.image_dims[0],
        "image_height_px": info.image_dims[1],
        "fx0_px": cal.cam0.fx,
        "fy0_px": cal.cam0.fy,
        "cx0_px": cal.cam0.cx,
        "cy0_px": cal.cam0.cy,
        "fx1_px": cal.cam1.fx,
        "fy1_px": cal.cam1.fy,
        "cx1_px": cal.cam1.cx,
        "cy1_px": cal.cam1.cy,
        "tx_mm": cal.translation[0],
        "ty_mm": cal.translation[1],
        "tz_mm": cal.translation[2],
        "baseline_mm": baseline,
        "rx_deg": cal.rotation[0],
        "ry_deg": cal.rotation[1],
        "rz_deg": cal.rotation[2],
        "reproj_mean0_px": reproj_mean0,
        "reproj_max0_px": reproj_max0,
        "reproj_mean1_px": reproj_mean1,
        "reproj_max1_px": reproj_max1,
        "quality_passed": int(info.passed),
        "quality_notes": "; ".join(info.notes),
    }


DIC_SUMMARY_FIELDS = [
    "case", "dic_run", "dic_error",
    "subsets", "converged_fraction",
    "u_median_mm", "u_min_mm", "u_max_mm",
    "v_median_mm", "v_min_mm", "v_max_mm",
    "w_median_mm", "w_min_mm", "w_max_mm",
    "mag_median_mm", "mag_max_mm", "max_disp_px",
    "x_min_mm", "x_max_mm", "y_min_mm", "y_max_mm",
    "z_min_mm", "z_max_mm",
    "roi_x_min", "roi_x_max", "roi_y_min", "roi_y_max",
    "roi_pixels", "hole_radius_px", "hole_exclusion_px",
    "disparity_px", "epi_search_px",
    "seed_x", "seed_y",
    "reference_file", "deformed_file",
]


def dic_row(case: str,
            summary: dict[str, float | int | str] | None,
            roi: dict[str, float | int] | None,
            epi_search_px: int | None,
            error: str = "") -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in DIC_SUMMARY_FIELDS}
    row["case"] = case
    row["dic_run"] = int(summary is not None)
    row["dic_error"] = error

    if summary is not None:
        for key, value in summary.items():
            if key in row:
                row[key] = value

    if roi is not None:
        row["roi_x_min"] = roi["x_min"]
        row["roi_x_max"] = roi["x_max"]
        row["roi_y_min"] = roi["y_min"]
        row["roi_y_max"] = roi["y_max"]
        row["roi_pixels"] = roi["roi_pixels"]
        row["hole_radius_px"] = roi["hole_radius_px"]
        row["hole_exclusion_px"] = roi["hole_exclusion_px"]
        row["disparity_px"] = roi["disparity_px"]
        row["seed_x"] = roi["seed_x"]
        row["seed_y"] = roi["seed_y"]

    if epi_search_px is not None:
        row["epi_search_px"] = epi_search_px

    return row


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save a figure, falling back to mathtext if LaTeX is unavailable."""
    try:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    except RuntimeError:
        plt.rcParams["text.usetex"] = False
        fig.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def plot_calibration_residuals(case: str,
                               info: CalibrationInfo,
                               out_dir: Path) -> None:
    if not info.errors_cam0.size:
        return

    poses = [int(Path(name).stem.split("_")[1]) for name in info.filenames_cam0]
    order = np.argsort(poses)
    poses_sorted = np.asarray(poses, dtype=int)[order]
    errors0 = np.asarray(info.errors_cam0, dtype=float)[order]
    errors1 = np.asarray(info.errors_cam1, dtype=float)[order]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.plot(poses_sorted, errors0, "o-", label="cam0")
    ax.plot(poses_sorted, errors1, "s-", label="cam1")
    ax.set_xticks(poses_sorted)
    ax.set_xlabel("Calibration pose")
    ax.set_ylabel("Reprojection RMSE [px]")
    ax.set_title(f"Calibration residuals: {case.replace('_', r'\_')}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "calibration_residuals.png")


def plot_dic_results(case: str, out_dir: Path) -> None:
    files = _dic_result_files(out_dir)
    if not files:
        return

    results = dic.import_3d(
        data=files,
        delimiter=",",
        binary=False,
        layout="matrix",
        print_level=0,
    )
    stereo = results.stereo
    if stereo is None:
        return

    frame = 1 if stereo.u_mm.shape[0] > 1 else 0
    u = stereo.u_mm[frame]
    v = stereo.v_mm[frame]
    w = stereo.w_mm[frame]
    magnitude = np.sqrt(u * u + v * v + w * w)

    components = [
        (r"$u$", u, r"$u$ [mm]", True),
        (r"$v$", v, r"$v$ [mm]", True),
        (r"$w$", w, r"$w$ [mm]", True),
        (r"$|\mathbf{d}|$", magnitude, r"$|\mathbf{d}|$ [mm]", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_DOUBLE_CMAP)
    for ax, (label, values, colorbar_label, diverging) in zip(
            axes.ravel(), components):
        masked = np.ma.masked_invalid(values)
        finite_values = np.asarray(masked.compressed())
        if finite_values.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue

        if diverging:
            limit = float(np.max(np.abs(finite_values)))
            limit = max(limit, 1e-6)
            vmin, vmax = -limit, limit
            cmap = CMAP_DIC_FEA
        else:
            vmin = 0.0
            vmax = max(float(np.max(finite_values)), 1e-6)
            cmap = "viridis"

        image = ax.pcolor(results.ss_x, results.ss_y, masked,
                          cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel(r"$x$ [px]")
        ax.set_ylabel(r"$y$ [px]")
        ax.set_title(label)

        colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
        colorbar.set_label(colorbar_label)

    fig.suptitle(f"Stereo DIC: {case.replace('_', r'\_')}, deformed frame")
    fig.tight_layout()
    _save_figure(fig, out_dir / "dic3d_displacement.png")


# ================================================================================
# Case driver
# ================================================================================

DIC_OUTPUT_PATTERNS = (
    "dic3d_*.csv",
    "dic3d_displacement.png",
    "roi_mask.txt",
    "roi_overlay.png",
    "seed.txt",
    "dots_*.txt",
    "grid_*.txt",
    "calibration_summary.csv",
    "calibration_residuals.png",
)


def clear_dic_outputs(out_dir: Path) -> None:
    """Remove outputs from a previous analysis run in this directory."""
    for pattern in DIC_OUTPUT_PATTERNS:
        for path in out_dir.glob(pattern):
            path.unlink()


def process_case(case: str,
                 case_dir: Path,
                 args: argparse.Namespace) -> None:
    out_dir = test_dir(case_dir, DIC3D_DIRNAME)
    clear_dic_outputs(out_dir)
    print(f"Output directory: {out_dir}")

    info = calibrate_case(
        case_dir,
        out_dir,
        optimize_distortion=args.optimize_distortion,
        calibration_path=args.calibration,
        num_threads=args.num_threads,
    )
    print_calibration(info)
    if case == "pipe":
        print("  Note: the through-pipe calibration is a virtual pinhole fit;")
        print("  absolute 3D coordinates are not physical, but the relative")
        print("  displacement field may still be consistent with the rendered deformation (not verified yet).")
    _write_row_csv(out_dir / "calibration_summary.csv",
                   calibration_row(case, info))
    if not args.no_plots:
        plot_calibration_residuals(case, info, out_dir)

    if not info.passed and not args.force:
        message = (
            "calibration quality check failed; 3D DIC skipped "
            "(use --force to run it anyway)"
        )
        print(f"  {message}")
        _write_row_csv(out_dir / "dic3d_summary.csv",
                       dic_row(case, None, None, None, message))
        return

    if not info.passed:
        print("  --force: running 3D DIC despite the calibration warnings")
        print("  The through-media refraction is not represented by the "
              "pinhole model; treat the results with caution.")

    dic_dir = case_dir / STEREODIC_DIRNAME
    ref_image_path = dic_dir / "rtimage_0_cam0.bmp"
    ref_image = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
    if ref_image is None:
        raise FileNotFoundError(
            f"Could not read stereo DIC reference image: {ref_image_path}. "
            "Run application_4_fea.py with render_dic=True first."
        )

    mask, seed, roi = build_roi(ref_image, case, case_dir, info,
                                args.subset_size)
    print_roi(roi)
    save_roi(out_dir, ref_image, mask, seed, roi)

    epi_search_px: int | None = None
    summary: dict[str, float | int | str] | None = None
    error = ""
    try:
        epi_search_px, _ = run_dic(
            case_dir,
            out_dir,
            info,
            mask,
            seed,
            roi,
            subset_size=args.subset_size,
            subset_step=args.subset_step,
            max_displacement=args.max_displacement,
            epi_search_distance=args.epi_search_distance,
            epi_distance_threshold=args.epi_distance_threshold,
            num_threads=args.num_threads,
            print_level=args.print_level,
        )
        summary = summarise_dic(out_dir)
        print_dic_summary(summary)
    except Exception as exc:  # report and keep going with the other cases
        error = f"{type(exc).__name__}: {exc}"
        print(f"  3D DIC failed: {error}")

    _write_row_csv(out_dir / "dic3d_summary.csv",
                   dic_row(case, summary, roi, epi_search_px, error))

    if summary is not None and not args.no_plots:
        plot_dic_results(case, out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    case_names = list(CASE_NAMES)
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the stereo rig and run 3D DIC on the images rendered "
            "by application_4_fea.py."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=case_names,
        choices=case_names,
        help=f"Cases to analyse (default: {' '.join(case_names)})",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=31,
        help="DIC subset size in pixels; must be odd (default: 31)",
    )
    parser.add_argument(
        "--subset-step",
        type=int,
        default=5,
        help="DIC subset step in pixels (default: 5)",
    )
    parser.add_argument(
        "--max-displacement",
        type=int,
        default=20,
        help="Maximum temporal displacement in pixels (default: 20)",
    )
    parser.add_argument(
        "--epi-search-distance",
        type=int,
        default=None,
        help=(
            "Stereo search distance along the epipolar line in pixels "
            "(default: disparity + 20 px)"
        ),
    )
    parser.add_argument(
        "--epi-distance-threshold",
        type=int,
        default=5,
        help="Maximum epipolar distance in pixels (default: 5)",
    )
    parser.add_argument(
        "--optimize-distortion",
        action="store_true",
        help=(
            "Optimise camera distortion during calibration.  The rendered "
            "pinhole cameras have zero distortion, and the through-media "
            "refraction is not a radial distortion, so this is off by default."
        ),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Use this pyvale calibration file for all cases and skip dot "
            "detection (reprojection errors are not re-evaluated)"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run 3D DIC even if the calibration quality check fails",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the matplotlib figures",
    )
    parser.add_argument(
        "--print-level",
        type=int,
        default=0,
        help="DIC engine print level (default: 0)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for calibration and DIC (default: all)",
    )
    args = parser.parse_args(argv)

    if args.subset_size <= 0 or args.subset_size % 2 == 0:
        parser.error("--subset-size must be a positive odd number")
    if args.subset_step <= 0:
        parser.error("--subset-step must be positive")
    if args.subset_step > args.subset_size:
        parser.error("--subset-step must not exceed --subset-size")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    for case in args.cases:
        print()
        print("=" * 78)
        print(f"Case: {case}")
        print("=" * 78)
        case_dir = Path(BASE_TEST_DIR) / CASE_BASE_DIR / case
        if not case_dir.is_dir():
            print(f"Case output directory not found: {case_dir}")
            print("Run application_4_fea.py first.")
            continue
        try:
            process_case(case, case_dir, args)
        except Exception as exc:  # keep processing the remaining cases
            print(f"Case {case} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
