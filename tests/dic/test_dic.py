"""
================================================================================
example: thermocouples on a 2d plate

pyvale: the python validation engine
license: mit
copyright (c) 2024 the computer aided validation team
================================================================================
"""
from collections.abc import Generator
import glob
import os
from pathlib import Path
import shutil

os.environ["OMP_NUM_THREADS"] = "1"

from PIL import Image
import numpy as np
import pytest
import pyvale.dic as dic
import pyvale.data as dataset
import pyvale.calib as calib


TEST_DIR = Path(__file__).resolve().parent

ref0 = dataset.dic_plate_rigid_cam0_ref()
ref1 = dataset.dic_plate_rigid_cam1_ref()
def0 = dataset.dic_plate_rigid_cam0_def_small()
def1 = dataset.dic_plate_rigid_cam1_def_small()

ref0_hydro = dataset.dic_plate_with_hydro_cam0_ref()
ref1_hydro = dataset.dic_plate_with_hydro_cam1_ref()
def0_hydro = dataset.dic_plate_with_hydro_cam0_def()
def1_hydro = dataset.dic_plate_with_hydro_cam1_def()

def0_25px = dataset.dic_plate_rigid_cam0_def_25px()
def0_50px = dataset.dic_plate_rigid_cam0_def_50px()


calib_file = TEST_DIR / "calib.txt"
calib_data = calib.loadtxt(calib_file)

def_large = [def0_25px, def0_50px]

roi = dic.RegionOfInterest(ref_image=ref0)
roi.rect_region(x=100, y=100, size_x=200, size_y=200)

# ------------------------------------------------------------------------------
# Images
# ------------------------------------------------------------------------------

ref_image = Image.open(ref0)

if isinstance(def0, list):
    files = sorted(def0)
else:
    files = sorted(def0.parent.glob(def0.name))

ref_arr = np.array(ref_image)

# First deformed image used for intensity scaling tests
def_image = Image.open(files[4]) # 0.8 px displacement
def_arr = np.array(def_image)

original_dtype = def_arr.dtype

scale = 0.5
offset = 50

def_arr_float = def_arr.astype(np.float32)

def_arr_scaled = (def_arr_float * scale).astype(original_dtype)
def_arr_scaled_offset = (def_arr_float * scale + offset).astype(original_dtype)

# ------------------------------------------------------------------------------
# Ground truth displacements
# ------------------------------------------------------------------------------

u = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
u_short = [0.0, 0.5, 1.0]


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

def _clean_dir_contents(path: Path) -> None:
    if not path.is_dir():
        return
    for item in path.glob("*"):
        if item.is_file() or item.is_symlink():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            shutil.rmtree(item, ignore_errors=True)


@pytest.fixture
def dic_temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a clean temporary output directory before and after each test."""
    _clean_dir_contents(tmp_path)
    yield tmp_path
    _clean_dir_contents(tmp_path)


@pytest.fixture(autouse=True)
def clean_repo_dic_dir() -> Generator[None, None, None]:
    """Purge any stale CSV outputs in tests/dic before and after every test."""
    patterns = ("test_*.csv", "dic_results_*.csv", "test_fft_*.csv")
    for pattern in patterns:
        for f in TEST_DIR.glob(pattern):
            f.unlink(missing_ok=True)
    yield
    for pattern in patterns:
        for f in TEST_DIR.glob(pattern):
            f.unlink(missing_ok=True)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def validate_col(
    csv_file: str | Path,
    gt: float,
    col: int,
    rtol: float,
    atol: float,
) -> None:
    dic_data = np.loadtxt(csv_file, skiprows=1, delimiter=",")
    gt_u = np.full(dic_data.shape[0], gt)

    np.testing.assert_allclose(
        gt_u,
        dic_data[:, col],
        rtol=rtol,
        atol=atol,
        err_msg=f"Displacement mismatch for {gt} px",
    )


def validate(
    output_pattern: str | Path,
    gt: list[float],
    atol: float,
    rtol: float = 0.0,
    atol_stereo: float = 0.001,
    stereo: bool = False,
) -> None:
    output_files = sorted(glob.glob(str(output_pattern)))

    assert len(output_files) == len(gt), (
        f"Expected {len(gt)} output files but found {len(output_files)}"
    )

    for gt_i, output_file in zip(gt, output_files):
        # check horizontal displacement PIXELS
        validate_col(output_file, gt_i, 2, rtol, atol)

        # check vertical displacement PIXELS
        validate_col(output_file, -1.0 * gt_i, 3, rtol, atol)

        if stereo:
            # check horizontal displacement MM
            validate_col(output_file, -0.01 * gt_i, 13, rtol, atol_stereo)

            # check vertical displacement MM
            validate_col(output_file, 0.01 * gt_i, 14, rtol, atol_stereo)


def validate_hydro(
    output_pattern: str | Path,
    gt: list[float],
    atol: float,
    rtol: float = 0.0,
) -> None:
    output_files = sorted(glob.glob(str(output_pattern)))

    assert len(output_files) == len(gt), (
        f"Expected {len(gt)} output files but found {len(output_files)}"
    )

    for edge_disp, output_file in zip(gt, output_files):
        dic_data = np.loadtxt(output_file, skiprows=1, delimiter=",")

        x = dic_data[:, 0]
        y = dic_data[:, 1]
        u = dic_data[:, 2]
        v = dic_data[:, 3]

        cx = (20 + 1019) / 2.0
        cy = (20 + 1519) / 2.0
        width = 999.0
        height = 1499.0

        u_gt = 2.0 * edge_disp * (x - cx) / width
        v_gt = 2.0 * edge_disp * (y - cy) / height

        np.testing.assert_allclose(
            u,
            u_gt,
            rtol=rtol,
            atol=atol,
            err_msg=f"Horizontal displacement mismatch ({edge_disp} px)",
        )
        np.testing.assert_allclose(
            v,
            v_gt,
            rtol=rtol,
            atol=atol,
            err_msg=f"Vertical displacement mismatch ({edge_disp} px)",
        )


# ------------------------------------------------------------------------------
# SSD
# ------------------------------------------------------------------------------

def test_2d_ssd_rigid(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="SSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_ssd_rigid_",
    )

    output_files = dic_temp_dir / "test_ssd_rigid_*.csv"
    validate(output_pattern=output_files, gt=u, atol=0.01)


# ------------------------------------------------------------------------------
# NSSD
# ------------------------------------------------------------------------------

def test_2d_nssd_scaled_image_rigid(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref_arr,
        deformed=def_arr_scaled,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="NSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_nssd_scaled_image_rigid_",
    )

    output_file = (
        dic_temp_dir
        / "test_2d_nssd_scaled_image_rigid_def_img_0000.csv"
    )

    validate(output_pattern=output_file, gt=[0.8], atol=0.01, stereo=False)


# ------------------------------------------------------------------------------
# ZNSSD
# ------------------------------------------------------------------------------

def test_2d_znssd_scaled_offset_image_rigid(
    dic_temp_dir: Path,
) -> None:
    dic.calculate_2d(
        reference=ref_arr,
        deformed=def_arr_scaled_offset,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_znssd_scaled_offset_image_rigid_",
    )

    output_file = (
        dic_temp_dir
        / "test_2d_znssd_scaled_offset_image_rigid_def_img_0000.csv"
    )

    validate(output_pattern=output_file, gt=[0.8], atol=0.01, stereo=False)


# ------------------------------------------------------------------------------
# Raster ZNSSD Affine
# ------------------------------------------------------------------------------

def test_2d_image_scan_znssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="RASTER",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_image_scan_znssd_affine_",
    )

    validate(
        dic_temp_dir / "test_2d_image_scan_znssd_affine_*.csv",
        u,
        atol=0.01,
    )


# ------------------------------------------------------------------------------
# Raster ZNSSD Rigid
# ------------------------------------------------------------------------------

def test_2d_image_scan_znssd_rigid(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="RIGID",
        method="RASTER",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_image_scan_znssd_rigid_",
    )

    validate(
        dic_temp_dir / "test_2d_image_scan_znssd_rigid_*.csv",
        u,
        atol=0.01,
    )


# ------------------------------------------------------------------------------
# Raster NSSD
# ------------------------------------------------------------------------------

def test_2d_image_scan_nssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="NSSD",
        shape_function="AFFINE",
        method="RASTER",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_image_scan_nssd_affine_",
    )

    validate(
        dic_temp_dir / "test_2d_image_scan_nssd_affine_*.csv",
        u,
        atol=0.01,
    )


# ------------------------------------------------------------------------------
# Multiwindow RG
# ------------------------------------------------------------------------------

def test_2d_rg_znssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_rg_znssd_affine_",
    )

    validate(
        dic_temp_dir / "test_2d_rg_znssd_affine_*.csv",
        u,
        atol=0.01,
    )

# ------------------------------------------------------------------------------
# Multiwindow RG
# ------------------------------------------------------------------------------

def test_2d_rg_znssd_quad(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="QUAD",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_rg_znssd_quad_",
    )

    validate(
        dic_temp_dir / "test_2d_rg_znssd_quad_*.csv",
        u,
        atol=0.01,
    )

# ------------------------------------------------------------------------------
# singlewindow RG
# ------------------------------------------------------------------------------

def test_2d_singlewindow_znssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_2d_rg_znssd_affine_",
    )

    validate(
        dic_temp_dir / "test_2d_rg_znssd_affine_*.csv",
        u,
        atol=0.01,
    )

# ------------------------------------------------------------------------------
# Large displacement FFT with mutlwindow
# ------------------------------------------------------------------------------

def test_2d_multiwindow_fft_large(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def_large,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=100,
        correlation_criteria="ZNSSD",
        shape_function="RIGID",
        method="MULTIWINDOW",
        output_basepath=dic_temp_dir,
        output_prefix="test_fft_",
    )

    outputs = [
        ("test_fft_rigid_cam0_frame12.csv", 25.0),
        ("test_fft_rigid_cam0_frame13.csv", 50.0),
    ]

    for filename, disp in outputs:
        output_file = dic_temp_dir / filename
        validate(output_file, [disp], atol=0.01)

# ------------------------------------------------------------------------------
# Large displacement FFT with mutlwindow
# ------------------------------------------------------------------------------

def test_2d_multiwindow_rg_fft_large(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def_large,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=100,
        correlation_criteria="ZNSSD",
        shape_function="RIGID",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_fft_",
    )

    outputs = [
        ("test_fft_rigid_cam0_frame12.csv", 25.0),
        ("test_fft_rigid_cam0_frame13.csv", 50.0),
    ]

    for filename, disp in outputs:
        output_file = dic_temp_dir / filename
        validate(output_file, [disp], atol=0.01)

# ------------------------------------------------------------------------------
# Large displacement FFT with singlewindow
# ------------------------------------------------------------------------------

def test_2d_singlewindow_fft_large(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0,
        deformed=def_large,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=100,
        correlation_criteria="ZNSSD",
        shape_function="RIGID",
        method="SINGLEWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_fft_",
    )

    outputs = [
        ("test_fft_rigid_cam0_frame12.csv", 25.0),
        ("test_fft_rigid_cam0_frame13.csv", 50.0),
    ]

    for filename, disp in outputs:
        output_file = dic_temp_dir / filename
        validate(output_file, [disp], atol=0.01)


# ------------------------------------------------------------------------------
# Multiwindow RG
# ------------------------------------------------------------------------------

def test_2d_hydro_rg_znssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0_hydro,
        deformed=def0_hydro,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=21,
        subset_step=10,
        max_displacement=2,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_hydro_",
    )

    validate_hydro(
        dic_temp_dir / "test_hydro_*.csv",
        u_short,
        atol=0.005,
    )

# ------------------------------------------------------------------------------
# Multiwindow RG
# ------------------------------------------------------------------------------

def test_2d_hydro_rg_znssd_quad(dic_temp_dir: Path) -> None:
    dic.calculate_2d(
        reference=ref0_hydro,
        deformed=def0_hydro,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=21,
        subset_step=10,
        max_displacement=2,
        correlation_criteria="ZNSSD",
        shape_function="QUAD",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_hydro_",
    )

    validate_hydro(
        dic_temp_dir / "test_hydro_*.csv",
        u_short,
        atol=0.008,  # more noise for quad
    )

# ------------------------------------------------------------------------------
# STEREO
# ------------------------------------------------------------------------------

def test_3d_rg_znssd_affine(dic_temp_dir: Path) -> None:
    dic.calculate_3d(
        reference=[ref0, ref1],
        deformed=[def0, def1],
        roi_mask=roi.mask,
        calibration=calib_data,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_3d_rg_znssd_affine_",
    )

    validate(
        dic_temp_dir / "test_3d_rg_znssd_affine_*.csv",
        u,
        atol=0.01,
        atol_stereo=0.0001,
        stereo=True,
    )


def test_3d_rg_znssd_affine_incremental(dic_temp_dir: Path) -> None:
    dic.calculate_3d(
        reference=[ref0, ref1],
        deformed=[def0, def1],
        roi_mask=roi.mask,
        calibration=calib_data,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="AFFINE",
        incremental_update="IMAGE",
        incremental_update_value=1,
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_3d_rg_znssd_incremental_affine_",
    )

    validate(
        dic_temp_dir / "test_3d_rg_znssd_incremental_affine*.csv",
        gt=u,
        stereo=True,
        atol=0.01,
        atol_stereo=0.0001,
    )


def test_3d_rg_znssd_quad(dic_temp_dir: Path) -> None:
    dic.calculate_3d(
        reference=[ref0, ref1],
        deformed=[def0, def1],
        roi_mask=roi.mask,
        calibration=calib_data,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=60,
        correlation_criteria="ZNSSD",
        shape_function="QUAD",
        method="MULTIWINDOW_RG",
        output_basepath=dic_temp_dir,
        output_prefix="test_3d_rg_znssd_quad_",
    )

    validate(
        dic_temp_dir / "test_3d_rg_znssd_quad_*.csv",
        gt=u,
        stereo=True,
        atol=0.01,
        atol_stereo=0.0005,
    )


def test_f32_support(dic_temp_dir: Path) -> None:
    np.random.seed(100)
    ref_arr = np.random.uniform(0, 200, size=(400, 400)).astype(np.float32)
    def_arr = np.roll(ref_arr, 1, axis=1)
    def_arr = np.roll(def_arr, -1, axis=0)

    roi = dic.RegionOfInterest(ref_arr)
    roi.rect_boundary(10, 10, 10, 10)

    dic.calculate_2d(
        reference=ref_arr,
        deformed=def_arr,
        roi_mask=roi.mask,
        seed=[250, 250],
        subset_size=31,
        subset_step=15,
        max_displacement=10,
        output_basepath=dic_temp_dir,
        output_prefix="test_f32_support_",
    )

    validate(
        dic_temp_dir / "test_f32_support*.csv",
        gt=[1.0],
        stereo=False,
        atol=0.001,
        atol_stereo=0.00001,
    )

