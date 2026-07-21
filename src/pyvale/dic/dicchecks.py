# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import numpy as np
import glob
import os
import sys
from PIL import Image
from pathlib import Path
from enum import Enum

import pyvale.common_py.util as common_py_util

"""
This module contains functions for checking arguments passed to the 2D DIC
Engine.
"""

class ScanMethod(str, Enum):
    MULTIWINDOW_RG = "MULTIWINDOW_RG"
    SINGLEWINDOW_RG = "SINGLEWINDOW_RG"
    MULTIWINDOW = "MULTIWINDOW"
    RASTER = "RASTER"

class Shape(str, Enum):
    RIGID = "RIGID"
    AFFINE = "AFFINE"
    QUAD = "QUAD"

class CorrCrit(str, Enum):
    SSD = "SSD"
    NSSD = "NSSD"
    ZNSSD = "ZNSSD"

class Interp(str, Enum):
    BSPLINE = "BSPLINE"
    HERMITE = "HERMITE"

class IncrementalMethod(str, Enum):
    IMAGE = "IMAGE"
    COST = "COST"
    ITER = "ITER"


def multiwindow_init(subset_size: int, 
                     subset_step: int,
                     max_displacement: int,
                     multiwindow_overlap: float,
                     multiwindow_subset_size: list[int],
                     multiwindow_search_area: list[int]) -> tuple[list[int], list[int], list[int]]:


    # check multiwindow_subset_size and multiwindow_search_area are same length
    if len(multiwindow_subset_size) != len(multiwindow_search_area):
        raise ValueError(f"multiwindow_subset_size and multiwindow_search_area must be the same length. "
                         f"Got lengths {len(multiwindow_subset_size)} and {len(multiwindow_search_area)}")

    # check if multiwindow_subset_size and multiwindow_search_area are descending
    if any(multiwindow_subset_size[i] < multiwindow_subset_size[i+1] for i in range(len(multiwindow_subset_size)-1)):
        raise ValueError(f"multiwindow_subset_size must be in descending order. "
                         f"Got {multiwindow_subset_size}")

    if any(multiwindow_search_area[i] < multiwindow_search_area[i+1] for i in range(len(multiwindow_search_area)-1)):
        raise ValueError(f"multiwindow_search_area must be in descending order. "
                         f"Got {multiwindow_search_area}")

    # check if the overlap is a value between 0 and 100
    if multiwindow_overlap < 0 or multiwindow_overlap > 1:
        raise ValueError(f"multiwindow_overlap must be a fractional value between 0 and 1."
                         f"Got {multiwindow_overlap}")

    # if they are both empty then use max_displacement as the largest subset_size and multiwindow_search_area
    if len(multiwindow_subset_size) == 0 and len(multiwindow_search_area) == 0:

        # get descending powers from max_displacement down
        powers_of_two = [2**i for i in range(int(np.floor(np.log2(max(2*max_displacement, subset_size)))), -1, -1)]

        # if elements of power_of_two are less than subset_size then remove them
        powers_of_two = [p for p in powers_of_two if p >= subset_size]

        # only append max_displacement if it is greater than or equal to subset_size
        if 2*max_displacement >= subset_size:
            multiwindow_subset_size = [2*max_displacement] + powers_of_two
            multiwindow_search_area = [2*max_displacement] + powers_of_two
        else:
            multiwindow_subset_size = powers_of_two
            multiwindow_search_area = powers_of_two

    # check that all multiwindow_subset_sizes are less than or equal to the
    # multiwindow_search_area elements
    for i in range(len(multiwindow_subset_size)):
        if multiwindow_subset_size[i] > multiwindow_search_area[i]:
            raise ValueError(f"multiwindow_subset_size elements must be less than or equal to the corresponding "
                             f"multiwindow_search_area elements. Got {multiwindow_subset_size[i]} and "
                             f"{multiwindow_search_area[i]} at index {i}")

    overlap = [x * (1.0-multiwindow_overlap) for x in multiwindow_subset_size]
    overlap.append(subset_step)
    overlap  = list(map(int,overlap))

    multiwindow_subset_size.append(subset_size)
    multiwindow_search_area.append(subset_size)

    return overlap, multiwindow_subset_size, multiwindow_search_area






def check_correlation_criteria(correlation_criteria: str) -> None:
    """
    Validate that the correlation criteria is one of the allowed values.

    Checks whether input ``correlation_criteria`` is among the
    accepted options: ``"SSD"``, ``"NSSD"``, or ``"ZNSSD"``. If not, raises a
    ``ValueError``.

    Parameters
    ----------
    correlation_criteria : str
        The correlation type. Must be one of: ``"SSD"``, ``"NSSD"``, or ``"ZNSSD"``.

    Raises
    ------
    ValueError
        If ``correlation_criteria`` is not one of the allowed values.
    """

    allowed_values = {"SSD", "NSSD", "ZNSSD"}

    if correlation_criteria not in allowed_values:
        raise ValueError(f"Invalid correlation_criteria: "
                         f"{correlation_criteria}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_shape_function(shape: Shape) -> int:
    """
    Returns the number of parameters associated with that shape function.

    Parameters
    ----------
    shape_function : str
        The shape function type. Must be either ``RIGID``, ``AFFINE`` or ``QUAD``.

    Returns
    -------
    int
        The number of parameters for the specified shape function:
        - 2 for ``RIGID``
        - 6 for ``AFFINE``
        - 12 for ``QUAD``
    """

    if (shape==Shape.RIGID):
        num_params = 2
    elif (shape==Shape.AFFINE): 
        num_params = 6
    elif (shape==Shape.QUAD): 
        num_params = 12
    
    return num_params



def check_interpolation(interpolation_routine: str) -> None:
    """
    Validate that the interpolation routine is one of the allowed methods.

    Checks whether interpolation_routine is a supported
    interpolation method. Allowed values are ``"BSPLINE"`` and ``"HERMITE"``. If the input
    is not one of these, a ``ValueError`` is raised.

    Parameters
    ----------
    interpolation_routine : str
        The interpolation method to validate. Must be either ``"BSPLINE"`` or ``"HERMITE"``.

    Raises
    ------
    ValueError
        If ``interpolation_routine`` is not a supported value.

    """

    allowed_values = {"BSPLINE", "HERMITE"}

    if interpolation_routine not in allowed_values:
        raise ValueError(f"Invalid interpolation_routine: "
                         f"{interpolation_routine}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_method(method: str) -> None:
    """
    Validate that the scan type  one of the allowed methods.

    Parameters
    ----------
    method : str
        Allowed values are ``"MULTIWINDOW_RG"``, ``"MULTIWINDOW"``, ``"SINGLEWINDOW_RG"``, ``"RASTER"``.

    Raises
    ------
    ValueError
        If ``method`` is not a supported value.

    """

    allowed_values = {"MULTIWINDOW_RG", "MULTIWINDOW", "SINGLEWINDOW_RG", "SINGLEWINDOW_RG", "RASTER"}

    if method not in allowed_values:
        raise ValueError(f"Invalid method: {method}. "
                         f"Allowed values are: {', '.join(allowed_values)}")



def check_thresholds(threshold: float, 
                     precision: float) -> None:
    """
    Ensures that ``threshold``, and ``precision``
    are all floats strictly between 0 and 1. Raises a ``ValueError`` if any condition fails.

    Parameters
    ----------
    threshold : float
        correlation/cost coeff minumum value to be considered matching subset.
    precision : float
        Desired precision for the optimizer.

    Raises
    ------
    ValueError
        If any input value is not a float strictly between 0 and 1.
    """

    if not (0 < threshold < 1):
        raise ValueError("threshold must be a float "
                         "strictly between 0 and 1.")

    if not (0 < precision < 1):
        raise ValueError("Optimizer precision must be a float strictly "
                         "between 0 and 1.")

def check_subsets(subset_size: int, subset_step: int) -> None:
    """

    Parameters
    ----------
    subset_size : int
        Threshold for the Levenberg optimization method.
    subset_step : int
        Threshold for the brute-force optimization method.

    Raises
    ------
    ValueError
        If any input value is not a float strictly between 0 and 1.
    """


    # Enforce scalar types for non-FFT methods
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be an odd number.")

    # check if subset_step is larger than the subset_size
    if subset_step > subset_size:
        raise ValueError("subset_step is larger than the subset_size.")



def check_and_update_rg_seed(seed: list[int] | list[np.int32] | list[tuple[int, int]] | np.ndarray,
                             roi_mask: np.ndarray,
                             method: str,
                             px_hori: int,
                             px_vert: int,
                             subset_size: int,
                             subset_step: int) -> list[int]:
    """
    Validate and update the region-growing seed location to align with image bounds and subset spacing.

    This function checks the format and bounds of the seed coordinates used for a region-growing (RG)
    scanning method. It adjusts the seed to the nearest valid grid point based on the subset step size,
    clamps it to the image dimensions, and ensures it lies within the region of interest (ROI) mask.

    If the scanning method is not reliability guided, the function returns a default seed of [0, 0].
    This seed is not used any other scan method methods.

    Parameters
    ----------
    seed : list[int], list[np.int32], list[tuple[int, int]] or np.ndarray
        Initial seed coordinates as either a flat list ``[x0, y0, x1, y1, ...]`` or a list of
        coordinate tuples ``[(x0, y0), (x1, y1), ...]``.
    roi_mask : np.ndarray
        A 2D binary mask (same size as the image) indicating the region of interest.
    method : str
        The scanning method to be used. Only "RG" triggers validation and adjustment logic.
    px_hori : int
        Width of the image in pixels.
    px_vert : int
        Height of the image in pixels.
    subset_step : int
        Step size used for subset spacing; seed is aligned to this grid.

    Returns
    -------
    list of int
        The adjusted seed coordinates flattened as ``[x0, y0, x1, y1, ...]`` for the C++ engine.

    Raises
    ------
    ValueError
        If the seed is improperly formatted or out of image/ROI bounds.
    """

    if "RG" not in method:
        return [0,0]

    valid_int_types = (int, np.integer)

    if isinstance(seed, np.ndarray):
        seed_values = seed.tolist()
    else:
        seed_values = seed

    if not isinstance(seed_values, list) or len(seed_values) == 0:
        raise ValueError(
            "Reliability Guided seed must contain one or more seed points in either "
            "[x0, y0, x1, y1, ...] or [(x0, y0), (x1, y1), ...] format."
        )

    if all(isinstance(seed_point, tuple) for seed_point in seed_values):
        if not all(
            len(seed_point) == 2
            and all(isinstance(coord, valid_int_types) for coord in seed_point)
            for seed_point in seed_values
        ):
            raise ValueError(
                "Reliability Guided seed tuples must each contain two integers: "
                "seed=[(x0, y0), (x1, y1), ...]"
            )
        seed_points = seed_values
    elif all(isinstance(seed_point, list) for seed_point in seed_values):
        if not all(
            len(seed_point) == 2
            and all(isinstance(coord, valid_int_types) for coord in seed_point)
            for seed_point in seed_values
        ):
            raise ValueError(
                "Reliability Guided seed coordinate lists must each contain two integers: "
                "seed=[[x0, y0], [x1, y1], ...]"
            )
        seed_points = [tuple(seed_point) for seed_point in seed_values]
    else:
        if len(seed_values) < 2 or len(seed_values) % 2 != 0:
            raise ValueError(
                "Reliability Guided seed must contain one or more seed points in either "
                "[x0, y0, x1, y1, ...] or [(x0, y0), (x1, y1), ...] format."
            )
        if not all(isinstance(coord, valid_int_types) for coord in seed_values):
            raise ValueError(
                "Reliability Guided seed must contain integer coordinates in either "
                "[x0, y0, x1, y1, ...] or [(x0, y0), (x1, y1), ...] format."
            )
        seed_points = list(zip(seed_values[::2], seed_values[1::2]))

    updated_seeds = []

    for idx, seed_point in enumerate(seed_points):

        x, y = seed_point
        if x < 0 or x >= px_hori or y < 0 or y >= px_vert:
            raise ValueError(f"Seed {idx} ({x}, {y}) goes outside the image bounds: ({px_hori}, {px_vert})")

        corner_x = x - subset_size//2
        corner_y = y - subset_size//2

        def round_to_step(value: int, step: int) -> int:
            return round(value / step) * step

        # snap to grid
        new_x = round_to_step(corner_x, subset_step)
        new_y = round_to_step(corner_y, subset_step)

        # check if all pixel values within the seed location are within the ROI
        # seed coordinates are the central pixel to the subset
        max_x = new_x + subset_size//2+1
        max_y = new_y + subset_size//2+1


        # check whether all values in the roi_mask are 0
        all_zeros = not np.any(roi_mask)
        if (all_zeros):
            raise ValueError("All values in the ROI mask are 0. Please check the "
                            "ROI mask and try again.")

        # Check if all pixel values in the ROI are valid
        for i in range(new_x, max_x):
            for j in range(new_y, max_y):

                if i < 0 or i >= px_hori or j < 0 or j >= px_vert:
                    raise ValueError(f"Seed {idx} ({x}, {y}) goes outside the image bounds at pixel ({i}, {j})")

                if not roi_mask[j, i]:
                    raise ValueError(f"Seed {idx} ({x}, {y}) goes outside the ROI at pixel ({i}, {j})")

        updated_seeds.append(new_x)
        updated_seeds.append(new_y)

    return updated_seeds

def check_images(reference: np.ndarray | str | Path,
                 deformed: np.ndarray | str | Path | list[Path],
                 roi: np.ndarray, debug_level: int) -> tuple[list[str], list[str], int, int, Path | None]:
    """
    Validate reference and deformed images, checks consistency in shape/format.

    This function accepts either:
    - A file path to a reference image and a glob pattern for a sequence of deformed image files, or
    - Numpy arrays for both reference and deformed images.

    It ensures:
    - The reference and deformed images are the same type (both paths or both arrays).
    - The reference image exists and is readable (if passed as a path).
    - All deformed images exist and match the reference image shape.
    - If images are RGB or multi-channel, only the first channel is used.
    - The `roi` (region of interest) has the same shape as the reference image (when arrays are used directly).

    Parameters
    ----------
    reference : np.ndarray, str, pathlib.Path
        Either a NumPy array representing the reference image, or a file path to a reference image.
    deformed : np.ndarray, str, pathlib.Path, list[pathlib.Path]
        Either a NumPy array representing a sequence of deformed images (shape: [N, H, W]),
        or a glob pattern string pointing to multiple image files.
    roi : np.ndarray
        A 2D NumPy array defining the region of interest. Must match the reference image shape
        if ``reference`` is an array.
    debug_level: int
        Determines how much information to provide in console output.

    Returns
    -------
    basename : list of str
        List of base filenames of all images (empty if images are passed as arrays).
    fullpath : list of str
        List of full paths of all images (empty if images are passed as arrays).
    w : int
        Width of the images in pixels.
    h : int
        Height of the images in pixels.
    temp_dir : pathlib.Path or None
        Path to the temporary directory created to store array-based images on disk.
        ``None`` if file-based input was used. Caller is responsible for cleanup
        (e.g. ``shutil.rmtree(temp_dir)``).

    Raises
    ------
    ValueError
        If there is a type mismatch between ``reference`` and ``deformed``,
        if image files are not found or unreadable,
        or if image shapes do not match.
    FileNotFoundError
        If no files are found matching the deformed image pattern.
    """

    basename = []
    fullpath = []
    temp_dir = None

    # Normalize Path or str to Path
    if isinstance(reference, (str, Path)):
        reference = Path(reference)
    if isinstance(deformed, (str, Path)):
        deformed = Path(deformed)

    # check matching filetypes
    if isinstance(reference, np.ndarray):
        # both must be arrays
        if not isinstance(deformed, np.ndarray):
            raise ValueError(f"Mismatch: reference is array but deformed is {type(deformed)}")

    elif isinstance(reference, Path):
        # deformed must be Path (glob pattern) OR list[Path]
        if not (isinstance(deformed, Path) or (isinstance(deformed, list) and all(isinstance(p, Path) for p in deformed))):
            raise ValueError(f"Invalid deformed type for file-based input: {type(deformed)}")

    else:
        raise ValueError(f"Unsupported reference type: {type(reference)}")

    # File-based input
    if isinstance(reference, Path):
        if not reference.is_file():
            raise ValueError(f"Reference image does not exist: {reference}")

        if debug_level > 0:
            common_py_util.info("Ref img: " + str(reference))

        ref_img = Image.open(reference)

        if debug_level > 0:
            common_py_util.info(f"Ref img shape: {ref_img.size}")

        basename.append(os.path.basename(reference))
        fullpath.append(str(reference))

        if isinstance(deformed, Path):
            files = sorted(glob.glob(str(deformed)))
        else:
            files = sorted(deformed, key=lambda p: os.path.basename(p))

        if not files:
            raise FileNotFoundError(f"No deformation images found: {deformed}")

        if debug_level > 1:
            common_py_util.info(f"Found {len(files)} deformation images in dir: {os.path.dirname(files[0])}")

        basename.extend(os.path.basename(f) for f in files)
        fullpath.extend(str(f) for f in files)

        for i, file in enumerate(files):
            def_img = Image.open(file)
            if def_img.size != ref_img.size:
                raise ValueError(f"Shape mismatch: '{file}' has shape {def_img.size}, expected {ref_img.size}")

    # Array-based input
    else:
        assert isinstance(reference, np.ndarray)
        assert isinstance(deformed, np.ndarray)

        ref_arr = reference
        def_arr = deformed

        # Promote a single deformed image to a stack [1, H, W]
        if ref_arr.shape == def_arr.shape:
            def_arr = def_arr.reshape((1, def_arr.shape[0], def_arr.shape[1]))

        # Validate shapes
        if ref_arr.shape != def_arr[0].shape:
            raise ValueError(
                f"Shape mismatch: reference={ref_arr.shape}, deformed[0]={def_arr[0].shape}"
            )

        if ref_arr.shape != roi.shape:
            raise ValueError(
                f"Shape mismatch: reference={ref_arr.shape}, roi={roi.shape}"
            )

        # Drop channel dim if multi-channel
        if ref_arr.ndim == 3:
            if debug_level > 0:
                print(f"Reference array has {ref_arr.shape[2]} channels. Using channel 0.")
            ref_arr = ref_arr[:, :, 0]

        # Create a tmp directory under cwd
        temp_dir = Path.cwd() / "tmp_dic"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if debug_level > 0:
            print(f"Saving array images to temporary directory: {temp_dir}\n")

        # Save reference image
        ref_filename = "ref_img.tiff"
        ref_path = temp_dir / ref_filename
        Image.fromarray(ref_arr).save(ref_path)
        basename.append(ref_filename)
        fullpath.append(str(ref_path))

        # Save deformed images
        for i in range(def_arr.shape[0]):
            frame = def_arr[i]
            if frame.ndim == 3:
                if debug_level > 0:
                    print(f"Deformed array [{i}] has {frame.shape[2]} channels. Using channel 0.")
                frame = frame[:, :, 0]

            def_filename = f"def_img_{i:04d}.tiff"
            def_path = temp_dir / def_filename
            Image.fromarray(frame).save(def_path)
            basename.append(def_filename)
            fullpath.append(str(def_path))

        if debug_level > 1:
            print(f"Saved {def_arr.shape[0]} deformed images to {temp_dir}")
            for name in basename[1:]:
                print(f"  - {name}")
            print("")

        ref_img = Image.open(ref_path)

    w, h = ref_img.size

    return basename, fullpath, w, h, temp_dir



def print_config_summary(image_width: int,
                         image_height: int,
                         num_def_img: int,
                         max_iterations: int,
                         correlation_criteria: str,
                         shape_function: str,
                         interpolation_routine: str,
                         fft_filter: bool,
                         fft_filter_threshold: float,
                         fft_filter_radius: int,
                         fft_filter_corr_power: float,
                         method: str,
                         precision: float,
                         threshold: float,
                         max_displacement: int,
                         subset_size: int,
                         subset_step: int,
                         num_threads: int | None,
                         debug_level: int,
                         updated_seeds: list[int] | None = None,
                         epi_distance: int | None = None) -> None:
    if debug_level <= 0:
        return

    common_py_util.print_title("Config")
    common_py_util.info_out("Width of Images: ", f"{image_width} [px]")
    common_py_util.info_out("Height of Images: ", f"{image_height} [px]")
    common_py_util.info_out("Number of Deformed Images: ", num_def_img)
    common_py_util.info_out("Max number of solver iterations: ", max_iterations)
    common_py_util.info_out("Correlation Criterion: ", correlation_criteria)
    common_py_util.info_out("Shape Function: ", shape_function)
    common_py_util.info_out("Interpolation Routine: ", interpolation_routine)
    common_py_util.info_out("FFT displacement filter enabled: ", fft_filter)
    common_py_util.info_out("FFT displacement filter threshold: ", fft_filter_threshold)
    common_py_util.info_out("FFT displacement filter radius: ", fft_filter_radius)
    common_py_util.info_out("FFT displacement filter correlation power: ", fft_filter_corr_power)
    common_py_util.info_out("Image Scan Method: ", method)
    common_py_util.info_out("Optimization Precision:", precision)
    common_py_util.info_out("Correlation Cutoff Threshold:", threshold)
    common_py_util.info_out("Estimate for Max Displacement:", f"{max_displacement} [px]")
    if epi_distance is not None:
        common_py_util.info_out("Estimate for Epipolar Distance:", f"{epi_distance} [px]")
    common_py_util.info_out("Subset Size:", f"{subset_size} [px]")
    common_py_util.info_out("Subset Step:", f"{subset_step} [px]")
    if num_threads is None:
        import pyvale.common_cpp.common_cpp as common_cpp
        num_threads = common_cpp.get_num_threads()
    common_py_util.info_out("Number of OMP threads:", num_threads)
    common_py_util.info_out("Debug level: ", debug_level)
    if updated_seeds is not None and "RG" in method:
        for i in range(0, len(updated_seeds), 2):
            x, y = updated_seeds[i], updated_seeds[i + 1]
            common_py_util.info_out(f"Reliability Guided Seed {i//2}:", f"({x}, {y})")



