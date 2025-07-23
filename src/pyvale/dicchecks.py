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

"""
This module contains functions for checking arguments passed to the 2D DIC
Engine.
"""

def check_output_directory(output_basepath: str,
                           output_prefix: str) -> None:
    """
    Check for existing output files in a directory and prompt user confirmation before overwriting.

    This function verifies whether the specified output directory exists and checks for any existing
    files that match a given prefix and have `.csv` or `.dic2d` extensions. If such files are found,
    a list is displayed and the user is prompted to confirm whether to continue. If the user declines,
    the program exits to prevent data loss.

    Parameters
    ----------
    output_basepath : str
        Path to the output directory where files are or will be saved.
    output_prefix : str
        Filename prefix used to identify potential conflicting output files.

    Raises
    ------
    SystemExit
        If the output directory does not exist or the user chooses not to proceed after
        being warned about existing files.
    """

    # check if there's output files
    try:
        files = os.listdir(output_basepath)
    except FileNotFoundError:
        print("")
        print(f"Output directory '{output_basepath}' does not exist.")
        sys.exit(1)

    # Check for any matching files
    conflicting_files = [
        f for f in files 
        if f.startswith(output_prefix) and (f.endswith(".csv") or f.endswith(".dic2d"))]

    if conflicting_files:
        conflicting_files.sort()
        print("The following output files already exist and may be overwritten:")
        for f in conflicting_files:
            print(f"  - {os.path.join(output_basepath, f)}")
        print("")

        user_input = input("Do you want to continue? (y/n): ").strip().lower()

        if user_input not in ("y", "yes", "Y", "YES"):
            print("Aborting to avoid overwriting data in output directory.")
            exit(0)


def check_correlation_criteria(correlation_criteria: str) -> None:
    """
    Validate that the correlation criteria is one of the allowed values.

    Checks whether input `correlation_criteria` is among the
    accepted options: "SSD", "NSSD", or "ZNSSD". If not, raises a `ValueError`.

    Parameters
    ----------
    correlation_criteria : str
        The correlation type. Must be one of: "SSD", "NSSD", or "ZNSSD".

    Raises
    ------
    ValueError
        If `correlation_criteria` is not one of the allowed values.
    """

    allowed_values = {"SSD", "NSSD", "ZNSSD"}

    if correlation_criteria not in allowed_values:
        raise ValueError(f"Invalid correlation_criteria: "
                         f"{correlation_criteria}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_shape_function(shape_function: str) -> int:
    """
    Checks whether input `shape_function` is one of the allowed
    values ("RIGID" or "AFFINE"). If valid, it returns the number of transformation
    parameters associated with that shape function.

    Parameters
    ----------
    shape_function : str
        The shape function type. Must be either "RIGID" or "AFFINE".

    Returns
    -------
    int
        The number of parameters for the specified shape function:
        - 2 for "RIGID"
        - 6 for "AFFINE"

    Raises
    ------
    ValueError
        If `shape_function` is not one of the allowed values.
    """

    if (shape_function=="RIGID"):
        num_params = 2
    elif (shape_function=="AFFINE"): 
        num_params = 6
    else:
        raise ValueError(f"Invalid shape_function: {shape_function}. "
                         f"Allowed values are: 'AFFINE', 'RIGID'.")

    return num_params



def check_interpolation(interpolation_routine: str) -> None:
    """
    Validate that the interpolation routine is one of the allowed methods.

    Checks whether interpolation_routine is a supported
    interpolation method. Allowed values are "BILINEAR" and "BICUBIC". If the input
    is not one of these, a `ValueError` is raised.

    Parameters
    ----------
    interpolation_routine : str
        The interpolation method to validate. Must be either "BILINEAR" or "BICUBIC".

    Raises
    ------
    ValueError
        If `interpolation_routine` is not a supported value.

    """

    allowed_values = {"BILINEAR", "BICUBIC"}

    if interpolation_routine not in allowed_values:
        raise ValueError(f"Invalid interpolation_routine: "
                         f"{interpolation_routine}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_scanning_method(scanning_method: str) -> None:
    """
    Validate that the scan type  one of the allowed methods.

    Allowed values are "RG", "IMAGE_SCAN", "FFT", "IMAGE_SCAN_WITH_BF", "FFT_test". If `scanning_method`
    is not one of these, a `ValueError` is raised.

    Parameters
    ----------
    interpolation_routine : str
        The interpolation method to validate. Must be either "BILINEAR" or "BICUBIC".

    Raises
    ------
    ValueError
        If `interpolation_routine` is not a supported value.

    """

    allowed_values = {"RG", "IMAGE_SCAN", "FFT", "IMAGE_SCAN_WITH_BF", "FFT_test"}

    if scanning_method not in allowed_values:
        raise ValueError(f"Invalid scanning_method: {scanning_method}. "
                         f"Allowed values are: {', '.join(allowed_values)}")



def check_thresholds(opt_threshold: float, 
                     bf_threshold: float, 
                     opt_precision: float) -> None:
    """
    Ensures that `opt_threshold`, `bf_threshold`, and `opt_precision`
    are all floats strictly between 0 and 1. Raises a `ValueError` if any condition fails.

    Parameters
    ----------
    opt_threshold : float
        Threshold for the Levenberg optimization method.
    bf_threshold : float
        Threshold for the brute-force optimization method.
    opt_precision : float
        Desired precision for the optimizer.

    Raises
    ------
    ValueError
        If any input value is not a float strictly between 0 and 1.
    """

    if not (0 < opt_threshold < 1):
        raise ValueError("opt_threshold must be a float "
                         "strictly between 0 and 1.")

    if not (0 < bf_threshold < 1):
        raise ValueError("bf_threshold must be a float "
                         "strictly between 0 and 1.")
    
    if not (0 < opt_precision < 1):
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



def check_and_update_rg_seed(seed: list[int], roi_mask: np.ndarray, scanning_method: str, px_hori: int, px_vert: int, subset_size: int, subset_step: int) -> list[int]:
    """
    Validate and update the region-growing seed location to align with image bounds and subset spacing.

    This function checks the format and bounds of the seed coordinates used for a region-growing (RG)
    scanning method. It adjusts the seed to the nearest valid grid point based on the subset step size,
    clamps it to the image dimensions, and ensures it lies within the region of interest (ROI) mask.

    If the scanning method is not "RG", the function returns a default seed of [0, 0]. 
    This seed is not used any other scan method methods.

    Parameters
    ----------
    seed : list of int
        The initial seed coordinates as a list of two integers: [x, y].
    roi_mask : np.ndarray
        A 2D binary mask (same size as the image) indicating the region of interest.
    scanning_method : str
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
        The adjusted seed coordinates [x, y] aligned to the subset grid and within bounds.

    Raises
    ------
    ValueError
        If the seed is improperly formatted, out of image bounds, or not a list of two integers.
    """

    if scanning_method != "RG":
        return [0,0]

    if not (isinstance(seed, list) and len(seed) == 2 and all(isinstance(coord, int) for coord in seed)):
        raise ValueError("Reliability Guided seed is either missing or has been defined incorrectly. must be a list of two integers: seed=[x, y]")

    x, y = seed

    corner_x = x - subset_size//2
    corner_y = y - subset_size//2

    def round_to_step(value: int, step: int) -> int:
        return round(value / step) * step

    # snap to grid
    new_x = round_to_step(corner_x, subset_step)
    new_y = round_to_step(corner_y, subset_step)


    # Clamp to image bounds
    new_x = min(max(new_x, 0), px_hori - 1)
    new_y = min(max(new_y, 0), px_vert - 1)

    # check if all pixel values within the seed location are within the ROI
    # seed coordinates are the central pixel to the subset
    max_x = new_x + subset_size//2+1
    max_y = new_y + subset_size//2+1


    # Check if all pixel values in the ROI are valid
    for i in range(corner_x, max_x):
        for j in range(corner_y, max_y):

            if i < 0 or i >= px_hori or j < 0 or j >= px_vert:
                raise ValueError(f"Seed ({x}, {y}) goes outside the image bounds at pixel ({i}, {j})")

            if not roi_mask[j, i]:
                raise ValueError(f"Seed ({x}, {y}) goes outside the ROI at pixel ({i}, {j})")

    return [new_x, new_y]


def check_and_get_images(reference: np.ndarray | str | Path,
                         deformed: np.ndarray | str | Path,
                         roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load and validate reference and deformed images, checks consistency in shape/format.

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
    deformed : np.ndarray, str, pathlib.Path
        Either a NumPy array representing a sequence of deformed images (shape: [N, H, W]),
        or a glob pattern string pointing to multiple image files.
    roi : np.ndarray
        A 2D NumPy array defining the region of interest. Must match the reference image shape
        if `reference` is an array.

    Returns
    -------
    ref_arr : np.ndarray
        The reference image as a 2D NumPy array.
    def_arr : np.ndarray
        A 3D NumPy array containing all deformed images with shape (N, H, W).
    filenames : list of str
        List of base filenames of deformed images (empty if deformed images were passed as arrays).

    Raises
    ------
    ValueError
        If there is a type mismatch between `reference` and `deformed`,
        if image files are not found or unreadable,
        or if image shapes do not match.
    FileNotFoundError
        If no files are found matching the deformed image pattern.
    """

    filenames = []


    # Normalize Path or str to Path
    if isinstance(reference, (str, Path)):
        reference = Path(reference)
    if isinstance(deformed, (str, Path)):
        deformed = Path(deformed)

    # check matching filetypes 
    if type(reference) is not type(deformed):
        raise ValueError(
            f"Mismatch in file types: reference={type(reference)}, "
            f"deformed={type(deformed)}")


    # File-based input
    if isinstance(reference, Path):
        assert isinstance(deformed, Path)

        if not reference.is_file():
            raise ValueError(f"Reference image does not exist: {reference}")
        print("Using reference image: ")
        print(f"  - {reference}\n")

        # Load reference image
        ref_arr = np.array(Image.open(reference))
        print(f"Reference image shape: {ref_arr.shape}")
        if ref_arr.ndim == 3:
            print(f"Reference image appears to have {ref_arr.shape[2]} channels. Using channel 0.")
            ref_arr = ref_arr[:, :, 0]
        print("")

        # Find deformation image files
        files = sorted(glob.glob(str(deformed)))
        if not files:
            raise FileNotFoundError(f"No deformation images found: {deformed}")

        print(f"Found {len(files)} deformation images:")
        for file in files:
            print(f"  - {file}")
            filenames.append(os.path.basename(file))
        print("")

        def_arr = np.zeros((len(files), *ref_arr.shape), dtype=ref_arr.dtype)

        for i, file in enumerate(files):
            img = np.array(Image.open(file))
            if img.ndim == 3:
                print(f"Deformed image {file} appears to have {img.shape[2]} channels. Using channel 0.")
                img = img[:, :, 0]
            if img.shape != ref_arr.shape:
                raise ValueError(f"Shape mismatch: '{file}' has shape {img.shape}, expected {ref_arr.shape}")
            def_arr[i] = img

    # Array-based input
    else:
        assert isinstance(reference, np.ndarray)
        assert isinstance(deformed, np.ndarray)
        ref_arr = reference
        def_arr = deformed

        if (reference.shape != deformed[0].shape or reference.shape != roi.shape):
            raise ValueError(f"Shape mismatch: reference {reference.shape}, "
                             f"deformed[0] {deformed[0].shape}, roi {roi.shape}")
    
    # it might be the case that the roi has been manipulated prior to DIC run
    # and therefore we need to to prevent the roi mask from being a 'view'
    roi_c = np.ascontiguousarray(roi)

    return ref_arr, def_arr, roi_c, filenames



def check_strain_files(strain_files: str | Path) -> list[str]:
   
    filenames = []

    # Find deformation image files
    files = sorted(glob.glob(str(strain_files)))
    if not files:
        raise FileNotFoundError(f"No DIC data found: {strain_files}")

    for file in files:
        filenames.append(os.path.basename(file))

    return filenames


def print_title(a: str):
    line_width = 80
    half_width = 39

    print('-' * line_width)

    # Center the title between dashes
    left_dashes = '-' * (half_width - len(a) // 2)
    right_dashes = '-' * (half_width - len(a) // 2)
    print(f"{left_dashes} {a} {right_dashes}")

    print('-' * line_width)
