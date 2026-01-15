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
    values ("RIGID", "AFFINE" or "QUAD"). If valid, it returns the number of transformation
    parameters associated with that shape function.

    Parameters
    ----------
    shape_function : str
        The shape function type. Must be either "RIGID", "AFFINE" or "QUAD".

    Returns
    -------
    int
        The number of parameters for the specified shape function:
        - 2 for "RIGID"
        - 6 for "AFFINE"
        - 12 for "QUAD"

    Raises
    ------
    ValueError
        If `shape_function` is not one of the allowed values.
    """

    if (shape_function=="RIGID"):
        num_params = 2
    elif (shape_function=="AFFINE"): 
        num_params = 6
    elif (shape_function=="QUAD"): 
        num_params = 12
    else:
        raise ValueError(f"Invalid shape_function: {shape_function}. "
                         f"Allowed values are: 'AFFINE', 'RIGID', 'QUAD'.")

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



def check_method(method: str) -> None:
    """
    Validate that the scan type  one of the allowed methods.
    Allowed values are "MULTIWINDOW_RG", "MULTIWINDOW", "SINGLEWINDOW_RG", "SINGLEWINDOW_RG_INCREMENTAL", "IMAGE_SCAN".

    Parameters
    ----------
    interpolation_routine : str
        The interpolation method to validate. Must be either "BILINEAR" or "BICUBIC".

    Raises
    ------
    ValueError
        If `interpolation_routine` is not a supported value.

    """

    allowed_values = {"MULTIWINDOW_RG", "MULTIWINDOW", "SINGLEWINDOW_RG", "SINGLEWINDOW_RG_INCREMENTAL", "IMAGE_SCAN"}

    if method not in allowed_values:
        raise ValueError(f"Invalid method: {method}. "
                         f"Allowed values are: {', '.join(allowed_values)}")



def check_thresholds(threshold: float, 
                     bf_threshold: float, 
                     precision: float) -> None:
    """
    Ensures that `threshold`, `bf_threshold`, and `precision`
    are all floats strictly between 0 and 1. Raises a `ValueError` if any condition fails.

    Parameters
    ----------
    threshold : float
        correlation/cost coeff minumum value to be considered matching subset.
    bf_threshold : float
        Threshold for the brute-force optimization method.
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

    if not (0 < bf_threshold < 1):
        raise ValueError("bf_threshold must be a float "
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



def check_and_update_rg_seed(seed: list[int] | list[np.int32] | np.ndarray, roi_mask: np.ndarray, method: str, px_hori: int, px_vert: int, subset_size: int, subset_step: int) -> list[int]:
    """
    Validate and update the region-growing seed location to align with image bounds and subset spacing.

    This function checks the format and bounds of the seed coordinates used for a region-growing (RG)
    scanning method. It adjusts the seed to the nearest valid grid point based on the subset step size,
    clamps it to the image dimensions, and ensures it lies within the region of interest (ROI) mask.

    If the scanning method is not "RG", the function returns a default seed of [0, 0]. 
    This seed is not used any other scan method methods.

    Parameters
    ----------
    seed : list[int], list[np.int32] or np.ndarray
        The initial seed coordinates as a list of two integers: [x, y].
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
        The adjusted seed coordinates [x, y] aligned to the subset grid and within bounds.

    Raises
    ------
    ValueError
        If the seed is improperly formatted, out of image bounds, or not a list of two integers.
    """

    if "RG" not in method:
        return [0,0]

    if (len(seed) != 2):
        raise ValueError(f"Reliability Guided seed does not have two elements: " \
                         f"seed={seed}. Seed " \
                         f" must be a list of two integers: seed=[x, y]")

    if not isinstance(seed, (list, np.ndarray)) or not all(isinstance(coord, (int, np.int32)) for coord in seed):
        raise ValueError("Reliability Guided seed must be a list of two integers: seed=[x, y]")

    x, y = seed

    if x < 0 or x >= px_hori or y < 0 or y >= px_vert:
        raise ValueError(f"Seed ({x}, {y}) goes outside the image bounds: ({px_hori}, {px_vert})")

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

    # Check if all pixel values in the ROI are valid
    for i in range(new_x, max_x):
        for j in range(new_y, max_y):

            if i < 0 or i >= px_hori or j < 0 or j >= px_vert:
                raise ValueError(f"Seed ({x}, {y}) goes outside the image bounds at pixel ({i}, {j})")

            if not roi_mask[j, i]:
                raise ValueError(f"Seed ({x}, {y}) goes outside the ROI at pixel ({i}, {j})")

    return [new_x, new_y]


def check_and_get_images(reference: np.ndarray | str | Path,
                         deformed: np.ndarray | str | Path | list[Path],
                         roi: np.ndarray, debug_level: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    deformed : np.ndarray, str, pathlib.Path, list[pathlib.Path]
        Either a NumPy array representing a sequence of deformed images (shape: [N, H, W]),
        or a glob pattern string pointing to multiple image files.
    roi : np.ndarray
        A 2D NumPy array defining the region of interest. Must match the reference image shape
        if `reference` is an array.
    debug_level: int
        Determines how much information to provide in console output.

    Returns
    -------
    image_stack: np.ndarray
        A 3D NumPy array containing all deformed images with shape (N, H, W).
    filenames : list of str
        List of base filenames of all images (empty if images are passed as arrays).

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
        assert isinstance(reference, Path)

        if not reference.is_file():
            raise ValueError(f"Reference image does not exist: {reference}")


        if (debug_level>0):
            print("Using reference image: ")
            print(f"  - {reference}\n")

        # Load reference image
        ref_arr = np.array(Image.open(reference))

        if ref_arr.ndim == 3:
            if (debug_level>0):
                print(f"Reference image appears to have {ref_arr.shape[2]} channels. Using channel 0.")
            ref_arr = ref_arr[:, :, 0]

        if (debug_level>0):
            print(f"Reference image shape: {ref_arr.shape}")
            print("")

        filenames.append(os.path.basename(reference))

        if isinstance(deformed, Path):
            files = sorted(glob.glob(str(deformed)))
        else:
            files = [str(p) for p in deformed]

        if not files:
            raise FileNotFoundError(f"No deformation images found: {deformed}")



        if debug_level > 0:
            print(f"Found {len(files)} deformation images:")
            for file in files:
                print(f"  - {file}")
            print("")

        # populate filenames list. Stars with ref image.
        filenames.extend(os.path.basename(f) for f in files)

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

        # user might only pass a single deformed image. need to convert to 'stack'
        if (reference.shape == deformed.shape):
            def_arr = def_arr.reshape((1,def_arr.shape[0],def_arr.shape[1]))

        elif (reference.shape != deformed[0].shape or reference.shape != roi.shape):
            raise ValueError(f"Shape mismatch: reference={reference.shape}, "
                             f"deformed[0]={deformed[0].shape}, roi={roi.shape}")

        # check ROI dimensions agrees with reference image
        if (reference.shape != roi.shape):
            raise ValueError(f"Shape mismatch: reference={reference.shape}, "
                             f"roi={roi.shape}")
 
        # need to set some dummy filenames in the case that the user passes numpy arrays
        filenames = ["ref_img"]
        for f in range(0,def_arr.shape[0]):
            filenames.append(f"def_img_{f:04d}")

    # it might be the case that the roi has been manipulated prior to DIC run
    # and therefore we need to to prevent the roi mask from being a 'view'
    roi_c = np.ascontiguousarray(roi)

    # Build image stack: reference first, then deformed images
    image_stack = np.concatenate(([ref_arr], def_arr), axis=0)

    return image_stack, roi_c, filenames




def print_title(a: str):
    line_width = 80
    half_width = 39

    print('-' * line_width)

    # Center the title between dashes
    left_dashes = '-' * (half_width - len(a) // 2)
    right_dashes = '-' * (half_width - len(a) // 2)
    print(f"{left_dashes} {a} {right_dashes}")

    print('-' * line_width)
