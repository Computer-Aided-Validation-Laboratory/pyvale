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
from typing import Union


# import cython module
import pyvale.dic.dic2dcpp as dic2dcpp
from pyvale.dic.dicresults import DICResults


def DIC2D(reference: np.ndarray | str,
          deformed: np.ndarray | str,
          roi_mask: np.ndarray,
          seed: list[int]=[],
          subset_size: int = 21,
          subset_step: int = 10,
          correlation_criteria: str="ZNSSD",
          shape_function: str="AFFINE",
          interpolation_routine: str="BICUBIC",
          max_iterations: int=40,
          opt_precision: float=0.001,
          opt_threshold: float=0.1,
          bf_threshold: float=0.2,
          max_displacement: int=128,
          scanning_method: str="RG",
          output_at_end: bool=False,
          output_basepath: str="./",
          output_binary: bool=False,
          output_prefix: str="results",
          output_delimiter: str=" ") -> None:

    # do checks on vars in python land
    ref_arr, def_arr, filenames = check_and_get_images(reference,deformed,roi_mask)
    check_correlation_criteria(correlation_criteria)
    check_interpolation(interpolation_routine)
    check_scanning_method(scanning_method)
    check_thresholds(opt_threshold, bf_threshold, opt_precision)
    check_output_directory(output_basepath, output_prefix)
    check_subsets(subset_size, subset_step)
    updated_seed = check_and_update_rg_seed(seed, roi_mask, scanning_method, ref_arr.shape[1], ref_arr.shape[0], subset_step)
    num_params = check_shape_function(shape_function)


    # Assign values to config struct for c++ land
    config = dic2dcpp.Config()
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = opt_precision
    config.opt_threshold = opt_threshold
    config.bf_threshold = bf_threshold
    config.max_disp = max_displacement
    config.corr_crit = correlation_criteria
    config.shape_func = shape_function
    config.interp_routine = interpolation_routine
    config.scan_method = scanning_method
    config.px_hori = ref_arr.shape[1]
    config.px_vert = ref_arr.shape[0]
    config.num_def_img = def_arr.shape[0]
    config.num_params = num_params
    config.rg_seed = updated_seed
    config.filenames = filenames

    # assigning c++ struct vals for save config
    saveconf = dic2dcpp.SaveConfig()
    saveconf.basepath = output_basepath
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end

    # calling the c++ dic engine
    dic2dcpp.dic_engine(ref_arr, def_arr, roi_mask, config, saveconf)




def check_output_directory(output_basepath: str,
                           output_prefix: str) -> None:
    """
    Check for existing output files in a directory and prompt user confirmation before overwriting.

    This function verifies whether the specified output directory exists and checks for any existing
    files that match a given prefix and have `.dat` or `.bin` extensions. If such files are found,
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
        if f.startswith(output_prefix) and (f.endswith(".dat") or f.endswith(".bin"))]

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
    Validate the shape function type and return the corresponding number of parameters.

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



def check_and_update_rg_seed(seed: list[int], roi_mask: np.ndarray, scanning_method: str, px_hori: int, px_vert: int, subset_step: int) -> list[int]:
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
        raise ValueError("rg_seed is either missing or has been defined incorrectly. must be a list of two integers: rg_seed=[x, y]")

    x, y = seed

    if not (0 <= x < px_hori and 0 <= y < px_vert):
        raise ValueError(f"Seed ({x}, {y}) is out of image bounds ({px_hori}, {px_vert})")

    def round_to_step(value: int, step: int) -> int:
        return round(value / step) * step

    new_x = round_to_step(x, subset_step)
    new_y = round_to_step(y, subset_step)

    # Clamp to image bounds
    new_x = min(max(new_x, 0), px_hori - 1)
    new_y = min(max(new_y, 0), px_vert - 1)

    if (new_x, new_y) != (x, y):
        print(f"Seed adjusted from ({x}, {y}) to ({new_x}, {new_y}) to align with subset step of {subset_step}.")
    
    # check if the new seed location is within the roi
    if not roi_mask[new_x, new_y]:
        print(f"seed location ({new_x}, {new_y}) is not in the Region of interest (ROI) mask. Please select a seed point within the ROI.")

    return [new_x, new_y]


def check_and_get_images(reference: Union[np.ndarray, str],
                         deformed: Union[np.ndarray, str],
                         roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    reference : Union[np.ndarray, str]
        Either a NumPy array representing the reference image, or a file path to a reference image.
    deformed : Union[np.ndarray, str]
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

    # check matching filetypes 
    if type(reference) is not type(deformed):
        raise ValueError(
            f"Mismatch in file types: reference={type(reference)}, "
            f"deformed={type(deformed)}")


    # if the reference is a string rather than a numpy array
    if isinstance(reference, str):
        assert isinstance(deformed, str)

        # check reference image exists 
        if not os.path.isfile(reference):
            raise ValueError(f"Reference image does not exist: {reference}")
        print("Using reference image: ")
        print(f"  - {reference}\n")

        # get shape. check channels
        ref_arr = np.array(Image.open(reference))
        print(f"Reference image shape: {ref_arr.shape}")
        if ref_arr.ndim == 3:
            print(f"Reference image appears to have {ref_arr.shape[2]} channels. Using channel 0.")
            ref_arr = ref_arr[:, :, 0]
        print("")

        # deformed files
        files = sorted(glob.glob(deformed))
        if not files:
            raise FileNotFoundError(f"No deformation images found: {deformed}")

        # can't find any deformed files
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

            # check deformed image shape matches reference
            if img.shape != ref_arr.shape:
                raise ValueError(f"Shape mismatch: '{file}' has shape {img.shape}", f"expected {ref_arr.shape}")

            def_arr[i] = img

    else:
        assert isinstance(reference, np.ndarray)
        assert isinstance(deformed, np.ndarray)
        ref_arr = reference
        def_arr = deformed

        if (reference.shape != deformed[0].shape or reference.shape != roi.shape):
            raise ValueError(f"Shape mismatch: reference {reference.shape}, "
                             f"deformed[0] {deformed[0].shape}, roi {roi.shape}")

    return ref_arr, def_arr, filenames
