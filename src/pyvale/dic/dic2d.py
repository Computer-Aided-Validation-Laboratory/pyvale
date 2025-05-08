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
from pyvale.dic import dic2dcpp
from pyvale.dic.dicresults import DICResults


def DIC2D(reference: np.ndarray | str,
          deformed: np.ndarray | str,
          roi_mask: np.ndarray,
          subset_step: int=10, 
          subset_size: int=21,
          correlation_criteria: str="ZNSSD",
          shape_function: str="AFFINE",
          interpolation_routine: str="BICUBIC",
          max_iterations: int=40,
          precision: float=0.001,
          threshold_levenberg: float=0.1,
          threshold_bruteforce: float=0.2,
          range_bruteforce: int=10,
          scanning_method: str="IMAGE_SCAN",
          output_at_end: bool=False,
          output_basepath: str="./",
          output_format: str=".dat",
          output_layout: str="col",
          output_prefix: str="results",
          output_delimiter: str=" ") -> DICResults:

    # do var checks in python land
    ref_arr, def_arr = check_and_get_images(reference,deformed,roi_mask)
    check_correlation_criteria(correlation_criteria)
    check_interpolation(interpolation_routine)
    check_scanning_method(scanning_method)
    check_thresholds(threshold_levenberg, threshold_levenberg, precision)
    check_subset_size(subset_size)
    check_output_directory(output_basepath, output_format, output_layout,
                           output_prefix, output_delimiter)
    num_params = check_shape_function(shape_function)

    # Assign values to config struct
    config = dic2dcpp.Config()
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = precision
    config.threshold_lm = threshold_levenberg
    config.threshold_bf = threshold_bruteforce
    config.range_bf = range_bruteforce
    config.corr_crit = correlation_criteria
    config.shape_func = shape_function
    config.interp_routine = interpolation_routine
    config.scan_method = scanning_method
    config.px_horizontal = ref_arr.shape[1]
    config.px_vertical = ref_arr.shape[0]
    config.num_def_img = def_arr.shape[0]
    config.num_params = num_params

    saveconf = dic2dcpp.SaveConfig()
    saveconf.basepath = output_basepath
    saveconf.format = output_format
    saveconf.layout = output_layout
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end


    print(ref_arr.dtype, def_arr.dtype, roi_mask.dtype)
    print(ref_arr.shape, def_arr.shape, roi_mask.shape)
    results = dic2dcpp.engine(ref_arr, def_arr, roi_mask, config, saveconf)




def check_output_directory(output_basepath: str,
                           output_format: str,
                           output_layout: str,
                           output_prefix: str,
                           output_delimiter: str) -> None:


    # check if there's output files
    try:
        files = os.listdir(output_basepath)
    except FileNotFoundError:
        print(f"Output path '{output_basepath}' does not exist.")
        sys.exit(1)

    # Check for any matching files
    conflicting_files = [
        f for f in files 
        if f.startswith(output_prefix) and f.endswith(output_format)
    ]

    if conflicting_files:
        conflicting_files.sort()
        print("The following files already exist:")
        for f in conflicting_files:
            print(f"  - {os.path.join(output_basepath, f)}")

        user_input = input("Do you want to continue? (y/n): ").strip().lower()

        if user_input not in ("y", "yes", "Y", "YES"):
            print("Aborting to avoid overwriting data in output directory.")
            exit(0)



def DIC2Dgpu() -> None:
    """
    Executes the c++ 2D DIC routine on GPU architecture.
    """

    print("This is a work in progress...")

    return None


def build_test() -> None:
    """
    Returns the build information of the diccppinterface module.
    """
    dic2dcpp.build_info()




def check_correlation_criteria(correlation_criteria: str) -> None:
    allowed_values = {"SSD", "NSSD", "ZNSSD"}

    if correlation_criteria not in allowed_values:
        raise ValueError(f"Invalid correlation_criteria: "
                         f"{correlation_criteria}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_shape_function(shape_function: str) -> int:

    if (shape_function=="RIGID"):
        num_params = 2
    elif (shape_function=="AFFINE"): 
        num_params = 6
    else:
        raise ValueError(f"Invalid shape_function: {shape_function}. "
                         f"Allowed values are: 'AFFINE', 'RIGID'.")

    return num_params



def check_interpolation(interpolation_routine: str) -> None:
    allowed_values = {"BILINEAR", "BICUBIC"}

    if interpolation_routine not in allowed_values:
        raise ValueError(f"Invalid interpolation_routine: "
                         f"{interpolation_routine}. Allowed values are: "
                         f"{', '.join(allowed_values)}")



def check_scanning_method(scanning_method: str) -> None:
    allowed_values = {"IMAGE_SCAN", "IMAGE_SCAN_WITH_BF", "RG"}

    if scanning_method not in allowed_values:
        raise ValueError(f"Invalid scannign_method: {scanning_method}. "
                         f"Allowed values are: {', '.join(allowed_values)}")



def check_thresholds(threshold_levenberg: float, 
                     threshold_bruteforce: float, 
                     precision: float) -> None:
    
    if not (0 < threshold_levenberg < 1):
        raise ValueError("threshold_levenberg must be a float "
                         "strictly between 0 and 1.")

    if not (0 < threshold_bruteforce < 1):
        raise ValueError("threshold_bruteforce must be a float "
                         "strictly between 0 and 1.")
    
    if not (0 < precision < 1):
        raise ValueError("precision must be a float strictly "
                         "between 0 and 1.")





def check_subset_size(subset_size: int):
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be an odd number.")




def check_and_get_images(reference: Union[np.ndarray, str],
                 deformed: Union[np.ndarray, str],
                 roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Check type consistency
    if type(reference) is not type(deformed):
        raise ValueError(
            f"Mismatch in input types: reference={type(reference)}, "
            f"deformed={type(deformed)}"
        )

    if isinstance(reference, str):
        if not os.path.isfile(reference):
            raise ValueError(
                f"Reference image does not exist: {reference}"
            )

        ref_arr = np.flip(np.array(Image.open(reference)),axis=0)

        files = sorted(glob.glob(deformed))
        if not files:
            raise FileNotFoundError(
                f"No deformation images found: {deformed}"
            )
        

        print(f"Found {len(files)} deformation images:")
        for file in files:
            print("-", file)

        def_arr = np.zeros((len(files), *ref_arr.shape), dtype=ref_arr.dtype)

        for i, file in enumerate(files):
            img = np.flip(np.array(Image.open(file)), axis=0)
            if img.shape != ref_arr.shape:
                raise ValueError(
                    f"Shape mismatch: '{file}' has shape {img.shape}, "
                    f"expected {ref_arr.shape}"
                )
            def_arr[i] = img

    else:
        if (reference.shape != deformed[0].shape or 
                reference.shape != roi.shape):
            raise ValueError(
                f"Shape mismatch: reference {reference.shape}, "
                f"deformed[0] {deformed[0].shape}, roi {roi.shape}"
            )

    return ref_arr, def_arr


