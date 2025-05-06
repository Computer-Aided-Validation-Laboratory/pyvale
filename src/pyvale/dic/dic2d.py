"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""


import numpy as np
import os
import sys


# import cython module
from pyvale.dic import dic2dcpp
from pyvale.dic.dicresults import DICResults


def DIC2D(reference: np.ndarray,
          deformed: np.ndarray,
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
          scanning_method: str="IMAGE_SCAN") -> DICResults:


        """
        Executes the c++ 2D DIC routine on CPU architecture.
        """
        
        # do var checks in python land
        check_correlation_criteria(correlation_criteria)
        check_interpolation(interpolation_routine)
        check_scanning_method(scanning_method)
        check_thresholds(threshold_levenberg, threshold_levenberg, precision)
        check_subset_size(subset_size)
        check_image_sizes(reference,deformed,roi_mask)
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
        config.px_horizontal = reference.shape[1]
        config.px_vertical = reference.shape[0]
        config.num_def_images = deformed.shape[0]
        config.num_params = num_params


        results = dic2dcpp.engine(reference,
          deformed,
          roi_mask,
          config)


    



def DIC2Dsave(results: DICResults,
              base_path: str="./",
              format: str=".dat",
              layout: str="col",
              prefix: str="image",
              delimiter: str=" ") -> None:


    # check if there's output files
    try:
        files = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Output path '{base_path}' does not exist.")
        sys.exit(1)

    # Check for any matching files
    conflicting_files = [
        f for f in files 
        if f.startswith(prefix) and f.endswith(format)
    ]

    if conflicting_files:
        print("The following files already exist:")
        for f in conflicting_files:
            print(f"  - {os.path.join(base_path, f)}")

        user_input = input("Do you want to continue and overwrite these files? (y/n) -> None: ").strip().lower()

        if user_input not in ("y", "yes"):
            print("Aborting to avoid overwriting data.")
            return



def DIC2Dgpu() -> None:
    """
    Executes the c++ 2D DIC routine on GPU architecture.
    """

    print("This is a work in progress...")

    return None


def build_info() -> None:
    """
    Returns the build information of the diccppinterface module.
    """
    build = dic2dcpp.build_info()
    return build




def check_correlation_criteria(correlation_criteria: str) -> None:
    allowed_values = {"SSD", "NSSD", "ZNSSD"}

    if correlation_criteria not in allowed_values:
        raise ValueError(f"Invalid correlation_criteria: {correlation_criteria}. Allowed values are: {', '.join(allowed_values)}")



def check_shape_function(shape_function: str) -> int:

    if (shape_function=="RIGID"):
        num_params = 2
    elif (shape_function=="AFFINE"): 
        num_params = 6
    else:
        raise ValueError(f"Invalid shape_function: {shape_function}. Allowed values are: 'AFFINE', 'RIGID'.")

    return num_params



def check_interpolation(interpolation_routine: str) -> None:
    allowed_values = {"BILINEAR", "BICUBIC"}

    if interpolation_routine not in allowed_values:
        raise ValueError(f"Invalid interpolation_routine: {interpolation_routine}. Allowed values are: {', '.join(allowed_values)}")



def check_scanning_method(scanning_method: str) -> None:
    allowed_values = {"IMAGE_SCAN", "IMAGE_SCAN_WITH_BF", "RG"}

    if scanning_method not in allowed_values:
        raise ValueError(f"Invalid scannign_method: {scanning_method}. Allowed values are: {', '.join(allowed_values)}")



def check_thresholds(threshold_levenberg: float, threshold_bruteforce: float, precision: float) -> None:
    
    if not (0 < threshold_levenberg < 1):
        raise ValueError("threshold_levenberg must be a float strictly between 0 and 1.")

    if not (0 < threshold_bruteforce < 1):
        raise ValueError("threshold_bruteforce must be a float strictly between 0 and 1.")
    
    if not (0 < precision < 1):
        raise ValueError("precision must be a float strictly between 0 and 1.")

def check_subset_size(subset_size: int):
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be an odd number.")

def check_image_sizes(reference: np.ndarray,
                      deformed: np.ndarray, 
                      roi: np.ndarray) -> None:

    if reference.shape != deformed[0].shape or reference.shape != roi.shape:
        raise ValueError(f"Difference in image dimensions: "
                         f"reference {reference.shape}, "
                         f"deformed[0] {deformed[0].shape}, "
                         f"roi {roi.shape}")





