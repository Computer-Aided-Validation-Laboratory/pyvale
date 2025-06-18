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
          rg_seed: list[int]=[],
          subset_size: int = 21,
          subset_step: int = 10,
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
          output_binary: bool=False,
          output_prefix: str="results",
          output_delimiter: str=" ") -> DICResults:

    # do var checks in python land
    ref_arr, def_arr = check_and_get_images(reference,deformed,roi_mask)
    check_correlation_criteria(correlation_criteria)
    check_interpolation(interpolation_routine)
    check_scanning_method(scanning_method)
    check_thresholds(threshold_levenberg, threshold_levenberg, precision)
    check_output_directory(output_basepath, output_prefix, output_delimiter)
    check_subsets(subset_size, subset_step, scanning_method)
    updated_seed = check_and_update_rg_seed(rg_seed, roi_mask, scanning_method, ref_arr.shape[1], ref_arr.shape[0], subset_step)

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
    config.px_hori = ref_arr.shape[1]
    config.px_vert = ref_arr.shape[0]
    config.num_def_img = def_arr.shape[0]
    config.num_params = num_params
    config.rg_seed = updated_seed

    saveconf = dic2dcpp.SaveConfig()
    saveconf.basepath = output_basepath
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end

    dic2dcpp.dic_engine(ref_arr, def_arr, roi_mask, config, saveconf)




def check_output_directory(output_basepath: str,
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
        if f.startswith(output_prefix) and (f.endswith(".dat") or
                                            f.endswith(".bin"))]

    if conflicting_files:
        conflicting_files.sort()
        print("The following files already exist:")
        for f in conflicting_files:
            print(f"  - {os.path.join(output_basepath, f)}")
        print("")

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


def DICbuildinfo() -> None:
    """
    Prints the C++ build information.
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
    allowed_values = {"IMAGE_SCAN", "IMAGE_SCAN_WITH_BF", "RG", "FFT"}

    if scanning_method not in allowed_values:
        raise ValueError(f"Invalid scanning_method: {scanning_method}. "
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

def check_subsets(subset_size, subset_step, scanning_method: str) -> None:

    # Enforce scalar types for non-FFT methods
    if subset_size % 2 == 0:
        raise ValueError("subset_size must be an odd number.")



def check_and_update_rg_seed(seed: list[int], roi_mask: np.ndarray, scanning_method: str, px_hori: int, px_vert: int, subset_step: int) -> list[int]:
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
                 roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Check type consistency
    if type(reference) is not type(deformed):
        raise ValueError(
            f"Mismatch in input types: reference={type(reference)}, "
            f"deformed={type(deformed)}")

    if isinstance(reference, str):
        if not os.path.isfile(reference):
            raise ValueError(f"Reference image does not exist: {reference}")

        ref_arr = np.array(Image.open(reference))

        files = sorted(glob.glob(deformed))
        if not files:
            raise FileNotFoundError(f"No deformation images found: {deformed}")


        print(f"Found {len(files)} deformation images:")
        for file in files:
            print("  -", file)
        print("")

        def_arr = np.zeros((len(files), *ref_arr.shape), dtype=ref_arr.dtype)

        for i, file in enumerate(files):
            img = np.array(Image.open(file))
            if img.shape != ref_arr.shape:
                raise ValueError(
                    f"Shape mismatch: '{file}' has shape {img.shape}, "
                    f"expected {ref_arr.shape}")

            def_arr[i] = img

    else:
        if (reference.shape != deformed[0].shape or 
                reference.shape != roi.shape):
            raise ValueError(
                f"Shape mismatch: reference {reference.shape}, "
                f"deformed[0] {deformed[0].shape}, roi {roi.shape}")

    return ref_arr, def_arr

def DICdata_import(layout: str="column", 
                   data: str="./", 
                   binary: bool=False,
                   delimiter: str=" ") -> DICResults:

    # firstly check whether layout has been spcified as column or matrix data
    allowed_formats = {"column", "matrix"}
    if layout not in allowed_formats:
        raise ValueError(f"Invalid scanning_method: {layout}. "
                         f"Allowed values are: {', '.join(allowed_formats)}")



    # get the files
    files = sorted(glob.glob(data))
    if not files:
        raise FileNotFoundError(f"No results found in: {data}")

    ss_x_arr, ss_y_arr = None, None
    u_list, v_list, m_list, cost_list  = [],[],[],[]
    ftol_list, xtol_list, niter_list = [],[],[]

    for i,file in enumerate(files):

        if binary:

            # row size in bytes
            row_size = (3*4 + 6*8)
            ss_x_tmp, ss_y_tmp = [],[]
            u_tmp, v_tmp, m_tmp, cost_tmp = [],[],[],[]
            ftol_tmp, xtol_tmp, niter_tmp = [],[],[]

            with open(file, "rb") as f:

                while True:

                    row = f.read(row_size)

                    # check the length of the line is what it should be
                    if not row:
                        break
                    if len(row) != row_size:
                        raise ValueError("Incomplete row in binary file.")
                    
                    ss_x  = np.frombuffer(row[0:4], dtype=np.int32)[0]
                    ss_y  = np.frombuffer(row[4:8], dtype=np.int32)[0]
                    u     = np.frombuffer(row[8:16], dtype=np.float64)[0]
                    v     = np.frombuffer(row[16:24], dtype=np.float64)[0]
                    m     = np.frombuffer(row[24:32], dtype=np.float64)[0]
                    cost  = np.frombuffer(row[32:40], dtype=np.float64)[0]
                    ftol  = np.frombuffer(row[40:48], dtype=np.float64)[0]
                    xtol  = np.frombuffer(row[48:56], dtype=np.float64)[0]
                    niter = np.frombuffer(row[56:60], dtype=np.int32)[0]

                    # Combine coords, u, and v into one row
                    ss_x_tmp.append(ss_x)
                    ss_y_tmp.append(ss_y)
                    u_tmp.append(u)
                    v_tmp.append(v)
                    m_tmp.append(m)
                    cost_tmp.append(cost)
                    ftol_tmp.append(ftol)
                    xtol_tmp.append(xtol)
                    niter_tmp.append(niter)

            if i == 0:
                ss_x_arr = np.array(ss_x_tmp, dtype=np.int32)
                ss_y_arr = np.array(ss_y_tmp, dtype=np.int32)
            else:
                assert np.array_equal(ss_x_arr, ss_x_tmp)
                assert np.array_equal(ss_y_arr, ss_y_tmp)

            u_list.append(u_tmp)
            v_list.append(v_tmp)
            m_list.append(m_tmp)
            cost_list.append(cost_tmp)
            ftol_list.append(ftol_tmp)
            xtol_list.append(xtol_tmp)
            niter_list.append(niter_tmp)

        else:
            data = np.loadtxt(file, delimiter=delimiter)
            if data.shape[1] < 8:
                raise ValueError("Text data must have at least 8 columns.")

            ss_x = data[:, 0].astype(np.int32)
            ss_y = data[:, 1].astype(np.int32)
            u    = data[:, 2]
            v    = data[:, 3]
            m    = data[:, 4]
            cost = data[:, 5]
            ftol = data[:, 6]
            xtol = data[:, 7]
            niter = data[:, 8].astype(np.int32)

            if i == 0:
                ss_x_arr = ss_x
                ss_y_arr = ss_y
            else:
                assert np.array_equal(ss_x_arr, ss_x)
                assert np.array_equal(ss_y_arr, ss_y)

            u_list.append(u)
            v_list.append(v)
            m_list.append(m)
            cost_list.append(cost)
            ftol_list.append(ftol)
            xtol_list.append(xtol)
            niter_list.append(niter)

    # Convert lists to arrays
    u_arr     = np.array(u_list)
    v_arr     = np.array(v_list)
    m_arr  = np.array(m_list)
    cost_arr  = np.array(cost_list)
    ftol_arr  = np.array(ftol_list)
    xtol_arr  = np.array(xtol_list)
    niter_arr = np.array(niter_list)

    if layout == "matrix":
        x_unique = np.unique(ss_x_arr)
        y_unique = np.unique(ss_y_arr)
        X, Y = np.meshgrid(x_unique, y_unique)

        # Determine mesh shape
        rows, cols = Y.shape
        n_frames = u_arr.shape[0]

        def map_to_grid(flat_values):
            grid = np.full((n_frames, rows, cols), np.nan, dtype=np.float64)
            for i in range(len(ss_x_arr)):
                x_idx = np.where(x_unique == ss_x_arr[i])[0][0]
                y_idx = np.where(y_unique == ss_y_arr[i])[0][0]
                grid[:, y_idx, x_idx] = flat_values[:, i]
            return grid

        u_arr     = map_to_grid(u_arr)
        v_arr     = map_to_grid(v_arr)
        m_arr     = map_to_grid(m_arr)
        cost_arr  = map_to_grid(cost_arr)
        ftol_arr  = map_to_grid(ftol_arr)
        xtol_arr  = map_to_grid(xtol_arr)
        niter_arr = map_to_grid(niter_arr)

        return DICResults(X, Y, u_arr, v_arr, m_arr, 
                          cost_arr, ftol_arr, xtol_arr, niter_arr)
    else:
        return DICResults(ss_x_arr, ss_y_arr, u_arr, v_arr, 
                          m_arr, cost_arr, ftol_arr, xtol_arr, niter_arr)

