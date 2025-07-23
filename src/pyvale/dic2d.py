# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np
from pathlib import Path

# import cython module
import pyvale.dic2dcpp as dic2dcpp
import pyvale.dicchecks as dicchecks


def dic_2d(reference: np.ndarray | str | Path,
          deformed: np.ndarray | str | Path,
          roi_mask: np.ndarray,
          seed: list[int],
          subset_size: int = 21,
          subset_step: int = 10,
          correlation_criteria: str="ZNSSD",
          shape_function: str="AFFINE",
          interpolation_routine: str="BICUBIC",
          max_iterations: int=40,
          opt_precision: float=0.001,
          opt_threshold: float=0.9,
          bf_threshold: float=0.6,
          max_displacement: int=128,
          scanning_method: str="RG",
          fft_mad: bool=False,
          fft_mad_scale: float=3.0,
          output_at_end: bool=False,
          output_basepath: Path | str = "./",
          output_binary: bool=False,
          output_prefix: str="dic_results_",
          output_delimiter: str=",") -> None:
    """
    Perform 2D Digital Image Correlation (DIC) between a reference image and one or more deformed images.

    This function wraps a C++ DIC engine by preparing configuration parameters,
    performing input validation, and dispatching image data and settings. It supports
    pixel-level displacement and strain measurement over a defined region of interest (ROI).

    Parameters
    ----------
    reference : np.ndarray, str or pathlib.Path
        The reference image (2D array) or path to the image file.
    deformed : np.ndarray, str or pathlib.Path
        The deformed image(s) (3D array for multiple images) or path/pattern to image files.
    roi_mask : np.ndarray
        A binary mask indicating the Region of Interest (ROI) for analysis (same size as image).
    seed : list of int, optional
        Coordinates `[x, y]` of the seed point for Reliability-Guided (RG) scanning, default is empty.
    subset_size : int, optional
        Size of the square subset window in pixels (default: 21).
    subset_step : int, optional
        Step size between subset centers in pixels (default: 10).
    correlation_criteria : str, optional
        Metric for matching subsets: "ZNSSD", "NSSD" or "SSD" (default: "ZNSSD").
    shape_function : str, optional
        Deformation model: e.g., "AFFINE", "RIGID" (default: "AFFINE").
    interpolation_routine : str, optional
        Interpolation method used on image intensity. "BICUBIC" is currently the
        only supported option.
    max_iterations : int, optional
        Maximum number of iterations allowed for subset optimization (default: 40).
    opt_precision : float, optional
        Precision threshold for iterative optimization convergence (default: 0.001).
    opt_threshold : float, optional
        Minimum correlation improvement threshold to continue iterations (default: 0.9).
    bf_threshold : float, optional
        Correlation threshold used in rigid bruteforce check for a subset to be considered a
        good match(default: 0.6).
    max_displacement : int, optional
        Estimate for the Maximum displacement in any direction (in pixels) (default: 128).
    scanning_method : str, optional
        Subset scanning method: "RG" for Reliability-Guided (best overall approach), 
        "IMAGE_SCAN" for a standard scan across the image with no seeding 
        (best performance with for subpixel displacements with high quality images), 
        "FFT" for a multi-window FFT based approach (Good for large displacements)
    fft_mad : bool, optional
        The option to smooth FFT windowing data by identifying and replacing outliers using 
        a robust statistical method. For each subset, the function collects values from its 
        neighboring subsets (within a 5x5 window, i.e., radius = 2), computes the median and 
        Median Absolute Deviation (MAD), and determines whether the value at the current 
        subset is an outlier. If it is, the value is replaced with the median of 
        its neighbors. (default: False)
    fft_mad_scale : bool, optional
        An outlier is defined as a value whose deviation from the local median exceeds 
        `fft_mad_scale` times the MAD. This value choses the scaling factor that determines 
        the threshold for detecting outliers relative to the MAD.
    output_at_end : bool, optional
        If True, results will only be written at the end of processing (default: False).
    output_basepath : str or pathlib.Path, optional
        Directory path where output files will be written (default: "./").
    output_binary : bool, optional
        Whether to write output in binary format (default: False).
    output_prefix : str, optional
        Prefix for all output files (default: "dic_results_"). results will be
        named with output_prefix + original filename. THe extension will be
        changed to ".csv" or ".dic2d" depending on whether outputting as a binary.
    output_delimiter : str, optional
        Delimiter used in text output files (default: ",").

    Returns
    -------
    None
        All outputs are written to files; no values are returned.

    Raises
    ------
    ValueError
        If input checks fail (e.g., invalid image sizes, unsupported parameters).
    FileNotFoundError
        If provided file paths do not exist.
    """

    # do checks on vars in python land
    dicchecks.print_title("Initial Checks")
    ref_arr, def_arr, roi_c, filenames = dicchecks.check_and_get_images(reference,deformed,roi_mask)
    dicchecks.check_correlation_criteria(correlation_criteria)
    dicchecks.check_interpolation(interpolation_routine)
    dicchecks.check_scanning_method(scanning_method)
    dicchecks.check_thresholds(opt_threshold, bf_threshold, opt_precision)
    dicchecks.check_output_directory(str(output_basepath), output_prefix)
    dicchecks.check_subsets(subset_size, subset_step)
    updated_seed = dicchecks.check_and_update_rg_seed(seed, roi_mask, scanning_method, ref_arr.shape[1], ref_arr.shape[0], subset_size, subset_step)
    num_params = dicchecks.check_shape_function(shape_function)


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
    config.fft_mad = fft_mad
    config.fft_mad_scale = fft_mad_scale

    # assigning c++ struct vals for save config
    saveconf = dic2dcpp.SaveConfig()
    saveconf.basepath = str(output_basepath)
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end

    # calling the c++ dic engine
    dic2dcpp.dic_engine(ref_arr, def_arr, roi_c, config, saveconf)
