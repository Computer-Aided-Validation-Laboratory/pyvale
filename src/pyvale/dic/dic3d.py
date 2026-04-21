# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from logging import debug
import numpy as np
from pathlib import Path

# pyvale
import pyvale.dic.diccpp as diccpp
import pyvale.calib.calibcpp as calibcpp
import pyvale.dic.dicchecks as dicchecks
import pyvale.common_py.util as common_py_util
from pyvale.calib.calib_dataclass import Calib
import pyvale.common_cpp.common_cpp as common_cpp


def calculate_3d(reference: list[np.ndarray] | list[str] | list[Path],
                 deformed:  list[np.ndarray] | list[str] | list[Path],
                 roi_mask: np.ndarray,
                 calibration: Calib,
                 seed: list[int] | list[np.int32] | np.ndarray,
                 subset_size: int = 21,
                 subset_step: int = 10,
                 correlation_criteria: str="ZNSSD",
                 shape_function: str="AFFINE",
                 interpolation_routine: str="BSPLINE",
                 max_iterations: int=40,
                 precision: float=0.001,
                 threshold: float=0.9,
                 bf_threshold: float=0.6,
                 num_threads: int | None = None,
                 max_displacement: int=128,
                 method: str="MULTIWINDOW_RG",
                 fft_mad: bool=False,
                 fft_mad_scale: float=3.0,
                 output_at_end: bool=False,
                 output_basepath: Path | str = "./",
                 output_binary: bool=False,
                 output_prefix: str="dic_results_",
                 output_delimiter: str=",",
                 output_below_threshold: bool=False,
                 output_shape_params: bool=False,
                 debug_level: int=1) -> None:

    """
    Perform Stereo Digital Image Correlation (DIC) between a reference image and one or more deformed images.

    This function wraps a C++ DIC engine by preparing configuration parameters,
    performing input validation, and dispatching image data and settings. It supports
    pixel-level displacement and strain measurement over a defined region of interest (ROI).

    Parameters
    ----------
    reference : np.ndarray, str or pathlib.Path
        The reference image (2D array) or path to the image file.
    deformed : np.ndarray, str , pathlib.Path or list[pathlib.Path]
        The deformed image(s) (3D array for multiple images) or path/pattern to image files.
    roi_mask : np.ndarray
        A binary mask indicating the Region of Interest (ROI) for analysis (same size as image).
    seed : list[int], list[np.int32] or np.ndarray
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
    precision : float, optional
        Precision threshold for iterative optimization convergence (default: 0.001).
    threshold : float, optional
        Minimum correlation/cost coefficient value to be considered a matching subset (default: 0.9).
    num_threads : int, optional
        Number of threads to use for parallel computation (default: None, uses all available).
    bf_threshold : float, optional
        Correlation threshold used in rigid bruteforce check for a subset to be considered a
        good match(default: 0.6).
    max_displacement : int, optional
        Estimate for the Maximum displacement in any direction (in pixels) (default: 128).
    method : str, optional
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
    output_below_threshold : bool, optional
        If True, subset results with cost values that did not exceed the cost threshold
        will still be present in output (default: False).
    output_shape_params : bool, optional
        If True, all shape parameters will be saved in the output files (default: False).
    debug_level:

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



    if (debug_level>0):
        common_py_util.print_pyvale_banner()
        dicchecks.print_title("Initial Checks")

    # make sure ROI is in the correct format
    roi_c = np.ascontiguousarray(roi_mask)

    # do checks on vars in python land
    if (isinstance(reference[0], (str, Path)) and isinstance(deformed[0], (str, Path)) and 
        isinstance(reference[1], (str, Path)) and isinstance(deformed[1], (str, Path))):

        basenames0, fullpaths0, w0, h0 = dicchecks.check_images(reference[0],deformed[0],roi_mask,debug_level)
        basenames1, fullpaths1, w1, h1 = dicchecks.check_images(reference[1],deformed[1],roi_mask,debug_level)
    else:
        raise ValueError("Currently only file paths are accepted for reference and deformed images. Please provide paths to the images you want to analyze.")

    assert(w0 == w1)
    assert(h0 == h1)
    assert(len(basenames0) == len(basenames1))
    assert(len(basenames0) == len(basenames1))
    assert(len(fullpaths0) == len(fullpaths1))
    basenames = basenames0 + basenames1
    fullpaths = fullpaths0 + fullpaths1

    dicchecks.check_correlation_criteria(correlation_criteria)
    dicchecks.check_interpolation(interpolation_routine)
    dicchecks.check_method(method)
    dicchecks.check_thresholds(threshold, bf_threshold, precision)
    common_py_util.check_output_directory(str(output_basepath), output_prefix, debug_level)
    dicchecks.check_subsets(subset_size, subset_step)
    updated_seed = dicchecks.check_and_update_rg_seed(seed, roi_mask, method, w0, h0, subset_size, subset_step)
    num_params = dicchecks.check_shape_function(shape_function)

    # Assign values to config struct for c++ land
    config = diccpp.Config()
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = precision
    config.threshold = threshold
    config.bf_threshold = bf_threshold
    config.max_disp = max_displacement
    config.corr_crit = correlation_criteria
    config.shape_func = shape_function
    config.interp_routine = interpolation_routine
    config.scan_method = method
    config.px_hori = w0
    config.px_vert = h0
    config.num_def_img = len(basenames0)-1 # subtract ref image
    config.num_params = num_params
    config.rg_seed = updated_seed
    config.basenames = basenames
    config.fullpaths = fullpaths
    config.fft_mad = fft_mad
    config.fft_mad_scale = fft_mad_scale
    config.debug_level = debug_level

    # assigning c++ struct vals for save config
    saveconf = common_cpp.SaveConfig()
    saveconf.basepath = str(output_basepath)
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end
    saveconf.output_below_threshold = output_below_threshold
    saveconf.shape_params = output_shape_params


    # Convert cam0
    cpp_cam0 = calibcpp.CamIntrinsics()
    cpp_cam0.fx = calibration.cam0.fx
    cpp_cam0.fy = calibration.cam0.fy
    cpp_cam0.fs = calibration.cam0.fs
    cpp_cam0.cx = calibration.cam0.cx
    cpp_cam0.cy = calibration.cam0.cy
    cpp_cam0.distortion = calibration.cam0.distortion.tolist()

    # Convert cam1
    cpp_cam1 = calibcpp.CamIntrinsics()
    cpp_cam1.fx = calibration.cam1.fx
    cpp_cam1.fy = calibration.cam1.fy
    cpp_cam1.fs = calibration.cam1.fs
    cpp_cam1.cx = calibration.cam1.cx
    cpp_cam1.cy = calibration.cam1.cy
    cpp_cam1.distortion = calibration.cam1.distortion.tolist()

    # Create C++ Calib object
    calib = calibcpp.Calib()
    calib.cam0 = cpp_cam0
    calib.cam1 = cpp_cam1
    calib.rotation = calibration.rotation
    calib.translation = calibration.translation


    config.stereo = True

    #set the number of OMP threads
    if num_threads is not None:
        common_cpp.set_num_threads(num_threads)

    # calling the c++ dic engine
    with diccpp.ostream_redirect(stdout=True, stderr=True):
        diccpp.engine(roi_c, calib, config, saveconf)
