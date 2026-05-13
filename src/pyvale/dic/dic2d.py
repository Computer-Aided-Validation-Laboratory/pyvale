# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


from logging import debug
import numpy as np
from pathlib import Path
from typing import Literal

# pyvale
import pyvale.dic.diccpp as diccpp
import pyvale.calib.calibcpp as calibcpp
import pyvale.dic.dicchecks as dicchecks
import pyvale.common_py.util as common_py_util
from pyvale.calib.calib_dataclass import Calib
import pyvale.common_cpp.common_cpp as common_cpp

def calculate_2d(reference: np.ndarray | str | Path,
                 deformed: np.ndarray | str | Path | list[Path],
                 roi_mask: np.ndarray,
                 seed: list[int] | list[np.int32] | np.ndarray,
                 subset_size: int = 21,
                 subset_step: int = 10,
                 correlation_criteria: Literal["ZNSSD","NSSD","SSD"]="ZNSSD",
                 shape_function: Literal["AFFINE","QUAD","RIGID"]="AFFINE",
                 interpolation_routine: Literal["BSPLINE","HERMITE"]="BSPLINE",
                 max_iterations: int=40,
                 precision: float=0.001,
                 threshold: float=0.9,
                 bf_threshold: float=0.6,
                 num_threads: int | None = None,
                 max_displacement: int=64,
                 method: Literal["MULTIWINDOW_RG","SINGLEWINDOW_RG","MULTIWINDOW","RASTER"] = "MULTIWINDOW_RG",
                 incremental: bool=False,
                 incremental_update_condition: Literal["IMAGE","COST","ITER"]="IMAGE",
                 incremental_update_value: float=1,
                 multiwindow_overlap: float=0.5,
                 multiwindow_subset_sizes: list[int] = [],
                 multiwindow_search_areas: list[int] = [],
                 fft_mad: bool=False,
                 fft_mad_scale: float=3.0,
                 fft_save: bool=False,
                 output_basepath: Path | str = "./",
                 output_binary: bool=False,
                 output_prefix: str="dic_results_",
                 output_delimiter: str=",",
                 output_below_threshold: bool=False,
                 output_shape_params: bool=False,
                 debug_level: int=1) -> None:

    """
    Perform 2D Digital Image Correlation (DIC) between a reference image and one or more deformed images.

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
        Coordinates `[x, y]` of the seed point for Reliability-Guided (RG) scanning. It's possible
        to provide mutiple seed points using `[x0, y0, x1, y1,...]` format. If the method is not 
        RG, this will be ignored.
    subset_size : int, optional
        Size of the square subset window in pixels (default: 21).
    subset_step : int, optional
        Step size between subset centers in pixels (default: 10).
    correlation_criteria : str, optional
        Metric for matching subsets: ``"ZNSSD"``, ``"NSSD"`` or ``"SSD"`` (default: ``"ZNSSD"``).
    shape_function : str, optional
        Deformation model: e.g., ``"AFFINE"``, ``"QUAD"``, ``"RIGID"`` (default: ``"AFFINE"``).
    interpolation_routine : str, optional
        Interpolation method used on image intensity. Options are ``"BSPLINE"`` and
        ``"HERMITE"``. Implementation details can be found in our DIC theory
        documentation.  (default: `"BSPLINE"``).
    max_iterations : int, optional
        Maximum number of iterations allowed for subset optimization (default: 40).
    precision : float, optional
        Precision threshold for iterative optimization convergence (default: 0.001).
    threshold : float, optional
        Minimum correlation/cost coefficient value to be considered a matching subset (default: 0.9).
    num_threads : int, optional
        Number of threads to use for parallel computation (default: ``None``, uses all available).
    bf_threshold : float, optional
        Correlation threshold used in rigid bruteforce check for a subset to be considered a
        good match(default: 0.6).
    max_displacement : int, optional
        Estimate for the Maximum displacement in any direction (in pixels) (default: 128).
    method : str, optional
        Subset scanning method: 
        * ``"MULTIWINDOW_RG"``: 
          for multi-window Reliability-Guided DIC (best overall approach),
        * ``"SINGLEWINDOW_RG"``:
          uses a single window for the rigid estimate for
          each subset. The size of the window is determined by the `max_displacement` parameter.
        * ``"MULTIWINDOW"``:
          only uses the multi-window FFT strategy. Only works
          for rigid shape functions and is not recommended for general use, but
          can be very fast for large rigid displacements.
        * ``"RASTER"``:
          no FFT initialization. Performs a raster scan of the image. 
          No seed location is used and work is split evenly between threads. 
          Not recommended except for testing with small displacements
    incremental : bool, optional
        If True, then references images will be updated depending on the
        condition set by argument `incremental_update_condition`. This is useful
        for large deformations where the original reference may no longer be
        valid for tracking. If False, the original reference image(s) will be
        used for tracking all deformed images. Displacements will still be given relative to the 
        first reference image. Note, cost values will be reported for relative to the deformed and 
        updated reference image. (default: False).
    incremental_update_condition : str, optional
        Condition for updating reference images when `incremental` is True. Options include:
        `"IMAGE"` to update every `N` images, `"COST"` to update when the average ZNCC cost
        value falls below a threshold, `"ITER"` to update when the average number
        of subset optimizer iterations exceeds a threshold. (default: `"PER_IMAGE"`).
    incremental_update_value : float, optional
        Value corresponding to the `incremental_update_condition`. For example,
        if the condition is "IMAGE", this would be the number of images after
        which to update the reference. If the condition is `"COST"`, this would be
        the cost threshold for updating. If the condition is `"ITER"`, this would
        be the iteration threshold for updating. (default: 1).
    multiwindow_overlap : float, optional
        For multi-window methods, the percentage overlap between adjacent FFT windows 
        at each level (default: 0.5).
    multiwindow_subset_sizes: list[int], optional
        List of subset_sizes for the multi-window FFT approach. If
        None, defaults to powers of 2 with the largest window size determined by
        the next power of 2 above `max_displacement` (default: None).
    multiwindow_search_areas: list[int], optional
        List of search window sizes for the multi-window FFT approach. If None,
        defaults to the corresponding template window size (default: None).
    fft_mad : bool, optional
        Median Absolute Deviation (MAD) outlier removal flag that 
        will kill likely incorrect spikes in the rigid estimates 
        for each FFTCC window size. (default: ``False``)
    fft_mad_scale : bool, optional
        An outlier is defined as a value whose deviation from the local median exceeds 
        ``fft_mad_scale`` times the MAD. This value choses the scaling factor that determines 
        the threshold for detecting outliers relative to the MAD. A larger ``fft_mad_scale`` 
        is more tolerant, while a smaller value kills larger deviations.
    output_basepath : str or pathlib.Path, optional
        Directory path where output files will be written (default: ``"./"``).
    output_binary : bool, optional
        Whether to write output in binary format (default: ``False``).
    output_prefix : str, optional
        Prefix for all output files (default: ``dic_results_``). results will be
        named with output_prefix + original filename. THe extension will be
        changed to ``".csv"`` or ``".dic2d"`` depending on whether outputting as a binary.
    output_delimiter : str, optional
        Delimiter used in text output files (default: ``","``).
    output_below_threshold : bool, optional
        If True, subset results with cost values that did not exceed the cost threshold
        will still be present in output (default: ``False``).
    output_shape_params : bool, optional
        If True, all shape parameters will be saved in the output files (default: ``False``).
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

    if isinstance(reference, (str, Path)) and isinstance(deformed, (str, Path)):
        basenames, fullpaths, w, h = dicchecks.check_images(reference,deformed,roi_mask, debug_level)
    else:
        raise ValueError("Currently only file paths are accepted for reference and deformed images. Please provide paths to the images you want to analyze.")
        #image_stack, roi_c, basenames, fullpaths = dicchecks.check_and_get_images(reference,deformed,roi_mask, debug_level)


    # string to enum
    method_enum = dicchecks.ScanMethod(method)
    shape_function_enum = dicchecks.Shape(shape_function)
    correlation_criteria_enum = dicchecks.CorrCrit(correlation_criteria)
    interpolation_routine_enum = dicchecks.Interp(interpolation_routine)
    incremental_update_condition_enum = dicchecks.InrementalMethod(incremental_update_condition)

    # checks on the config
    mw_overlap, mw_subset_size, mw_search_area  = dicchecks.multiwindow_init(subset_size,
                                                                 subset_step,
                                                                 max_displacement, 
                                                                 multiwindow_overlap, 
                                                                 multiwindow_subset_sizes,
                                                                 multiwindow_search_areas)

    dicchecks.check_thresholds(threshold, bf_threshold, precision)
    common_py_util.check_output_directory(str(output_basepath), output_prefix, debug_level)
    dicchecks.check_subsets(subset_size, subset_step)
    updated_seeds = dicchecks.check_and_update_rg_seed(seed, roi_mask, method, w, h, subset_size, subset_step)
    num_params = dicchecks.check_shape_function(shape_function_enum)


    # Assign values to config struct for c++ land
    config = diccpp.Config()
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = precision
    config.threshold = threshold
    config.bf_threshold = bf_threshold
    config.max_disp = max_displacement
    config.corr_crit = correlation_criteria_enum.value
    config.shape_func = shape_function_enum.value
    config.interp_routine = interpolation_routine_enum.value
    config.shape_func = shape_function_enum.value
    config.scan_method = method_enum.value
    config.incremental = incremental
    config.incremental_update_cond = incremental_update_condition_enum.value
    config.incremental_update_val = incremental_update_value

    config.num_params = num_params
    config.px_hori = w
    config.px_vert = h
    config.num_def_img = len(basenames)-1 # subtract ref image
    config.num_params = num_params
    config.rg_seeds = updated_seeds
    config.basenames = basenames
    config.fullpaths = fullpaths
    config.fft_mad = fft_mad
    config.fft_mad_scale = fft_mad_scale
    config.fft_save = fft_save
    config.debug_level = debug_level

    multiwindowconf = diccpp.MultiwindowConfig()
    multiwindowconf.overlap = mw_overlap
    multiwindowconf.subset_size = mw_subset_size
    multiwindowconf.search_area = mw_search_area

    # assigning c++ struct vals for save config
    saveconf = common_cpp.SaveConfig()
    saveconf.basepath = str(output_basepath)
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.output_below_threshold = output_below_threshold
    saveconf.shape_params = output_shape_params


    #TODO: sort this out so you can actually read in intrinsic parameters for
    # single camera DIC
    # Convert cam0
    cpp_cam0 = calibcpp.CamIntrinsics()
    cpp_cam0.fx = 0.0
    cpp_cam0.fy = 0.0
    cpp_cam0.fs = 0.0
    cpp_cam0.cx = 0.0
    cpp_cam0.cy = 0.0
    cpp_cam0.distortion = [0.0,0.0,0.0,0.0,0.0]

    # Convert cam1
    cpp_cam1 = calibcpp.CamIntrinsics()
    cpp_cam1.fx = 0.0
    cpp_cam1.fy = 0.0
    cpp_cam1.fs = 0.0
    cpp_cam1.cx = 0.0
    cpp_cam1.cy = 0.0
    cpp_cam1.distortion = [0.0,0.0,0.0,0.0,0.0]

    # Create C++ Calib object
    calib = calibcpp.Calib()
    calib.cam0 = cpp_cam0
    calib.cam1 = cpp_cam1
    calib.rotation = [0.0,0.0,0.0]
    calib.translation = [0.0,0.0,0.0]


    config.stereo = False

    #set the number of OMP threads
    if num_threads is not None:
        common_cpp.set_num_threads(num_threads)

    # calling the c++ dic engine
    with diccpp.ostream_redirect(stdout=True, stderr=True):
        diccpp.engine(roi_c, calib, config, multiwindowconf, saveconf)
