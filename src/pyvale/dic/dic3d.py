# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
from logging import debug
import numpy as np
from pathlib import Path
from typing import Literal

# pyvale
import pyvale.dic.diccpp as diccpp
import pyvale.calib.calibcpp as calibcpp
import pyvale.dic.dicchecks as dicchecks
import pyvale.common_py.util as common_py_util
from pyvale.calib.calibdataclass import Calib
import pyvale.common_cpp.common_cpp as common_cpp


def calculate_3d(reference: list[np.ndarray] | list[str] | list[Path],
                 deformed:  list[np.ndarray] | list[str] | list[Path],
                 roi_mask: np.ndarray,
                 calibration: Calib,
                 seed: list[int] | list[np.int32] | list[tuple[int, int]] | np.ndarray,
                 subset_size: int = 21,
                 subset_step: int = 10,
                 correlation_criteria: Literal["ZNSSD","NSSD","SSD"]="ZNSSD",
                 shape_function: Literal["AFFINE","QUAD","RIGID"]="AFFINE",
                 interpolation_routine: Literal["BSPLINE","HERMITE"]="BSPLINE",
                 max_iterations: int=40,
                 precision: float=0.001,
                 threshold: float=0.9,
                 num_threads: int | None = None,
                 max_displacement: int=128,
                 epi_distance: int=300,
                 method: Literal["MULTIWINDOW_RG","SINGLEWINDOW_RG","MULTIWINDOW","RASTER"] = "MULTIWINDOW_RG",
                 incremental: bool=False,
                 incremental_update_condition: Literal["IMAGE","COST","ITER"]="IMAGE",
                 incremental_update_value: float | int=1,
                 multiwindow_overlap: float=0.0,
                 multiwindow_subset_sizes: list[int] = [],
                 multiwindow_search_areas: list[int] = [],
                 fft_filter: bool=True,
                 fft_filter_threshold: float=3.0,
                 fft_filter_radius: int=3,
                 fft_filter_corr_power: float=2.0,
                 fft_save: bool=False,
                 fft_precision: Literal["F64","F32"]="F32",
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
    seed : list[int], list[np.int32], list[tuple[int, int]] or np.ndarray
        Coordinates of the seed points for Reliability-Guided (RG) scanning. Accepts either
        the existing flat format ``[x0, y0, x1, y1, ...]`` or tuple format
        ``[(x0, y0), (x1, y1), ...]``. If the method is not RG, this will be ignored.
    subset_size : int, optional
        Size of the square subset window in pixels (default: 21).
    subset_step : int, optional
        Step size between subset centers in pixels (default: 10).
    correlation_criteria : str, optional
        Metric for matching subsets: ``"ZNSSD"``, ``"NSSD"`` or ``"SSD"`` (default: ``"ZNSSD"``).
    shape_function : str, optional
        Deformation model: e.g., "AFFINE", "RIGID" (default: "AFFINE").
    interpolation_routine : str, optional
        Interpolation method used on image intensity. Options are ``"BSPLINE"`` and ``"HERMITE"``. 
        Implementation details can be found in our DIC theory documentation. (default: ``“BSPLINE”``).
    max_iterations : int, optional
        Maximum number of iterations allowed for subset optimization (default: 40).
    precision : float, optional
        Precision threshold for iterative optimization convergence (default: 0.001).
    threshold : float, optional
        Minimum correlation/cost coefficient value to be considered a matching subset (default: 0.9).
    num_threads : int, optional
        Number of threads to use for parallel computation (default: None, uses all available).
    max_displacement : int, optional
        Estimate for the Maximum displacement for images from the same camera in any 
        direction (in pixels) (default: 128).
    epi_distance : int, optional
        Estimate for the maximum distance along the epipolar line (in pixels) between a identical point in 
        the left and right image (default: 300).
    method : str, optional
        The core algorithmic method used to perform the DIC.

        Options include:

        * ``"MULTIWINDOW_RG"``: Multi-window Reliability-Guided DIC
        (best overall approach).

        * ``"SINGLEWINDOW_RG"``: Uses a single window for the rigid estimate
        for each subset. The size of the window is determined by the
        ``max_displacement`` parameter.

        * ``"MULTIWINDOW"``: Uses only the multi-window FFT strategy.

        * ``"RASTER"``: No FFT initialization. Performs a raster scan of
        the image.

    incremental : bool, optional
        If True, then references images will be updated depending on the
        condition set by argument `incremental_update_condition`. This is useful
        for large deformations where the original reference may no longer be
        valid for tracking. If False, the original reference image(s) will be
        used for tracking all deformed images (default: False).
    incremental_update_condition : str, optional
        Condition for updating reference images when ``incremental`` is True. Options include:
        ``"IMAGE"`` to update every ``N`` images, ``"COST"`` to update when the average ZNCC cost
        value falls below a threshold, ``"ITER"`` to update when the average number
        of subset optimizer iterations exceeds a threshold. (default: `"PER_IMAGE"`).
    incremental_update_value : float, optional
        Value corresponding to the ``incremental_update_condition``. For example,
        if the condition is "IMAGE", this would be the number of images after
        which to update the reference. If the condition is ``"COST"``, this would be
        the cost threshold for updating. If the condition is ``"ITER"``, this would
        be the iteration threshold for updating. (default: 1).
    multiwindow_overlap : int, optional
        For multi-window methods, the percentage overlap between adjacent FFT windows 
        at each level (default: 50).
    multiwindow_template : list[int], optional
        List of template window sizes for the multi-window FFT approach. If
        None, defaults to powers of 2 with the largest window size determined by
        the next power of 2 above ``max_displacement`` (default: ``None``).
    multiwindow_search : list[int], optional
        List of search window sizes for the multi-window FFT approach. If None,
        defaults to the corresponding template window size (default: ``None``).
    fft_filter : bool, optional
        Enables outlier filtering for rigid FFT displacement estimates at each
        FFTCC window size. (default: ``False``)
    fft_filter_threshold : float, optional
        Rejection threshold for the FFT displacement outlier filter. Larger
        values are more tolerant, while smaller values reject more vectors.
        (default: ``3.0``)
    fft_filter_radius : int, optional
        Neighbourhood radius, in subset-grid steps, used by the FFT displacement
        outlier filter. (default: ``3``)
    fft_filter_corr_power : float, optional
        Exponent applied to correlation confidence when weighting neighbours in
        the FFT displacement outlier filter. (default: ``2.0``)
    fft_precision : str, optional
        Floating-point precision for FFT-only windowing buffers. Options are ``"F32"``
        for single precision and ``"F64"`` for double precision. (default: ``"F32"``).
    output_basepath : str or pathlib.Path, optional
        Directory path where output files will be written (default: ``"./"``).
    output_binary : bool, optional
        Whether to write output in binary format (default: False).
    output_prefix : str, optional
        Prefix for all output files (default: ``"dic_results_"``). results will be
        named with output_prefix + original filename. THe extension will be
        changed to ``".csv"`` or ``".dic2d"`` depending on whether outputting as a binary.
    output_delimiter : str, optional
        Delimiter used in text output files (default: ``","``).
    output_below_threshold : bool, optional
        If ``True``, subset results with cost values that did not exceed the cost threshold
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
        common_py_util.print_title("Initial Checks")

    # make sure ROI is in the correct format
    roi_c = np.ascontiguousarray(roi_mask)

    # do checks on vars in python land
    basenames0, fullpaths0, w0, h0, temp_dir = dicchecks._check_images(reference[0],deformed[0],roi_mask,debug_level)
    basenames1, fullpaths1, w1, h1, temp_dir = dicchecks._check_images(reference[1],deformed[1],roi_mask,debug_level)

    assert(w0 == w1)
    assert(h0 == h1)
    assert(len(basenames0) == len(basenames1))
    assert(len(basenames0) == len(basenames1))
    assert(len(fullpaths0) == len(fullpaths1))
    basenames = basenames0 + basenames1
    fullpaths = fullpaths0 + fullpaths1

    # string to enum
    method_enum = dicchecks.ScanMethod(method)
    shape_function_enum = dicchecks.Shape(shape_function)
    correlation_criteria_enum = dicchecks.CorrCrit(correlation_criteria)
    interpolation_routine_enum = dicchecks.Interp(interpolation_routine)
    incremental_update_condition_enum = dicchecks.IncrementalMethod(incremental_update_condition)

    # checks on the config
    mw_overlap, mw_subset_size, mw_search_area  = dicchecks._multiwindow_init(subset_size,
                                                                 subset_step,
                                                                 w0, h1,
                                                                 max_displacement, 
                                                                 multiwindow_overlap, 
                                                                 multiwindow_subset_sizes, 
                                                                 multiwindow_search_areas)



    # checks on the config
    dicchecks._check_thresholds(threshold, precision)
    common_py_util.check_output_directory(str(output_basepath), output_prefix, debug_level)
    dicchecks._check_subsets(subset_size, subset_step)
    updated_seeds = dicchecks._check_and_update_rg_seed(seed, roi_mask, method, w0, h0, subset_size, subset_step)
    num_params = dicchecks._check_shape_function(shape_function_enum)

    # Assign values to config struct for c++ land
    config = diccpp.Config()
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = precision
    config.threshold = threshold
    config.corr_crit = getattr(diccpp.CorrCrit, correlation_criteria_enum.name)
    config.shape_func = getattr(diccpp.ShapeFunc, shape_function_enum.name)
    config.interp_routine = getattr(diccpp.InterpRoutine, interpolation_routine_enum.name)
    config.shape_func = getattr(diccpp.ShapeFunc, shape_function_enum.name)
    config.scan_method = getattr(diccpp.ScanMethod, method_enum.name)
    config.incremental = incremental
    config.incremental_update_cond = getattr(diccpp.IncrementalCond, incremental_update_condition_enum.value)
    config.incremental_update_val = incremental_update_value
    config.px_hori = w0
    config.px_vert = h0
    config.num_def_img = len(basenames0)-1 # subtract ref image
    config.num_params = num_params
    config.rg_seeds = updated_seeds
    config.basenames = basenames
    config.fullpaths = fullpaths
    config.fft_filter = fft_filter
    config.fft_filter_threshold = fft_filter_threshold
    config.fft_filter_radius = fft_filter_radius
    config.fft_filter_corr_power = fft_filter_corr_power
    config.fft_save = fft_save
    config.debug_level = debug_level
    config.epi_distance = epi_distance
    config.max_disp = max_displacement

    # sort precision to use for FFT windowing
    if fft_precision=="F32":
        config.fft_precision = diccpp.FFTPrecision.FLOAT32
    elif fft_precision=="F64":
        config.fft_precision = diccpp.FFTPrecision.FLOAT64
    else:
        raise ValueError("fft_precision must be one of: F64, F32")

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

    dicchecks._print_config_summary(
        w0, h0, config.num_def_img, max_iterations, correlation_criteria,
        shape_function, interpolation_routine, fft_filter,
        fft_filter_threshold, fft_filter_radius, fft_filter_corr_power, method,
        precision, threshold, max_displacement, subset_size, subset_step,
        num_threads, debug_level, updated_seeds, epi_distance
    )

    # calling the c++ dic engine
    with diccpp.ostream_redirect(stdout=True, stderr=True):
        diccpp.engine(roi_c, calib, config, multiwindowconf, saveconf)


    if temp_dir is not None:

        # delete each file in filename
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        os.rmdir(temp_dir)
