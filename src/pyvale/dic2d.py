# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================



import numpy as np

# import cython module
import pyvale.dic2dcpp as dic2dcpp
import pyvale.dicchecks as dicchecks


def dic_2d(reference: np.ndarray | str,
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
          output_prefix: str="dic_results_",
          output_delimiter: str=" ") -> None:

    # do checks on vars in python land
    dicchecks.print_title("Initial Checks")
    ref_arr, def_arr, roi_c, filenames = dicchecks.check_and_get_images(reference,deformed,roi_mask)
    dicchecks.check_correlation_criteria(correlation_criteria)
    dicchecks.check_interpolation(interpolation_routine)
    dicchecks.check_scanning_method(scanning_method)
    dicchecks.check_thresholds(opt_threshold, bf_threshold, opt_precision)
    dicchecks.check_output_directory(output_basepath, output_prefix)
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

    # assigning c++ struct vals for save config
    saveconf = dic2dcpp.SaveConfig()
    saveconf.basepath = output_basepath
    saveconf.binary = output_binary
    saveconf.prefix = output_prefix
    saveconf.delimiter = output_delimiter
    saveconf.at_end = output_at_end

    # calling the c++ dic engine
    dic2dcpp.dic_engine(ref_arr, def_arr, roi_c, config, saveconf)
