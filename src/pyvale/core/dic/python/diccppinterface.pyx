# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


cimport numpy as np
from libcpp.string cimport string
import numpy as np
import time

# import cpp libraries
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "../cpp/dicbuildinfo.hpp" namespace "dic":
    const char* get_cpu_comp()
    const char* get_git_info()
    const char* get_git_dirty()
    const char* get_hostname()
    const char* get_build_time()


def build_info():
    return {
        "g++ Compiler Version": get_cpu_comp().decode("utf-8"),
        "Git Commit": get_git_info().decode("utf-8"),
        "No. of C++ files with uncommited changes": int(get_git_dirty()),
        "host": get_hostname().decode("utf-8"),
        "build_time": get_build_time().decode("utf-8")
    }


cdef extern from "../cpp/dicutil.hpp" namespace "util":
    #extern vector[int] ss_coord_list
    extern vector[int] niter_arr
    extern vector[double] u_arr
    extern vector[double] v_arr
    extern vector[double] p_arr
    extern vector[double] ftol_arr
    extern vector[double] xtol_arr
    
    cdef cppclass SaveConfig:
        SaveConfig() except +
        string format
        string layout
        string base_path
        string prefix
        string delimiter





cdef extern from "../cpp/dicmain.hpp" namespace "dic":

    cdef cppclass Config:
        Config() except +
        int ss_step
        int ss_size
        int max_iter
        double precision
        double threshold_lm
        double threshold_bf
        int range_bf
        string corr_crit
        string shape_func
        string interp_routine
        string scan_method
        
    void engine_2d(int* image_ref, 
                    int* image_def, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    const Config &config,
                    const SaveConfig &saveconfig)




# A wrapper function to call the C++ function from Python
def cpp_2d_dic_routine(np.ndarray[np.int32_t, ndim=2] reference_image,
                      np.ndarray[np.int32_t, ndim=3] deformed_images,
                      np.ndarray[bool, ndim=2] roi_mask,
                      int subset_step, 
                      int subset_size,
                      int max_iterations,
                      double precision,       
                      double threshold_levenberg,       
                      double threshold_bruteforce,   
                      int range_bruteforce,    
                      str correlation_criteria,
                      str shape_function,
                      str interpolation_routine,
                      str scanning_method,
                      str out_format,
                      str out_layout,
                      str out_base_path,
                      str out_prefix,
                      str out_delimiter):

    # typed memoryviews for the image arrays
    cdef int[:, ::1] image_ref = reference_image
    cdef bool[:, ::1] image_roi = roi_mask
    cdef int[:, :, ::1] image_def_stack = deformed_images

    # the the image dimensions and the number of deformed images
    cdef int px_vertical = reference_image.shape[0]
    cdef int px_horizontal = reference_image.shape[1]
    cdef int num_def_images = deformed_images.shape[0]

    # DIC Configuration
    cdef Config config
    config.ss_step = subset_step
    config.ss_size = subset_size
    config.max_iter = max_iterations
    config.precision = precision
    config.threshold_lm = threshold_levenberg
    config.threshold_bf = threshold_bruteforce
    config.range_bf = range_bruteforce
    config.corr_crit = correlation_criteria.encode('utf-8')
    config.shape_func = shape_function.encode('utf-8')
    config.interp_routine = interpolation_routine.encode('utf-8')
    config.scan_method = scanning_method.encode('utf-8')

    # Save Config
    cdef SaveConfig saveconfig
    saveconfig.format = out_format.encode('utf-8')
    saveconfig.layout = out_layout.encode('utf-8')
    saveconfig.base_path = out_base_path.encode('utf-8')
    saveconfig.prefix = out_prefix.encode('utf-8')
    saveconfig.delimiter = out_delimiter.encode('utf-8')



    # call c++ 2D DIC engine
    engine_2d(&image_ref[0,0],
               &image_def_stack[0,0,0],
               &image_roi[0,0],
               px_vertical,
               px_horizontal,
               num_def_images,
               config,
               saveconfig)


    # Expose C++ result arrays as NumPy arrays (zero-copy)
    #cdef int[::1] ss_list_view = <int [:ss_coord_list.size()]>ss_coord_list.data()
    cdef int[::1] niter_arr_view = <int [:niter_arr.size()]>niter_arr.data()
    cdef double[::1] u_arr_view = <double [:u_arr.size()]>u_arr.data()
    cdef double[::1] p_arr_view = <double [:p_arr.size()]>p_arr.data()
    cdef double[::1] v_arr_view = <double [:v_arr.size()]>v_arr.data()
    cdef double[::1] ftol_arr_view = <double [:ftol_arr.size()]>ftol_arr.data()
    cdef double[::1] xtol_arr_view = <double [:xtol_arr.size()]>xtol_arr.data()

    # get 1d arrays without copying
    u_1d = np.frombuffer(u_arr_view, dtype=np.float64)
    v_1d = np.frombuffer(v_arr_view, dtype=np.float64)
    p_1d = np.frombuffer(p_arr_view, dtype=np.float64)
    niter_1d = np.frombuffer(niter_arr_view, dtype=np.int32)
    #subsets_1d = np.frombuffer(ss_list_view, dtype=np.int32)
    ftol_1d = np.frombuffer(ftol_arr_view, dtype=np.float64)
    xtol_1d = np.frombuffer(xtol_arr_view, dtype=np.float64)

    
    # reshape to desired dimension
    niter = niter_1d.reshape(num_def_images, px_horizontal//subset_step, px_vertical//subset_step)
    #subsets = subsets_1d.reshape(px_horizontal//subset_step, px_vertical//subset_step, 2)
    u = u_1d.reshape(num_def_images, px_vertical//subset_step, px_horizontal//subset_step)
    v = v_1d.reshape(num_def_images, px_horizontal//subset_step, px_vertical//subset_step)
    p = p_1d.reshape(num_def_images, 6 * px_horizontal//subset_step, px_vertical//subset_step)
    ftol = ftol_1d.reshape(num_def_images, px_horizontal//subset_step, px_vertical//subset_step)
    xtol = xtol_1d.reshape(num_def_images, px_horizontal//subset_step, px_vertical//subset_step)

    #return subsets, niter, u, v, p, ftol, xtol
    return niter, u, v, p, ftol, xtol
