"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""


cimport numpy as np
from libcpp.string cimport string
import numpy as np
import time
from icecream import ic

# import cpp libraries
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "../cpp/dicengine.hpp" namespace "dic2d":
    void dicengine(int* image_ref, 
                    int* image_def, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    int max_iter,
                    double tol,
                    string& corr_crit, 
                    string& shape_func,
                    string& interp_routine,
                    string& scan_method)


    # Declare the result arrays 
    extern vector[int] ss_coord_list
    extern vector[int] niter_arr
    extern vector[double] u_arr
    extern vector[double] v_arr
    extern vector[double] p_arr
    # extern vector[double] ftol_arr
    # extern vector[double] xtol_arr

# A wrapper function to call the C++ function from Python
def cpp_2d_dic_routine(np.ndarray[np.int32_t, ndim=2] reference_image,
                      np.ndarray[np.int32_t, ndim=3] deformed_images,
                      np.ndarray[bool, ndim=2] roi_mask,
                      int subset_step, 
                      int subset_size,
                      int max_iterations,
                      double tolerance,       
                      str correlation_criteria,
                      str shape_function,
                      str interpolation_routine,
                      str scanning_method):

    # typed memoryviews for the image arrays
    cdef int[:, ::1] image_ref = reference_image
    cdef bool[:, ::1] image_roi = roi_mask
    cdef int[:, :, ::1] image_def_stack = deformed_images

    # the the image dimensions and the number of deformed images
    cdef int px_vertical = reference_image.shape[0]
    cdef int px_horizontal = reference_image.shape[1]
    cdef int num_def_images = deformed_images.shape[0]

    cdef int max_iter = max_iterations
    cdef double tol = tolerance


    # other arguments needed for C++ DIC routine
    cdef string corr_crit = correlation_criteria.encode('utf-8')
    cdef string shape_func = shape_function.encode('utf-8')
    cdef string interp_routine = interpolation_routine.encode('utf-8')
    cdef string scan_method = scanning_method.encode('utf-8')


    # call c++ 2D DIC engine
    dicengine(&image_ref[0,0],
               &image_def_stack[0,0,0],
               &image_roi[0,0],
               px_vertical,
               px_horizontal,
               num_def_images,
               subset_step,
               subset_size,
               max_iter,
               tol,
               corr_crit, 
               shape_func,
               interp_routine,
               scan_method)


    # Expose C++ result arrays as NumPy arrays (zero-copy)
    cdef int[::1] ss_list_view = <int [:ss_coord_list.size()]>ss_coord_list.data()
    cdef int[::1] niter_arr_view = <int [:niter_arr.size()]>niter_arr.data()
    cdef double[::1] u_arr_view = <double [:u_arr.size()]>u_arr.data()
    cdef double[::1] p_arr_view = <double [:p_arr.size()]>p_arr.data()
    cdef double[::1] v_arr_view = <double [:v_arr.size()]>v_arr.data()
    
    u_1d = np.frombuffer(u_arr_view, dtype=np.float64)
    v_1d = np.frombuffer(v_arr_view, dtype=np.float64)
    p_1d = np.frombuffer(p_arr_view, dtype=np.float64)
    niter_1d = np.frombuffer(niter_arr_view, dtype=np.int32)
    subsets_1d = np.frombuffer(ss_list_view, dtype=np.int32)

    niter = niter_1d.reshape(num_def_images, ss_coord_list.size()//2)
    subsets = subsets_1d.reshape(ss_coord_list.size()//2, 2)
    u = u_1d.reshape(num_def_images, ss_coord_list.size()//2)
    v = v_1d.reshape(num_def_images, ss_coord_list.size()//2)
    p = p_1d.reshape(num_def_images, 6 * ss_coord_list.size()//2)

    