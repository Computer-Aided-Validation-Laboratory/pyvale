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


cdef extern from "../cpp/dicengine.hpp" namespace "dic2d":
    void dicengine(int* image_ref, 
                    int* image_def, 
                    int* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    string& corr_crit, 
                    string& shape_func,
                    string& interp_routine,
                    string& scan_method)

cdef extern from "../cpp/dicoptimization.hpp" namespace "optimization":
    void collect_results(int *ss_coords, int* u, int* v, int* p)


# A wrapper function to call the C++ function from Python
def cpp_2d_dic_routine(np.ndarray[np.int32_t, ndim=2] reference_image,
                      np.ndarray[np.int32_t, ndim=3] deformed_images,
                      np.ndarray[np.int32_t, ndim=2] roi_mask,
                      int subset_step, 
                      int subset_size,                    
                      str correlation_criteria,
                      str shape_function,
                      str interpolation_routine,
                      str scanning_method):

    # typed memoryviews for the image arrays
    cdef int[:, ::1] image_ref = reference_image
    cdef int[:, ::1] image_roi = roi_mask
    cdef int[:, :, ::1] image_def = deformed_images

    # the the image dimensions and the number of deformed images
    cdef int px_vertical = reference_image.shape[0]
    cdef int px_horizontal = reference_image.shape[1]
    cdef int num_def_images = deformed_images.shape[0]


    # other arguments needed for C++ DIC routine
    cdef string corr_crit = correlation_criteria.encode('utf-8')
    cdef string shape_func = shape_function.encode('utf-8')
    cdef string interp_routine = interpolation_routine.encode('utf-8')
    cdef string scan_method = scanning_method.encode('utf-8')


    # call c++ 2D DIC engine
    dicengine(&image_ref[0,0],
               &image_def[0,0,0],
               &image_roi[0,0],
               px_vertical,
               px_horizontal,
               num_def_images,
               subset_step,
               subset_size,
               corr_crit, 
               shape_func,
               interp_routine,
               scan_method)


    # collecting the results
    n_images = deformed_images.shape[0]
    n_subsets = 1234


    ss_coords = np.zeros((2,n_subsets), dtype=np.int32)
    u_arr = np.zeros((n_images, n_subsets), dtype=np.float64)
    v_arr = np.zeros((n_images, n_subsets), dtype=np.float64)
    niter = np.zeros((n_images, n_subsets), dtype=np.int32)
    p_arr = np.zeros((n_images, 6, n_subsets), dtype=np.float64)

    # memoryviews
    cdef int[:, ::1] c_ss_coords = ss_coords
    cdef int[:, ::1] c_niter = niter
    cdef double[:, ::1] c_u_arr = u_arr
    cdef double[:, ::1] c_v_arr = v_arr
    cdef double[:, :, ::1] c_p_arr = p_arr

    print("here")
    # collect_results(&c_ss_coords[0,0], &c_u[0,0], &c_v[0,0], &c_p[0,0,0], &c_niter[0,0])

    # # starting timer
    # time_start_loop = time.perf_counter()


    # # std::vector to np.ndarray coercion. See here for more info on syntax: 
    # #       https://github.com/cython/cython/issues/4487
    # #       https://stackoverflow.com/questions/59666307/convert-c-vector-to-numpy-array-in-cython-without-copying

    # cdef double[::1] test1 = <double [:image_buffer_c.size()]>image_buffer_c.data()
    # cdef double[::1] test2 = <double [:depth_buffer_c.size()]>depth_buffer_c.data()

    # np_image_buffer = np.asarray(test1).copy()
    # np_depth_buffer = np.asarray(test2).copy()

    # # convert back to a 2d array for easy integration back into python code. suprisingly quick!
    # image_buffer_2d = np_image_buffer.reshape(buffer_height,buffer_width)
    # depth_buffer_2d = np_depth_buffer.reshape(buffer_height,buffer_width)

    # #ending timer
    # time_end_loop = time.perf_counter()
    # time_cpp_loop = time_end_loop - time_start_loop
    # print(f"{'Cython coercion of vector to np.array time':75}" + f"{time_cpp_loop:.8f}" + " [s]")

    
    # return depth_buffer_2d