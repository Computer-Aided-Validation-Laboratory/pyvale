cimport numpy as np
import numpy as np
import time

# import cpp libraries
from libcpp.vector cimport vector


# Declare a C++ function signature using `cdef extern`
cdef extern from "src/raytrace.hpp":
    void raytrace_gpu_setup(
        double focal_length,
        double pixel_size,
        int buffer_width,
        int buffer_height,
        np.ndarray[np.float64_t, ndim=1] cam_pos_world,
        np.ndarray[np.float64_t, ndim=2] cam_rot, # (3, 3)
        np.ndarray[np.float64_t, ndim=1] coords_data,
        vector[double] &depth_buffer_c,
        vector[double] &image_buffer_c)

    # vector[double] return_image_buffer()
    # vector[double] return_depth_buffer()



# A wrapper function to call the C++ function from Python
def call_gpu_raytrace(
        double focal_length,
        double pixel_size,
        int buffer_width,
        int buffer_height,
        np.ndarray[np.float64_t, ndim=1] cam_pos_world,
        np.ndarray[np.float64_t, ndim=2] cam_rot,
        np.ndarray[np.float64_t, ndim=1] coords_world):


    cdef vector[double] image_buffer_c
    cdef vector[double] depth_buffer_c

    # call cuda raytracer
    raytrace_gpu_setup(focal_length,
    pixel_size,
    buffer_width,
    buffer_height,
    cam_pos_world, 
    cam_rot, 
    coords_world, 
    depth_buffer_c,
    image_buffer_c)


    # starting timer
    time_start_loop = time.perf_counter()


    # std::vector to np.ndarray coercion. See here for more info on syntax: 
    #       https://github.com/cython/cython/issues/4487
    #       https://stackoverflow.com/questions/59666307/convert-c-vector-to-numpy-array-in-cython-without-copying

    cdef double[::1] test1 = <double [:image_buffer_c.size()]>image_buffer_c.data()
    cdef double[::1] test2 = <double [:depth_buffer_c.size()]>depth_buffer_c.data()

    np_image_buffer = np.asarray(test1).copy()
    np_depth_buffer = np.asarray(test2).copy()

    # convert back to a 2d array for easy integration back into python code. suprisingly quick!
    image_buffer_2d = np_image_buffer.reshape(buffer_height,buffer_width)
    depth_buffer_2d = np_depth_buffer.reshape(buffer_height,buffer_width)

    #ending timer
    time_end_loop = time.perf_counter()
    time_cpp_loop = time_end_loop - time_start_loop
    print(f"{'Cython coercion of vector to np.array time':75}" + f"{time_cpp_loop:.8f}" + " [s]")

    
    return depth_buffer_2d