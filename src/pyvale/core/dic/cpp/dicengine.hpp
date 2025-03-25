// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICENGINE_H
#define DICENGINE_H


// STD library Header files
#include <vector>
#include <iostream>


namespace dic2d {


    void dicengine(int* image_ref, 
                    int* image_def_stack, 
                    int* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    int max_iter,
                    double tol,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method);

    void image_scan(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step, int max_iter, double tol);
    void reliability_guided(int n_img, int n_subset, int edge, int px_horizontal, int px_vertical, int subset_size, int subset_step, int max_iter, double tol);

}

#endif //DICENGINE_H
