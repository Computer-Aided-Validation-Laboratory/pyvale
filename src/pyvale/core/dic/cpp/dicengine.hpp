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

// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicdeformed.hpp"
#include "./dicoptimization.hpp"


// #ifdef DEBUG
//     #define LOG(x) std::cout << "[VERBOSE] " << x << std::endl;
// #else
//     #define LOG(x)  // Do nothing
// #endif


namespace dic2d {


    void dicengine(int* image_ref, 
                    int* image_def_stack, 
                    int* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method);

    void image_scan(int px_horizontal, int px_vertical, int subset_size, int subset_step);
    void reliability_guided(int px_horizontal, int px_vertical, int subset_size, int subset_step);

}

#endif //DICENGINE_H
