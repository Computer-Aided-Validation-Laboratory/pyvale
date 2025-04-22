// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <vector>

// Program Header files
#include "./dicinterpolator.cpp"
#include "./dicsmooth.hpp"
#include "dicinterpolator.hpp"

namespace strain {

    
    void engine(std::string interp, std::string tensor) {
    
        std::vector<double> smoothed;

        int ss_step = 10;
        int ss_size = 51;
        int sw = 5;
        int vsg = ((sw-1) * ss_step) + ss_size;
        int num_def_images = 1;
    
        // loop over the displacement images
        for (int img = 0; img < num_def_images; img++) {
    
            // smooth the displacement field
            smooth::bilinear(

            // differentiate the dispacelemt field (differentiate along x then along y)
            interpolator::cspline_eval_deriv(std::vector<double> &px, std::vector<double> &field, double value, int length);
            interpolator::cspline_eval_deriv(std::vector<double> &px, std::vector<double> &field, double value, int length);

            // strain calculation
        }


    }





} // namespace strain
