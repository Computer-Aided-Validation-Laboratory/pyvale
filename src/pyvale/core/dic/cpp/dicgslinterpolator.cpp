// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>


// GNU Scientific Library Header files
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_interp2d.h>


// Program Header files
#include "./dicgslinterpolator.hpp"



namespace gsl_interpolation {

    gsl_spline2d* create_spline(std::string &interp_type, std::vector<double> &image_ref_dbl, int px_horizontal, int px_vertical){

        std::vector<double> x(px_horizontal,0);
        for (int i = 0; i < px_horizontal; ++i) {
            x[i] = i; 
        }

        std::vector<double> y(px_vertical,0);
        for (int i = 0; i < px_vertical; ++i) {
            y[i] = i; 
        }
        

        // Declare Tinterp outside the condition block
        const gsl_interp2d_type *Tinterp = nullptr;

        // Set interpolator type based on interp_type
        if (interp_type == "bicubic") {
            Tinterp = gsl_interp2d_bicubic;
        } else if (interp_type == "bilinear") {
            Tinterp = gsl_interp2d_bilinear;
        } else {
            std::cerr << "Unknown Interpolator type: \'" << interp_type << "\'." << std::endl;
            std::cerr << "Allowed values: bicubic, bilinear. " << std::endl;
            return nullptr; // Or handle error as necessary
        }

        // Create the interpolator object using Tinterp
        gsl_spline2d *spline = gsl_spline2d_alloc(Tinterp, px_horizontal, px_vertical);

        // initialise our interpolator with the required image
        gsl_spline2d_init(spline, x.data(), y.data(), image_ref_dbl.data(), px_horizontal, px_vertical);

        return spline;

    }
}