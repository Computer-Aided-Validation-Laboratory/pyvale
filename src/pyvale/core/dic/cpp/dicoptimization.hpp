// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICOPTIMIZATION_H
#define DICOPTIMIZATION_H

// STD library Header files
#include <vector>
#include <iostream>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>


// Program Header files
#include "./dicinterpolator.hpp"

namespace optimization {

    struct Data {
        std::vector<double> subset_coords_x;
        std::vector<double> subset_coords_y;
        std::vector<double> subset_values;
        gsl_spline2d* spline;  
        gsl_interp_accel* xacc;
        gsl_interp_accel* yacc;
    };
 
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    gsl_vector *p_arr;
    Data optData;
    int p_length;


    void init(std::string &interp_routine, std::string &shape_func, int subset_size, gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc);
    int cost_function(const gsl_vector *p_gsl, void *data, gsl_vector * f);
    int jacobian_function(const gsl_vector *p_gsl, void *data, gsl_matrix *J);
    void update_data(std::vector<double> &subset_coords_x,std::vector<double> &subset_coords_y, std::vector<double> &subset);



}

#endif //DICOPTIMIZATION_H