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

    void init(std::string &interp_routine, std::string &shape_func, int subset_size, gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc);
    int cost_function(const gsl_vector *p_gsl, void *data, gsl_vector * f);
    int jacobian_function(const gsl_vector *p_gsl, void *data, gsl_matrix *J);
    void update_data(std::vector<double> &subset_coords_x,std::vector<double> &subset_coords_y, std::vector<double> &subset);
    void execute(bool seed=true, double xtol=1e-15, double gtol=1e-20,double ftol=1e-15,int max_iter=100);

}

#endif //DICOPTIMIZATION_H