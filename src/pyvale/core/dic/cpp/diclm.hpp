// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICLM_H
#define DICLM_H

// STD library Header files
#include <vector>
#include <iostream>
#include <chrono>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>

// Program Header files
#include "./dicinterpolator.hpp"

namespace lm {

    void init(int n);

    void loop(std::vector<double> &subset_def,
                 std::vector<double> &subset_def_x,
                 std::vector<double> &subset_def_y,
                 gsl_spline2d *spline,
                 gsl_interp_accel* xacc,
                 gsl_interp_accel* yacc,
                 int n);
    


    void subset_def_params(std::vector<double> &subset_def, int n);
    
    void subset_ref_params(std::vector<double> &p,
                           std::vector<double> &subset_coords_x, 
                           std::vector<double> &subset_coords_y, 
                           gsl_spline2d* spline,
                           gsl_interp_accel* xacc,
                           gsl_interp_accel* yacc,
                           int n);


    void calculate_q(
                 std::vector<double> &subset_def,                           \
                 std::vector<double> &subset_coords_x, 
                 std::vector<double> &subset_coords_y, 
                 gsl_spline2d* spline,
                 gsl_interp_accel* xacc,
                 gsl_interp_accel* yacc,
                 int n);

    void calculate_hessian();

    void calculate_costfunc_p(std::vector<double> &subset_ref, std::vector<double> &subset_def, int n);

    void calculate_costfunc_pdp(std::vector<double> &subset_def, std::vector<double> &subset_def_x, std::vector<double> &subset_def_y,
                                gsl_spline2d *spline, gsl_interp_accel* xacc, gsl_interp_accel* yacc, int n);
    void calculate_deltap();

    bool invertMatrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& inverse);

}

#endif //DICLM_H