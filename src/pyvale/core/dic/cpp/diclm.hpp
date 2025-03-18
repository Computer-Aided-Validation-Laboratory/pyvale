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

    void init(std::string &, std::string &, int);

    void  solve(std::vector<double> &, std::vector<double> &,  std::vector<double> &, gsl_spline2d*, gsl_interp_accel*,  gsl_interp_accel*, int n);
    void    ssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, gsl_spline2d*, gsl_interp_accel*,  gsl_interp_accel*, int n);
    void   nssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, gsl_spline2d*, gsl_interp_accel*,  gsl_interp_accel*, int n);
    void  znssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, gsl_spline2d*, gsl_interp_accel*,  gsl_interp_accel*, int n);

    bool invertMatrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& inverse);
    void new_shape_func_params(std::vector<double> &pdp, std::vector<std::vector<double>> &invH, std::vector<double> &gradient);

    // shape functions and their derivatives with respect to optimization parameters
    void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void drigid_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
}

#endif //DICLM_H