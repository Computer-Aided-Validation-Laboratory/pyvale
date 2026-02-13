// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSHAPEFUNC_H
#define DICSHAPEFUNC_H

// STD library Header files
#include <vector>

// common_cpp header files
#include <Eigen/Dense>

// DIC Header files



namespace shapefunc {

    void get_pixel_affine(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    void get_pixel_rigid(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    void get_pixel_quad(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);

    void get_daffine_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);
    void get_drigid_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);
    void get_dquad_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);

    void get_displacement_affine(double &u, double &v, const double x, const double y, const std::vector<double> &p);
    void get_displacement_rigid(double &u, double &v, const double  x, const double  y, const std::vector<double > &p);
    void get_displacement_quad(double &u, double &v, const double  x, const double  y, const std::vector<double> &p);
}

#endif // DICSHAPEFUNC_HPP
