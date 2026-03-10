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




struct Affine {
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);
    static void get_displacement(double &u, double &v, const double  x, const double  y, const std::vector<double> &p);
    static constexpr int num_params = 6;
};

struct Quad {
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);
    static void get_displacement(double &u, double &v, const double  x, const double  y, const std::vector<double> &p);
    static constexpr int num_params = 12;
};
struct Rigid {
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);
    static void get_displacement(double &u, double &v, const double  x, const double  y, const std::vector<double> &p);
    static constexpr int num_params = 2;
};

#endif // DICSHAPEFUNC_HPP
