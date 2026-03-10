// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <string>
#include <cmath>

// Program Header files
#include "./dicshapefunc.hpp"

    // Shape function declarations
    void Affine::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2]) * x + p[3] * y;
        y_new = p[1] + (1.0+p[5]) * y + p[4] * x;
    }

    void Rigid::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + x;
        y_new = p[1] + y;
    }

    void Quad::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x + p[7]*x*y + p[8]*y*y;
        y_new = p[1] + (1.0+p[5])*y + p[4]*x + p[9]*x*x + p[10]*x*y + p[11]*y*y;
    }

    void Quad::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x + p[7]*x*y + p[8]*y*y;
        double y_new = p[1] + (1.0+p[5])*y + p[4]*x + p[9]*x*x + p[10]*x*y + p[11]*y*y;
        u = x_new - x;
        v = y_new - y;
    }

    void Affine::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2]) * x + p[3] * y;
        double y_new = p[1] + (1.0+p[5]) * y + p[4] * x;
        u = x_new - x;
        v = y_new - y;
    }

    void Rigid::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        u = p[0];
        v = p[1];
    }


    void Affine::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy){
        dfdp[0] = dfdx;
        dfdp[1] = dfdy;
        dfdp[2] = dfdx * x;
        dfdp[3] = dfdx * y;
        dfdp[4] = dfdy * x;
        dfdp[5] = dfdy * y;
    }

    void Rigid::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y,  const double dfdx, const double dfdy){
            dfdp[0] = dfdx;
            dfdp[1] = dfdy;
    }

    void Quad::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy){
        dfdp[0]  = dfdx;
        dfdp[1]  = dfdy;
        dfdp[2]  = dfdx * x;
        dfdp[3]  = dfdx * y;
        dfdp[4]  = dfdy * x;
        dfdp[5]  = dfdy * y;
        dfdp[6]  = dfdx * x*x;
        dfdp[7]  = dfdx * x*y;
        dfdp[8]  = dfdx * y*y;
        dfdp[9]  = dfdy * x*x;
        dfdp[10] = dfdy * x*y;
        dfdp[11] = dfdy * y*y;
    }
