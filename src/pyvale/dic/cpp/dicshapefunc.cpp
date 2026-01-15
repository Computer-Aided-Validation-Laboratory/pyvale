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
#include "./dicresults.hpp"

namespace shapefunc {

    // Function pointer 
    void (*get_pixel)(double &, double &, const double, const double, const std::vector<double> &);
    void (*get_dfdp)(std::vector<double>&, double, double, double, double);
    void (*get_displacement)(OptResult &results, double ss_x, double ss_y, std::vector<double> &p);


    // Shape function declarations
    inline void get_pixel_affine(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2]) * x + p[3] * y;
        y_new = p[1] + (1.0+p[5]) * y + p[4] * x;
    }

    inline void get_pixel_rigid(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + x;
        y_new = p[1] + y;
    }

    inline void get_pixel_quad(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x + p[7]*x*y + p[8]*y*y;
        y_new = p[1] + (1.0+p[5])*y + p[4]*x + p[9]*x*x + p[10]*x*y + p[11]*y*y;
    }

    inline void get_displacement_from_quad(OptResult &res, double ss_x, double ss_y, std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2])*ss_x + p[3]*ss_y + p[6]*ss_x*ss_x + p[7]*ss_x*ss_y + p[8]*ss_y*ss_y;
        double y_new = p[1] + (1.0+p[5])*ss_y + p[4]*ss_x + p[9]*ss_x*ss_x + p[10]*ss_x*ss_y + p[11]*ss_y*ss_y;
        res.u = x_new - ss_x;
        res.v = y_new - ss_y;
        res.mag = std::sqrt(res.u * res.u + res.v * res.v);
    }

    inline void get_displacement_from_affine(OptResult &res, double ss_x, double ss_y, std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2]) * ss_x + p[3] * ss_y;
        double y_new = p[1] + (1.0+p[5]) * ss_y + p[4] * ss_x;
        res.u = x_new - ss_x;
        res.v = y_new - ss_y;
        res.mag = std::sqrt(res.u * res.u + res.v * res.v);
    }

    inline void get_displacement_from_rigid(OptResult &res, double ss_x, double ss_y, std::vector<double> &p){
        res.u = p[0];
        res.v = p[1];
        res.mag = std::sqrt(res.u*res.u + res.v*res.v);
    }


    inline void get_daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy){
        dfdp[0] = dfdx;
        dfdp[1] = dfdy;
        dfdp[2] = dfdx * x;
        dfdp[3] = dfdx * y;
        dfdp[4] = dfdy * x;
        dfdp[5] = dfdy * y;
    }

    inline void get_drigid_dp(std::vector<double> &dfdp, double x, double y,  double dfdx, double dfdy){
            dfdp[0] = dfdx;
            dfdp[1] = dfdy;
    }

    inline void get_dquad_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy){
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


    // Setter for the current function
    void set(const std::string& shape_func) {
        if (shape_func == "RIGID") {
            get_pixel = get_pixel_rigid;
            get_dfdp = get_drigid_dp;
            get_displacement = get_displacement_from_rigid;
        }
        else if (shape_func == "AFFINE") {
            get_pixel = get_pixel_affine;
            get_dfdp = get_daffine_dp;
            get_displacement = get_displacement_from_affine;
        }
        else if (shape_func == "QUAD") {
            get_pixel = get_pixel_quad;
            get_dfdp = get_dquad_dp;
            get_displacement = get_displacement_from_quad;
        }
        else {
            std::cerr << "Unexpected Shape Function: '" << shape_func << "'" << std::endl;
            std::cerr << "Allowed Values: 'RIGID', 'AFFINE', 'QUAD'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

}
