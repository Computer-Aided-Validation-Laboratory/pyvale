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
#include "./dicresults.hpp"



namespace shapefunc {

    // Function pointer 
    extern void (*get_pixel)(double &, double &, const double, const double, const std::vector<double> &);
    extern void (*get_dfdp)(std::vector<double>&, double, double, double, double);
    extern void (*get_displacement)(OptResult &result, double ss_x, double ss_y, std::vector<double> &p);

    // Shape function declarations
    inline void get_pixel_affine(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    inline void get_pixel_rigid(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);
    inline void get_pixel_quad(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);

    int num_params(std::string &shape_func);


    // Setter for the current function
    void set(const std::string &func_name);

}
#endif // DICSMOOTH_H
