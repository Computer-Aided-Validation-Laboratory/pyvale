// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================




#ifndef DICINTERPOLATOR_H
#define DICINTERPOLATOR_H



// STD library Header files
#include <vector>

// Program Header files
#include "./dicutil.hpp"

namespace interpolator {

    struct Data {
        double interp_value;
        double interp_dx;
        double interp_dy;
    };

    inline void coeff_calc(std::vector<double> &tridiag_solution, double dy, double dx, size_t index, double * b, double * c, double * d);
    inline int index_lookup(std::vector<double> &px, double x, size_t index_lo, size_t index_hi);
    void cspline_init(std::vector<double> &px, std::vector<double> &data);
    double cspline_eval_deriv(std::vector<double> &px, std::vector<double> &data, double value, int length);
    void bicubic_init(util::Image *image);
    double eval_bicubic(double x, double y);
    double eval_bicubic_dx(double x, double y);
    double eval_bicubic_dy(double x, double y);
    Data eval_bicubic_and_derivs(double x, double y);

}

#endif //DICINTERPOLATOR_H




