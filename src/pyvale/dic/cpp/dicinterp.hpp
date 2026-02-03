// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICINTERP_H
#define DICINTERP_H

// STD library Header files

// Program Header files




inline int idx_from_2d(const int x, const int y, const int length){
    return y*length+x;
}



/**
 * @brief namespace for bicubic spline interpolation. 
 * 
 * Based on the implementation by GNU Scientific Library (GSL).
 * Main difference is the removal of the binary search for index lookup.
 * For use in DIC, we only ever need integer locations and therefore its
 * sufficient to get the floor value of the subpixel location.
 * 
 */
struct InterpVals {
    double f;
    double dfdx;
    double dfdy;
};

class Interpolator {
public:

    int px_vert;
    int px_hori;

    /**
     * @brief Evaluates the bicubic interpolation at a specified point.
     * 
     * Computes the interpolated value at (x,y) using bicubic interpolation from the surrounding pixel values.
     * 
     * @param x The x-coordinate of the interpolation point
     * @param y The y-coordinate of the interpolation point
     * @return The interpolated value at (x,y)
     */
    virtual double eval(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const = 0;

    /**
     * @brief Evaluates the x-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to x at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The x-derivative of the interpolated function at (x,y)
     */
    virtual double eval_dx(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const = 0;

    /**
     * @brief Evaluates the y-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to y at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The y-derivative of the interpolated function at (x,y)
     */
    virtual double eval_dy(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const = 0;

    /**
     * @brief Evaluates the bicubic interpolation and its derivatives at a specified point.
     * 
     * Computes the interpolated value and its partial derivatives at (x,y) in a single call.
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return Data struct containing the interpolated value and its x and y derivatives
     */
    virtual InterpVals eval_and_derivs(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const = 0;


    virtual ~Interpolator() = default;

};

#endif //DICINTERP_H




