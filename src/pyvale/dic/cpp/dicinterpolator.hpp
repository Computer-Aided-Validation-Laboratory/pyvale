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

private: 

    /**
     * @brief Calculates the coefficients for cubic spline interpolation.
     * 
     * Computes the coefficients b, c, and d for the cubic spline polynomial.
     * 
     * @param tridiag_solution The solution vector from the tridiagonal system
     * @param dy Difference in function values
     * @param dx Difference in x values
     * @param index Current index in the data array
     * @param b Pointer to store the computed b coefficient
     * @param c Pointer to store the computed c coefficient
     * @param d Pointer to store the computed d coefficient
     */
    inline void coeff_calc(std::vector<double> &tridiag_solution, double dy, double dx, size_t index, double * b, double * c, double * d);
    
    inline void index_lookup_xy(const int ss_x, const int ss_y, size_t &xi, size_t &yi, const double subpx_x, const double subpx_y) const;

    /**
     * @brief Finds the index of the pixel that contains the given coordinate.
     * 
     * Determines the lower index of the interval containing the specified value.
     * 
     * @param px Vector of pixel coordinates
     * @param x The coordinate to look up
     * @return The index of the pixel containing the coordinate
     */
    inline int index_lookup(const std::vector<double> &px, double x) const;

    /**
     * @brief Initializes the cubic spline coefficients.
     * 
     * Sets up the tridiagonal system and solves it to obtain the cubic spline coefficients.
     * 
     * @param px Vector of x coordinates
     * @param data Vector of function values at the x coordinates
     */
    void cspline_init(const std::vector<double> &px, const std::vector<double> &data,
                      std::vector<double> &local_tridiag_sol);

    /**
     * @brief Evaluates the derivative of a cubic spline at a specified point.
     * 
     * Computes the first derivative of the cubic spline function at the given value.
     * 
     * @param px Vector of x coordinates
     * @param data Vector of function values at the x coordinates
     * @param value The point at which to evaluate the derivative
     * @param length The length of the px and data arrays
     * @return The derivative value at the specified point
     */
    double cspline_eval_deriv(std::vector<double> &px, std::vector<double> &data, 
                              std::vector<double> &local_tridiag_sol, double value, int length);


public:
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dxy;
    std::vector<double> px_y;
    std::vector<double> px_x;
    double *image;
    int px_vert;
    int px_hori;

    /**
     * @brief Initializes the bicubic interpolator with deformed image data.
     * 
     * Sets up the necessary data structures and computes derivatives required for bicubic interpolation.
     * 
     * @param img Pointer to the image data array
     * @param px_hori Width of the image in pixels
     * @param px_vert Height of the image in pixels
     */
    Interpolator(double * img, int px_hori, int px_vert);

    /**
     * @brief Evaluates the bicubic interpolation at a specified point.
     * 
     * Computes the interpolated value at (x,y) using bicubic interpolation from the surrounding pixel values.
     * 
     * @param x The x-coordinate of the interpolation point
     * @param y The y-coordinate of the interpolation point
     * @return The interpolated value at (x,y)
     */
    double eval_bicubic(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const;

    /**
     * @brief Evaluates the x-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to x at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The x-derivative of the interpolated function at (x,y)
     */
    double eval_bicubic_dx(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const;

    /**
     * @brief Evaluates the y-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to y at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The y-derivative of the interpolated function at (x,y)
     */
    double eval_bicubic_dy(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const;

    /**
     * @brief Evaluates the bicubic interpolation and its derivatives at a specified point.
     * 
     * Computes the interpolated value and its partial derivatives at (x,y) in a single call.
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return Data struct containing the interpolated value and its x and y derivatives
     */
    InterpVals eval_bicubic_and_derivs(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const;
};

#endif //DICINTERPOLATOR_H




