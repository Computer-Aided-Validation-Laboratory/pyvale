// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================




#ifndef DICINTERPBSPLINE_H
#define DICINTERPBSPLINE_H


// STD library Header files
#include <vector>
#include <cmath>

// Program Header files
#include "./dicinterp.hpp"


class Bspline : public Interpolator {

private:

    std::vector<double> coeff;
    double *image;

    // Recursive spline prefilter
    void prefilter_x();
    void prefilter_y();

    // 1D cubic B-spline basis and derivatives
    static inline void basis(double t, double B[4]);
    static inline void basis_d(double t, double Bd[4]);

    std::vector<double> coeff_padded;
    int padded_hori;      // width of padded array (px_hori + 4)
    int padded_vert;      // height of padded array (px_vert + 4)

public:

    /**
     * @brief Initializes the bicubic interpolator with deformed image data.
     * 
     * Sets up the necessary data structures and computes derivatives required for bicubic interpolation.
     * 
     * @param img Pointer to the image data array
     * @param px_hori Width of the image in pixels
     * @param px_vert Height of the image in pixels
     */
    Bspline(double * img, int px_hori, int px_vert);

    /**
     * @brief Evaluates the bicubic interpolation at a specified point.
     * 
     * Computes the interpolated value at (x,y) using bicubic interpolation from the surrounding pixel values.
     * 
     * @param x The x-coordinate of the interpolation point
     * @param y The y-coordinate of the interpolation point
     * @return The interpolated value at (x,y)
     */
    double eval(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const override;

    /**
     * @brief Evaluates the x-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to x at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The x-derivative of the interpolated function at (x,y)
     */
    double eval_dx(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const override;

    /**
     * @brief Evaluates the y-derivative of bicubic interpolation at a specified point.
     * 
     * Computes the partial derivative with respect to y at point (x,y).
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return The y-derivative of the interpolated function at (x,y)
     */
    double eval_dy(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const override;

    /**
     * @brief Evaluates the bicubic interpolation and its derivatives at a specified point.
     * 
     * Computes the interpolated value and its partial derivatives at (x,y) in a single call.
     * 
     * @param x The x-coordinate of the point
     * @param y The y-coordinate of the point
     * @return Data struct containing the interpolated value and its x and y derivatives
     */
    InterpVals eval_and_derivs(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const override;

};

#endif //DICINTERPBSPLINE_H
