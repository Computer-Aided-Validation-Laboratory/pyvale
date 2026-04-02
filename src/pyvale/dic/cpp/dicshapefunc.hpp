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




/**
 * @brief Affine shape function for DIC subset deformation.
 *        Models translation, rotation, shear and normal strain (6 parameters).
 *
 *        Coordinate convention: (x, y) are in the local subset frame, relative
 *        to the subset's top-left corner. Shape function parameters describe
 *        deformation anchored at the subset centre, so get_displacement() should
 *        be evaluated at (cx - global_x, cy - global_y) to recover u and v at
 *        the centre.
 *
 *        p = [u, v, du/dx, du/dy, dv/dx, dv/dy]
 */
struct Affine {

    /**
     * @brief Maps a reference pixel to its deformed position.
     * @param[out] x_new  Deformed x-coordinate in local subset frame [pixels]
     * @param[out] y_new  Deformed y-coordinate in local subset frame [pixels]
     * @param[in]  x      Reference x-coordinate in local subset frame [pixels]
     * @param[in]  y      Reference y-coordinate in local subset frame [pixels]
     * @param[in]  p      Shape function parameters [u, v, du/dx, du/dy, dv/dx, dv/dy]
     */
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);

    /**
     * @brief Computes the Jacobian row df/dp for this pixel, used to build the
     *        Hessian and gradient in the Levenberg-Marquardt optimisation.
     * @param[out] dfdp  Jacobian entries (6 elements): [dfdx, dfdy, dfdx*x, dfdx*y, dfdy*x, dfdy*y]
     * @param[in]  x     Reference x-coordinate in local subset frame [pixels]
     * @param[in]  y     Reference y-coordinate in local subset frame [pixels]
     * @param[in]  dfdx  Image gradient in x at this pixel
     * @param[in]  dfdy  Image gradient in y at this pixel
     */
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);

    /**
     * @brief Computes displacement (u, v) at a point in the local subset frame.
     *        To recover the displacement at the subset centre, pass
     *        x = cx - global_x, y = cy - global_y.
     * @param[out] u  Displacement in x-direction [pixels]
     * @param[out] v  Displacement in y-direction [pixels]
     * @param[in]  x  x-coordinate in local subset frame [pixels]
     * @param[in]  y  y-coordinate in local subset frame [pixels]
     * @param[in]  p  Shape function parameters [u, v, du/dx, du/dy, dv/dx, dv/dy]
     */
    static void get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p);

    /**
    * @brief Composes two Affine transforms to produce a single equivalent transform.
    *        If pA maps shape0->shape1 and pB maps shape1->shape2,
    *        then pC maps shape0->shape2.
    * @param[out] pC  Composed shape function parameters (6 elements)
    * @param[in]  pA  Shape function parameters for shape0->shape1 (6 elements)
    * @param[in]  pB  Shape function parameters for shape1->shape2 (6 elements)
    */
    static void compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB);

    static constexpr int num_params = 6; /**< Number of shape function parameters */
};

/**
 * @brief Quadratic shape function for DIC subset deformation.
 *        Extends affine with second-order terms, capturing bending and
 *        non-uniform strain fields (12 parameters).
 *
 *        Coordinate convention: same as Affine — (x, y) are in the local subset
 *        frame, relative to the subset's top-left corner.
 *
 *        p = [u, v, du/dx, du/dy, dv/dx, dv/dy,
 *             d2u/dx2, d2u/dxdy, d2u/dy2, d2v/dx2, d2v/dxdy, d2v/dy2]
 */
struct Quad {

    /**
     * @brief Maps a reference pixel to its deformed position.
     * @param[out] x_new  Deformed x-coordinate in local subset frame [pixels]
     * @param[out] y_new  Deformed y-coordinate in local subset frame [pixels]
     * @param[in]  x      Reference x-coordinate in local subset frame [pixels]
     * @param[in]  y      Reference y-coordinate in local subset frame [pixels]
     * @param[in]  p      Shape function parameters (12 total, see struct description)
     */
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);

    /**
     * @brief Computes the Jacobian row df/dp for this pixel.
     * @param[out] dfdp  Jacobian entries (12 elements):
     *                   [dfdx, dfdy, dfdx*x, dfdx*y, dfdy*x, dfdy*y,
     *                    dfdx*x^2, dfdx*x*y, dfdx*y^2, dfdy*x^2, dfdy*x*y, dfdy*y^2]
     * @param[in]  x     Reference x-coordinate in local subset frame [pixels]
     * @param[in]  y     Reference y-coordinate in local subset frame [pixels]
     * @param[in]  dfdx  Image gradient in x at this pixel
     * @param[in]  dfdy  Image gradient in y at this pixel
     */
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);

    /**
     * @brief Computes displacement (u, v) at a point in the local subset frame.
     *        To recover displacement at the subset centre, pass
     *        x = cx - global_x, y = cy - global_y.
     * @param[out] u  Displacement in x-direction [pixels]
     * @param[out] v  Displacement in y-direction [pixels]
     * @param[in]  x  x-coordinate in local subset frame [pixels]
     * @param[in]  y  y-coordinate in local subset frame [pixels]
     * @param[in]  p  Shape function parameters (12 total)
     */
    static void get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p);

    /**
    * @brief Composes two Quadratic transforms to produce a single equivalent transform.
    *        If pA maps shape0->shape1 and pB maps shape1->shape2,
    *        then pC maps shape0->shape2.
    *        Note: composing two quadratic maps produces terms beyond second order;
    *        these are truncated so the result remains a valid Quad parameter set.
    * @param[out] pC  Composed shape function parameters (12 elements)
    * @param[in]  pA  Shape function parameters for shape0->shape1 (12 elements)
    * @param[in]  pB  Shape function parameters for shape1->shape2 (12 elements)
    */
    static void compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB);

    static constexpr int num_params = 12; /**< Number of shape function parameters */
};

/**
 * @brief Rigid (translation-only) shape function for DIC subset deformation.
 *        Models pure translation with no rotation or strain (2 parameters).
 *        Displacement is constant across the subset — (x, y) do not affect the result.
 *
 *        Coordinate convention: same as Affine — (x, y) are in the local subset
 *        frame, relative to the subset's top-left corner.
 *
 *        p = [u, v]
 */
struct Rigid {
    /**
     * @brief Maps a reference pixel to its deformed position (pure translation).
     * @param[out] x_new  Deformed x-coordinate in local subset frame [pixels]
     * @param[out] y_new  Deformed y-coordinate in local subset frame [pixels]
     * @param[in]  x      Reference x-coordinate in local subset frame [pixels]
     * @param[in]  y      Reference y-coordinate in local subset frame [pixels]
     * @param[in]  p      Shape function parameters [u, v]
     */
    static void get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p);

    /**
     * @brief Computes the Jacobian row df/dp for this pixel.
     *        For rigid motion x and y do not contribute, so dfdp has only 2 elements.
     * @param[out] dfdp  Jacobian entries (2 elements): [dfdx, dfdy]
     * @param[in]  x     Reference x-coordinate in local subset frame [pixels] (unused)
     * @param[in]  y     Reference y-coordinate in local subset frame [pixels] (unused)
     * @param[in]  dfdx  Image gradient in x at this pixel
     * @param[in]  dfdy  Image gradient in y at this pixel
     */
    static void get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy);

    /**
     * @brief Computes displacement (u, v). For rigid motion displacement is constant
     *        across the subset, so (x, y) are unused.
     * @param[out] u  Displacement in x-direction [pixels], equal to p[0]
     * @param[out] v  Displacement in y-direction [pixels], equal to p[1]
     * @param[in]  x  x-coordinate in local subset frame [pixels] (unused)
     * @param[in]  y  y-coordinate in local subset frame [pixels] (unused)
     * @param[in]  p  Shape function parameters [u, v]
     */
    static void get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p);

    /**
    * @brief Composes two Rigid transforms to produce a single equivalent transform.
    *        If pA maps shape0->shape1 and pB maps shape1->shape2,
    *        then pC maps shape0->shape2.
    * @param[out] pC  Composed shape function parameters (2 elements)
    * @param[in]  pA  Shape function parameters for shape0->shape1 (2 elements)
    * @param[in]  pB  Shape function parameters for shape1->shape2 (2 elements)
    */
    static void compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB);

    static constexpr int num_params = 2; /**< Number of shape function parameters */
};

#endif // DICSHAPEFUNC_HPP
