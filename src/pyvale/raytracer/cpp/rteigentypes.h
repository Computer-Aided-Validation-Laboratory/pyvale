// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTEIGENCONST_H
#define RTEIGENCONST_H

#include "../../common_cpp/Eigen/Dense"
#include <iostream>
#include "rtelemconstants.h"

// ================================================================================
// Aliases for Eigen types
// ================================================================================

//Notes:
// 1. We cannot use the convenience typedefs like Matrix4d or Vector3d because everything in Eigen is column-major, whereas
// we want it to be compatible with the C-layout (row-major).
//
// 2. Matrix and array are technically the same thing, but with different methods.
// Matrix - Supports linear algebra (cross, dot products, etc.)
// Array - Supports coefficient-wise operations (useful in vectorising where we stack data for multiple mesh elements together)
// You can "flip" the wrapper by using .array() or .matrix(), but both were defined where relevant to keep code neater and avoid unnecessary
// wrapping if we use one variable only for e.g., linear algebra and do not require coefficient-wise operations.

// Dimensions: (Dynamic, Dynamic)
using EiMatrixDd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::StorageOptions::RowMajor>; // Dynamic-size matrix (Dd = dynamic double)
using EiArrayDd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::StorageOptions::RowMajor>; // Dynamic-size array (Dd = dynamic double) for coefficient-wise operations

// Dimensions: (4, 4)
using EiMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::StorageOptions::RowMajor>; 

// 3D vectors (3)
using EiVector3d = Eigen::Matrix<double, 1, 3, Eigen::StorageOptions::RowMajor>; 
using EiArray3d = Eigen::Array<double, 1, 3, Eigen::StorageOptions::RowMajor>;

// 2D vectors (2)
using EiVector2d = Eigen::Matrix<double, 1, 2, Eigen::StorageOptions::RowMajor>; 
using EiArray2d = Eigen::Array<double, 1, 2, Eigen::StorageOptions::RowMajor>; 

// Stacked 3D vectors (Dynamic, 3), where 1 row = 1 3D vector. Mostly used to stack e.g., coordinates (x,y,z) for many mesh elements together for vectorising intersections
using EiVectorD3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::StorageOptions::RowMajor>;
using EiArrayD3d = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::StorageOptions::RowMajor>; 

// Stacked 1D values (Dynamic, 1). Used to vectorise intersection calculations, mostly for dot products where we get single value as a result for each mesh element, and stack them together
// These do not have Eigen::StorageOptions::RowMajor as it causes dimensional errors in Eigen
using EiArrayD1d = Eigen::Array<double, Eigen::Dynamic, 1>; //D1d = (Dynamic, 1), double. 1 row = 1 element of a mesh
using EiBoolMask = Eigen::Array<bool, Eigen::Dynamic, 1>; // Boolean array. Used for boolean masks where 1 row = 1 element of a mesh


// ================================================================================
// Custom Eigen algebra functions (for our custom types)
// ================================================================================

/**
 * @brief Calculates the row-wise cross product for 2 matrices.
 * 
 * Treats each row of the input matrices as a 3D vector and computes the
 * cross product row by row. Also works when one input conceptually represents
 * a repeated row vector, so the input order determines the multiplication order.
 * 
 * @param[in] mat1 (const EiVectorD3d&) First input matrix. Expected shape is (D,3).
 * @param[in] mat2 (const EiVectorD3d&) Second input matrix. Expected shape is (D,3).
 * 
 * @return (EiVectorD3d) Matrix containing the row-wise cross product of the inputs.
 */
inline EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2) {
    // We shouldn't need these checks in principle due to how EiVectorD3d is defined and it never got triggered in tests,
    // so commented out,but keeping it here just in case
    /*
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    if (mat1.rows() != mat2.rows()){
        std::cerr << "Error: matrices need to have the same number of rows to find the cross product" << std::endl;
        return {};
    }*/
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    EiVectorD3d cross_product_result(number_of_rows, 3);
    cross_product_result.col(0) = mat1.col(1).cwiseProduct(mat2.col(2)) - mat1.col(2).cwiseProduct(mat2.col(1));
    cross_product_result.col(1) = mat1.col(2).cwiseProduct(mat2.col(0)) - mat1.col(0).cwiseProduct(mat2.col(2));
    cross_product_result.col(2) = mat1.col(0).cwiseProduct(mat2.col(1)) - mat1.col(1).cwiseProduct(mat2.col(0));
    return cross_product_result;
}

/**
 * @brief Calculates the row-wise cross product for 2 fixed-size 3D row vectors (EiVector3d / Eigen::Vector3d).
 * 
 * Lightweight version of the row-wise cross product specialised for a single 3D vector pair.
 * Should be lighter than the full version for matrices.
 * 
 * @param[in] a (const EiVector3d&) First input 3D vector.
 * @param[in] b (const EiVector3d&) Second input 3D vector.
 * 
 * @return (EiVector3d) Cross product of the 2 input vectors.
 */
static inline EiVector3d cross_rowwise_vec3(const EiVector3d& a, const EiVector3d& b) {
    EiVector3d c;
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
}

/**
 * @brief Calculates the row-wise dot product for 2 stacked 3D arrays.
 * 
 * Treats each row of the input arrays as a 3D vector and computes the
 * dot product row by row.
 * 
 * @param[in] mat1 (const EiArrayD3d&) First input array. Expected shape is (D,3).
 * @param[in] mat2 (const EiArrayD3d&) Second input array. Expected shape is (D,3).
 * 
 * @return (EiArrayD1d) Array containing one dot product result per row.
 */
static inline EiArrayD1d dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2){
    // Eigen should automatically convert EiVectorD3d to EiArrayD3d, so no need to do that while calling the function
    // These change just the object behaviour: arrays are for coefficient-wise operations, matrices for linear algebra. No data copying
    // However, if that breaks, just use e.g., mat1.array()
    return (mat1 * mat2).rowwise().sum();
}

/**
 * @brief Performs vectorised linear interpolation between 2 sets of 3D points.
 * 
 * Computes the interpolation row by row using the expression
 * (1 - weight) * point_A + weight * point_B, 
 * where each weight corresponds to one row of the input point arrays.
 * 
 * @param[in] points_A (const EiArrayD3d&) First set of points. Expected shape is (D,3).
 * @param[in] points_B (const EiArrayD3d&) Second set of points. Expected shape is (D,3).
 * @param[in] weights (const EiArrayD1d) Interpolation weights. Expected shape is (D,1).
 * 
 * @return (EiArrayD3d) Interpolated 3D points. Returns an empty array if input dimensions are incompatible.
 */
inline EiArrayD3d lerp_vectorised (const EiArrayD3d& points_A,
    const EiArrayD3d& points_B,
    const EiArrayD1d weights){

    if (points_A.rows() != points_B.rows() || points_A.rows() != weights.rows()) {
        std::cerr << "Error: points_A, points_B and weights need to have the same number of rows for lerp" << std::endl;
        return {};
    }
    if (points_A.cols() != 3 || points_B.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the interpolation." << std::endl;
        return {};
    }
    // Replicate the (N, 1) weights array to (N, 3) so we can take advantage of Eigen's coefficient-wise array operations
    EiArrayD3d weights_replicated = weights.replicate(1, 3);
    return (1.0 - weights_replicated ) * points_A + weights_replicated * points_B;
}

/**
 * @brief Minimum absolute determinant (|det(M)) threshold for the 3x3 Newton system before the seed is abandoned.
 * 
 * Used to reject nearly singular systems when solving with Cramer's rule.
 */
static const double eps_jac_det = 1e-12;

/**
 * @brief Solves a 3x3 linear system using Cramer's rule.
 * 
 * Solves the system A*x = b and writes the result into x. The solve fails if
 * the determinant magnitude (|det(A)|) is below the threshold defined by eps_jac_det.
 * 
 * Should be a bit faster and more accurate than M.inverse() * b
 * 
 * @param[in] A (const Eigen::Matrix3d&) Coefficient matrix of the linear system.
 * @param[in] b (const Eigen::Vector3d&) Right-hand side vector.
 * @param[out] x (Eigen::Vector3d&) Solution vector.
 * 
 * @return (bool) True if the system was solved successfully and the result is finite, otherwise false.
 */
static inline bool solve_3x3(const Eigen::Matrix3d& A,
                             const Eigen::Vector3d& b,
                             Eigen::Vector3d& x) {

    const double det_A = A.determinant();
    if (std::fabs(det_A) < eps_jac_det) return false;
    const double inv_det = 1.0 / det_A;

    Eigen::Matrix3d Ai;

    Ai = A; Ai.col(0) = b;
    x(0) = Ai.determinant() * inv_det;

    Ai = A; Ai.col(1) = b;
    x(1) = Ai.determinant() * inv_det;

    Ai = A; Ai.col(2) = b;
    x(2) = Ai.determinant() * inv_det;

    return std::isfinite(x(0)) && std::isfinite(x(1)) && std::isfinite(x(2));
}

/**
 * @brief Calculates the diagonal length of the axis-aligned bounding box of an element.
 * 
 * Uses the provided node coordinates to determine the minimum and maximum corners
 * of the axis-aligned bounding box, then returns the diagonal length. This value
 * is used as an element spatial scale for tolerance calculations (mostly residual tolerance)
 * 
 * @param[in] nodes (const EiVector3d*) Pointer to the element node coordinates.
 *      This is a pointer, so we can keep nodes as an array, but reuse this function for different elements.
 * @param[in] element_node_count (const ElementNodeCount) Number of nodes in the element.
 * 
 * @return (double) Diagonal length of the bounding box. Returns 1.0 if the computed length is zero.
 */
static inline double find_element_diagonal(const EiVector3d* nodes,
    const ElementNodeCount element_node_count) {
    EiVector3d low = nodes[0];
    EiVector3d high = nodes[0];
    for (int i = 1; i < element_node_count; ++i) {
        int element_idx = i * NODE_COORDINATES - 1;
        if (nodes[element_idx](0) < low(0)) low(0) = nodes[element_idx](0);
        if (nodes[element_idx](1) < low(1)) low(1) = nodes[element_idx](1);
        if (nodes[element_idx](2) < low(2)) low(2) = nodes[element_idx](2);
        if (nodes[element_idx](0) > high(0)) high(0) = nodes[element_idx](0);
        if (nodes[element_idx](1) > high(1)) high(1) = nodes[element_idx](1);
        if (nodes[element_idx](2) > high(2)) high(2) = nodes[element_idx](2);
    }
    const double d = (high - low).norm();
    return d > 0.0 ? d : 1.0;
}

#endif // RTEIGENCONST_H