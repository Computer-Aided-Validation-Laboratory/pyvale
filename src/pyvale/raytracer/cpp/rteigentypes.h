// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
#include "../../common_cpp/Eigen/Dense"
#include <iostream>
#include "rtelemconstants.h"

// Define aliases for the vectors and matrices from Eigen library AND custom math functions for these types

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


/* ********************************************** 
 * Custom Eigen algebra functions
********************************************** */

inline EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2) {
    // Row-wise cross product for 2 matrices (i.e., treating each row as a vector).
    // Also works for multiplying a matrix with a row vector, so the input order determines the multiplication order. Happy days.
    // Written because this otherwise can't be a one-liner like in NumPy - Eigen's cross product works only for vector types.

    // We shouldn't need this check in principle due to how EiVectorD3d is defined - remove it?
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    // This also should be possible to comment out in production since it is used strictly for intersection code, where these are always created to be the same.
    if (mat1.rows() != mat2.rows()){
        std::cerr << "Error: matrices need to have the same number of rows to find the cross product" << std::endl;
        return {};
    }
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    EiVectorD3d cross_product_result(number_of_rows, 3);
    cross_product_result.col(0) = mat1.col(1).cwiseProduct(mat2.col(2)) - mat1.col(2).cwiseProduct(mat2.col(1));
    cross_product_result.col(1) = mat1.col(2).cwiseProduct(mat2.col(0)) - mat1.col(0).cwiseProduct(mat2.col(2));
    cross_product_result.col(2) = mat1.col(0).cwiseProduct(mat2.col(1)) - mat1.col(1).cwiseProduct(mat2.col(0));
    return cross_product_result;
}

static inline EiVector3d cross_rowwise_vec3(const EiVector3d& a, const EiVector3d& b) {
    // Rowwise cross product specifically for a 3D vector (EiVector3d / Eigen::Vector3d) when we can be certain that this is always the case.
    // Should be lighter than the full version for matrices
    EiVector3d c;
    c(0) = a(1) * b(2) - a(2) * b(1);
    c(1) = a(2) * b(0) - a(0) * b(2);
    c(2) = a(0) * b(1) - a(1) * b(0);
    return c;
}

static inline EiArrayD1d dot_rowwise (const EiArrayD3d& mat1, const EiArrayD3d& mat2){
    // Eigen should automatically convert EiVectorD3d to EiArrayD3d, so no need to do that while calling the function
    // These change just the object behaviour: arrays are for coefficient-wise operations, matrices for linear algebra. No data copying
    // However, if that breaks, just use e.g., mat1.array()
    return (mat1 * mat2).rowwise().sum();
}

inline EiArrayD3d lerp_vectorised (const EiArrayD3d& points_A,
    const EiArrayD3d& points_B,
    const EiArrayD1d weights){
    // Linear interpolation between points stored in arrays points_A and points_B using weights from vector weights.
    // Vectorised version of calculating (1-weight) * point_a + weight * point_b
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

// Minimum |det(M)| for the 3x3 Newton system before we give up on the seed
static const double eps_jac_det = 1e-12;

static inline bool solve_3x3(const Eigen::Matrix3d& A,
                             const Eigen::Vector3d& b,
                             Eigen::Vector3d& x) {
// Solve a 3x3 system A*x = b using Cramer's rule. Returns false if |det(A)| below threshold
// Should be a bit faster and more accurate than M.inverse() * b
                            
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


static inline double find_element_diagonal(const EiVector3d* nodes, const ElementNodeCount element_node_count) {
    // Element spatial scale: diagonal of the axis-aligned bounding box of the node set. Used to scale the residual tolerance.
    EiVector3d low = nodes[0];
    EiVector3d high = nodes[0];
    for (int i = 1; i < element_node_count; ++i) {
        int element_idx = i * NODE_COORDINATES - 1;
        if (nodes[element_idx](0) < low(0)) low(0) = nodes[i](0);
        if (nodes[element_idx](1) < low(1)) low(1) = nodes[i](1);
        if (nodes[element_idx](2) < low(2)) low(2) = nodes[i](2);
        if (nodes[element_idx](0) > high(0)) high(0) = nodes[i](0);
        if (nodes[element_idx](1) > high(1)) high(1) = nodes[i](1);
        if (nodes[element_idx](2) > high(2)) high(2) = nodes[i](2);
    }
    const double d = (high - low).norm();
    return d > 0.0 ? d : 1.0;
}