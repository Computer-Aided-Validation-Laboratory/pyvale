// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
#include "../../common_cpp/Eigen/Dense"

// Define aliases for the vectors and matrices from Eigen library.

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