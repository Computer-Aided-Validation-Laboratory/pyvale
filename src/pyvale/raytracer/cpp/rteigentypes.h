// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
#include "../../common_cpp/Eigen/Dense"

// Define aliases for the vectors and matrices from Eigen library.
// Cannot use the convenience typedefs like Matrix4d or Vector3d because everything in Eigen is column-major, whereas
// we want it to be compatible with the C-layout (row-major)
using EiMatrix4d = Eigen::Matrix<double, 4, 4, Eigen::StorageOptions::RowMajor>; // Shape (4,4)
using EiVector3d = Eigen::Matrix<double, 1, 3, Eigen::StorageOptions::RowMajor>; // Vector; shape (3)
using EiMatrixDd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Dynamic-size matrix (Dd = dynamic double)
using EiArrayDd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // Dynamic-size array (Dd = dynamic double) for coefficient-wise operations
using EiVectorD3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Matrix shaped (D,3); mostly for coordinates to avoid having to loop constantly in the intersection code to get cross products etc. Think coordinates stacked together
using EiArrayD3d = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>; // Same as VectorD3d, just an array for coefficient-wise operations