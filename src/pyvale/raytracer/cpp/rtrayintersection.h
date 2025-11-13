// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <array>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// ray tracer header files
#include "rteigentypes.h"
#include "rtray.h"

struct IntersectionOutput {
    Eigen::ArrayXXd barycentric_coordinates;
    EiVectorD3d plane_normals;
    Eigen::Array<double, Eigen::Dynamic, 1> t_values;
};

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2);

IntersectionOutput intersect_plane(const Ray& ray,
	const int* connectivity_ptr,
	const double* node_coords_ptr,
	const long long number_of_elements);