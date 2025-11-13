// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// C++ header files
#include <Windows.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdint>
#include <iostream>
#include <array>
#include <vector>

// common_cpp header files
#include "../../common_cpp/Eigen/Dense"

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"


void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::vector<py::array_t<int>>& scene_connectivity,
    const std::vector<py::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors,
    const std::vector<Eigen::Ref<const EiVector3d>> camera_centers,
    const std::vector<Eigen::Ref<const EiVector3d>> pixel_00_centers,
    const std::vector<Eigen::Ref<const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings);
