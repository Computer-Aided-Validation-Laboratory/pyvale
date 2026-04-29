// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files 
#include <array>
#include <string>
#include <vector>
#include <filesystem>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"

EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS);

void render_ppm_image(const EiVector3d &camera_center,
    const EiVector3d &pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> &matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_filepath);