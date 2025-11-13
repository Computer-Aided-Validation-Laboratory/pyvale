#pragma once

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include<vector>

#include "rteigentypes.h"
#include "rtcamera.h"
#include "rtray.h"

inline EiVector3d get_color(Eigen::Index minRowIndex,
    const double* face_color_ptr);

EiVector3d return_ray_color(const Ray& ray,
    const std::vector < pybind11::array_t<int>>& scene_connectivity,
    const std::vector < pybind11::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors);

void render_ppm_image(const Eigen::Ref<const EiVector3d>& camera_center,
    const Eigen::Ref<const EiVector3d>& pixel_00_center,
    const Eigen::Ref<const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>& matrix_pixel_spacing,
    const std::vector < pybind11::array_t<int>>& scene_connectivity,
    const std::vector < pybind11::array_t<double>>& scene_coords,
    const std::vector<pybind11::array_t<double>>& scene_face_colors,
    const int image_height,
    const int image_width,
    const int number_of_samples);