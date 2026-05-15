// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>

// C++ header files
#include <cmath>
#include <filesystem>
#include <iostream>
#include <array>
#include <vector>
#include <chrono>

// common_cpp header files
#include "../../common_cpp/Eigen/Dense"

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"
#include "rtbvh.h"

namespace nb = nanobind;

void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_defocus_discs,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_normals_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_uvs,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_textures,
    const std::vector<int>& scene_surface_types,
    const std::vector<int>& materials,
    const std::vector<double>& refractive_indices,
    const int texture_sampler);

    /*
    // Version used if all meshes in the scene have solid colour surface
    void render_scene_color(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors);

 // Version used if all meshes in the scene have textured
    void render_scene_texture(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_uvs,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_textures);
    */