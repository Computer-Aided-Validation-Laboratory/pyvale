// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTMAIN_H
#define RTMAIN_H

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

/**
 * @brief Receives scene, image, and rendering data from Python.
 * Dispatches to build acceleration structures, then iterates over cameras to generate images.
 * 
 * Parameter list to be updated:
 * a) We will likely be sending much more data, so this is to be changed anyway
 * b) This interface needs to be made C-compatible anyway, so datatypes will likely change
 * c) It would be nice to compact it into something like a struct, because it is lengthy and becoming hard to keep track of
 * 
 */
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
    const std::vector<double>& scene_refractive_indices,
    const std::vector<int>& scene_mesh_priorities,
    const std::vector<int>& scene_mesh_object_types,
    const std::vector<double>& scene_mesh_thickness,
    const nb::DRef<EiVector3d>& background_color,
    const int texture_sampler,
    const int shading_type,
    const int output_format,
    const int max_depth,
    const bool grayscale_flag);

#endif // RTMAIN_H