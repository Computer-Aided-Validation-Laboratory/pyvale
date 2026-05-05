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

// Struct to store ray data in the stack-based shader
struct RayState{
    Ray ray;
    EiVector3d accumulated_color; // Accumulated multipliers (albedo, Fresnel terms, etc.)
    int depth;
};

static inline EiVector3d ray_blue_sky(const Ray& ray);

inline void ray_diffuse(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack);

inline void ray_specular(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack);

inline void ray_refractive(const RayState& current_state,
    const HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack);

EiVector3d return_ray_color_stack(const Ray& primary_ray, const TLAS& TLAS);

EiVector3d return_ray_color_new(const Ray& ray,
    const TLAS& TLAS,
    int depth = 0);

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