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

namespace nb = nanobind;

// struct TextureProjector {
//     double* texture_img_ptr;
//     int img_w;
//     int img_h;
//     EiVector3d proj_pos; // projector position
//     EiVector3d proj_dir; // projector direction
//     double proj_fov; // degrees

//     inline double apply_texture(double u, double v) {
    
//         int x = std::min(int(u * img_w), img_w - 1);
//         int y = std::min(int(v * img_h), img_h - 1);
    
//         int idx = y * img_w + x;

//         return texture_img_ptr[idx];
//     }
// };


struct TextureProjector {
    const double* texture_img_ptr;
    int img_w;
    int img_h;
    EiVector3d proj_pos;
    EiVector3d proj_dir;
    double proj_fov;

    inline double apply_texture(double u, double v) {

        // bounds check
        if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0)
            return 0.0;

        // flip vertically
        v = 1.0 - v;

        int x = std::min(int(u * img_w), img_w - 1);
        int y = std::min(int(v * img_h), img_h - 1);

        int idx = y * img_w + x;

        return texture_img_ptr[idx];
    }
};



EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS);

void render_ppm_image(const EiVector3d &camera_center,
    const EiVector3d &pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> &matrix_pixel_spacing,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_filepath,
    nb::ndarray<const double, nb::c_contig>& texture_img);