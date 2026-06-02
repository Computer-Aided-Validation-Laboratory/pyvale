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
#include <fstream>
#include <iostream>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"
#include "rtmathutils.h"

enum class RenderColor{
    COLOR = 0,
    GRAYSCALE = 1
};


EiVector3d return_ray_color_stack(const Ray& primary_ray, const double scene_ri, const TLAS& TLAS);

/*
EiVector3d return_ray_color_new(const Ray& ray,
    const TLAS& TLAS,
    int depth = 0);
*/

// Templated based on coloring mode to avoid having to branch etc.
template <RenderColor color>
void render_ppm_image(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath) {

    std::vector<uint8_t> buffer;
    //buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)
    buffer.resize(image_width * image_height * 3); // Preallocate memory for the image buffer

    // Pull invariants out of the loops to avoid re-computing these
    const EiVector3d pixel_row_0 = matrix_pixel_spacing.row(0);
    const EiVector3d pixel_row_1 = matrix_pixel_spacing.row(1);
    const EiVector3d defocus_row_0 = matrix_defocus_disc.row(0);
    const EiVector3d defocus_row_1 = matrix_defocus_disc.row(1);
    static const double color_scaling = 1/number_of_samples * 255.99; // Multiplication is faster than division, so we pre-divide it before looping

    #pragma omp parallel for schedule(dynamic) 
    for (int j = 0; j < image_height; j++) {
        //std::cerr << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush << std::endl;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * pixel_row_0 +
                    (j + offset[1]) * pixel_row_1;
                    //(i + offset[0]) * matrix_pixel_spacing.row(0) +
                    //(j + offset[1]) * matrix_pixel_spacing.row(1);
                    // Below is true for pinhole camera
                    //EiVector3d ray_origin = camera_center;
                    //EiVector3d ray_direction = pixel_sample - camera_center;
                    //Ray current_ray{ ray_origin, ray_direction };
                    // Thin lens approximation camera
                    std::array<double, 2> defocus_disc_offset = point_in_unit_disk();
                    //EiVector3d defocus_disc_sample = defocus_disc_offset[0] * matrix_defocus_disc.row(0) + defocus_disc_offset[1] * matrix_defocus_disc.row(1);
                    EiVector3d defocus_disc_sample = defocus_disc_offset[0] * defocus_row_0 + defocus_disc_offset[1] * defocus_row_1;
                    EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
                    EiVector3d ray_direction = pixel_sample - ray_origin; // ray direction in thin lens approx
                    Ray current_ray{ ray_origin, ray_direction.stableNormalized() }; 
                    pixel_color += return_ray_color_stack(current_ray, scene_ri, TLAS);
            
        }
            // Get the RGB components of the pixel color (in [0,1] range) and convert them to a single-channel grayscale
            //std::clamp(pixel_color[0], 0.0, 0.999);
            int px_idx = (i + j * image_width) * 3;
            if constexpr (color == RenderColor::GRAYSCALE) {
                double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
                //int gray_byte = int(gray / number_of_samples * 255.99);
                int gray_byte = gray * color_scaling;
                buffer[px_idx] = static_cast<uint8_t>(gray_byte);
                buffer[px_idx + 1] = static_cast<uint8_t>(gray_byte);
                buffer[px_idx + 2] = static_cast<uint8_t>(gray_byte);
                // Below is used if we reserve the buffer
                //buffer.push_back(static_cast<uint8_t>(gray_byte));
                //buffer.push_back(static_cast<uint8_t>(gray_byte));
                //buffer.push_back(static_cast<uint8_t>(gray_byte));
            }
            else if constexpr (color == RenderColor::COLOR) {
                //pixel_color /= number_of_samples;
                //pixel_color *= 255.999;
                pixel_color = pixel_color * color_scaling;
                buffer[px_idx] = static_cast<uint8_t>(pixel_color.x());
                buffer[px_idx + 1] = static_cast<uint8_t>(pixel_color.y());
                buffer[px_idx + 2] = static_cast<uint8_t>(pixel_color.z());
                // Below is used if we reserve the buffer
                //buffer.push_back(static_cast<uint8_t>(pixel_color.x()));
                //buffer.push_back(static_cast<uint8_t>(pixel_color.y()));
                //buffer.push_back(static_cast<uint8_t>(pixel_color.z()));
            }
        }
    }

    std::ofstream image_file;

    image_file.open(output_filepath);
    if (!image_file.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return;
    }

    image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
    image_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    image_file.close();
    std::cout << "\r Done. \n";
};

void mock_ray_shoot(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath);