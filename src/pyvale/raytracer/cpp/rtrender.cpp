// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

// raytracer header files
#include "rtrender.h"
#include "rthitrecord.h"
#include "rtrayintersection.h"
#include "rtmathutils.h"

EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS) {

    HitRecord intersection_record; // Create HitRecord struct
    // Look for intersection
    IntersectionOutput intersection;
    intersect_TLAS(ray, TLAS, intersection, intersection_record);

    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        //std::cout << "Coloring..." << std::endl;
        set_face_normal(ray, intersection_record.normal_surface);
        return intersection_record.barycentric_coordinates(0) * intersection_record.face_color + intersection_record.barycentric_coordinates(1) * intersection_record.face_color + intersection_record.barycentric_coordinates(2) * intersection_record.face_color;
    }

    // Blue sky gradient
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}

void render_ppm_image(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_filepath) {
    // Get camera parameters from the dict and cast it to Eigen types so it works with existing code; by reference to avoid copying data

    std::vector<uint8_t> buffer;
    buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)

    for (int j = 0; j < image_height; j++) {
        //std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush << std::endl;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * matrix_pixel_spacing.row(0) +
                    (j + offset[1]) * matrix_pixel_spacing.row(1);
                EiVector3d ray_direction = pixel_sample - camera_center;
                Ray current_ray{ camera_center, ray_direction.normalized() };
                pixel_color += return_ray_color(current_ray, TLAS);
            }
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray / number_of_samples * 255.99);
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
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
}