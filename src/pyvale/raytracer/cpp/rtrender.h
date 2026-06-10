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

// ================================================================================
// Enums for output render configuration
// ================================================================================

enum class RenderColor{
    COLOR = 0,
    GRAYSCALE = 1
};

enum class OutputType{
    PPM = 0,
    TIFF = 1
    //NP_BUFFER = 2 // Not available yet
};

// ================================================================================
// Outputwriter - writing the image pixel buffer to the chosen format
// ================================================================================

// TO DO: add NumPy array buffer

namespace outputwriter{

    // Pointer to the selected function
    extern void (*save_image)(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    /**
     * @brief  Helper that writes 16-bit integers in Little-Endian byte order
     */
    static inline void write_16bit(std::ofstream& image_file,
        uint16_t value);
    
    /**
     * @brief  Helper that writes 16-bit integers in Little-Endian byte order
     */
    static inline void write_32bit(std::ofstream& image_file,
        uint32_t value);
    
    /**
     * @brief Helper that writes a TIFF tag (12-byte IFD tag)
     */
    static inline void write_tag(std::ofstream& image_file,
        uint16_t tag,
        uint16_t type,
        uint32_t count,
        uint32_t value);

    /**
     * @brief Saves the stored pixel buffer in TIFF format.
     * 
     * Adds ".tiff" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */

    void saveTIFF(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    /**
     * @brief Saves the stored pixel buffer in PPM format.
     * 
     * Adds ".ppm" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    void savePPM(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    /**
     * @brief Setter for the appropriate output writing function based on the passed configuration.
     */
    void set(OutputType output_format);
}

// ================================================================================
// Return ray color 
// ================================================================================

/**
 * @brief Processes a single primary primary ray to find its corresponding pixel colour via iterative tracking of secondary rays.
 * 
 * Creates a thread-safe stack of RayState objects, then dispatches to intersect_TLAS to find the nearest intersection.
 * Adds a blue sky colour if there is no intersection. Otherwise, it checks and applies material absorption,
 * and verifies the InteriorList for nested dielectrics where applicable, to evaluate if the hit is true or false.
 * Finally, dispatches to the appropriate material colour and adds the output iteratively to the stack.
 * The stack is traversed until empty, terminated early due to Russian rulette, or the MAX_DEPTH is reached.
 * 
 * @param[in] primary_ray (const Ray&) The primary Ray with direction and origin determined by its corresponing pixel in render_image.
 * @param[in] scene_ri (const double) Refractive index of the scene (ambient medium) which is used as a fallback value in shading.
 * @param[in] TLAS (const &TLAS) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
 * 
 * @return (EiVector3d) A 3D, row-major Eigen vector storing the final colour of the pixel in the (r,g,b) format.
 * 
 */
EiVector3d return_ray_color_stack(const Ray& primary_ray,
    const double scene_ri,
    const TLAS& TLAS);

// ================================================================================
// render_image template for colour and grayscale
// ================================================================================

/**
 * @brief Iterates over each pixel in the viewport to shoot rays and retrieve their colours.
 * 
 * This is template-based to avoid having to branch etc. based on whether the output image is in grayscale or not.
 * Creates a buffer of pixels, then goes over the image height and width to determine the ray direction and origin
 * for each pixel. Dispatches the ray to return_ray_color_stack and retrieves the final colour value, which is then
 * averaged over n samples if anti-aliasing is on.
 * Finally, it clamps the colour between [0,1] and either converts it to grayscale or RGB, then stores it in the buffer, which
 * is dispatched to an appropriate output writer.
 * 
 * @param[in] camera_center (const EiVector3d&) Row-major 3D Eigen vector with the [x,y,z] coordinates of the chosen camera.
 * @param[in] pixel_00_center (const EiVector3d&) Row-major 3D Eigen vector with the [x,y,z] coordinates of the (0,0) (upper left) pixel of the viewport corresponding to the passed camera.
 * @param[in] matrix_pixel_spacing (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the vectors defining the horizontal and vertical spacing of the pixels in the viewport. They are defined towards the right, and downward.
 * @param[in] matrix_defocus_disc (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the horizontal and vertical defocus dics basis vectors for the thin lens approximation; it will be full of zeros if not DoF is not used.
 * @param[in] TLAS (const TLAS&) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
 * @param[in] image_height (const int) Output image height
 * @param[in] image_width (const int) Output image width
 * @param[in] number_of_samples (const int) Number of samples used for anti-aliasing
 * @param[in] scene_ri (const double) Refractive index of the scene (ambient medium) which is used as a fallback value in shading.
 * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
 *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
 */
template <RenderColor color>
void render_image(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    std::filesystem::path& output_filepath) {

    std::vector<uint8_t> buffer;
    buffer.resize(image_width * image_height * 3); // Preallocate memory for the image buffer

    // Pull invariants out of the loops to avoid re-computing these
    const EiVector3d pixel_row_0 = matrix_pixel_spacing.row(0);
    const EiVector3d pixel_row_1 = matrix_pixel_spacing.row(1);
    const EiVector3d defocus_row_0 = matrix_defocus_disc.row(0);
    const EiVector3d defocus_row_1 = matrix_defocus_disc.row(1);
    static const double color_scaling = 1.0 /number_of_samples; // Multiplication is faster than division, so we pre-divide it before looping

    #pragma omp parallel for schedule(dynamic) 
    for (int j = 0; j < image_height; j++) {
        //std::cerr << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush << std::endl;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * pixel_row_0 + (j + offset[1]) * pixel_row_1;
                    // Below is true for pinhole camera
                    //EiVector3d ray_origin = camera_center;
                    //EiVector3d ray_direction = pixel_sample - camera_center;

                    // Thin lens approximation camera
                    std::array<double, 2> defocus_disc_offset = point_in_unit_disk();
                    EiVector3d defocus_disc_sample = defocus_disc_offset[0] * defocus_row_0 + defocus_disc_offset[1] * defocus_row_1;
                    EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
                    EiVector3d ray_direction = pixel_sample - ray_origin; // ray direction in thin lens approx
                    Ray current_ray{ ray_origin, ray_direction.stableNormalized() }; 

                    //Clamp fireflies - optional, makes images less bright
                    //EiVector3d sample = return_ray_color_stack(current_ray, scene_ri, TLAS);
                    //double lum = 0.2126*sample.x() + 0.7152*sample.y() + 0.0722*sample.z();
                    //static constexpr double MAX_LUM = 10.0; // Tune per scene; hoist this out of the loop if using 
                    //if (lum > MAX_LUM) sample *= MAX_LUM / lum;
                    //pixel_color += sample;
                    pixel_color += return_ray_color_stack(current_ray, scene_ri, TLAS);
            
        }
            
            int px_idx = (i + j * image_width) * 3;
            // Divide by the number of samples to get the mean colour
            pixel_color = pixel_color * color_scaling;
            if constexpr (color == RenderColor::GRAYSCALE) {
                // Convert to a single-channel grayscale
                pixel_color[0] = std::clamp(pixel_color[0], 0.0, 0.999);
                double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
                // Clamp to the [0,1] range
                // Scale to bytes
                uint8_t gray_byte = static_cast<uint8_t>(pixel_color[0] * 255.999);
                buffer[px_idx] = gray_byte;
                buffer[px_idx + 1] = gray_byte;
                buffer[px_idx + 2] = gray_byte;
            }
            else if constexpr (color == RenderColor::COLOR) {
                // Clamp each channel to [0,1]
                pixel_color = pixel_color.cwiseMax(0.0).cwiseMin(1.0);
                // Scale to bytes
                pixel_color *= 255.999;
                buffer[px_idx] = static_cast<uint8_t>(pixel_color.x());
                buffer[px_idx + 1] = static_cast<uint8_t>(pixel_color.y());
                buffer[px_idx + 2] = static_cast<uint8_t>(pixel_color.z());
            }
        }
    }

    // Write the buffer in whatever output format we want
    outputwriter::save_image(buffer, image_height, image_width, output_filepath);
};

// ================================================================================
// Mock ray shooter for debug
// ================================================================================

/**
 * @brief Quick debug function that shoots a single ray into TLAS so it can be tracked, either to
 * compare with analytical solution or troubleshoot certain bugs without having to go through the entire image.
 * Note: The ray needs to be hard-coded into this function.
 */
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