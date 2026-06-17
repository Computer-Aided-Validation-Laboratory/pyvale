// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTRENDER_H
#define RTRENDER_H

// STD header files 
#include <array>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <csignal>

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
#include "rtsignal.h"
#include "rtiowriter.h"
#include "rtsobolsampler.h"

// commmon header files
#include "../../common_cpp/progressbar.hpp"
//#include "../../common_cpp/dicsignalhandler.hpp" in the future

// ================================================================================
// Enums for output render configuration - must match Python API
// ================================================================================

enum class RenderColor{
    COLOR = 0,
    GRAYSCALE = 1
};

/// @brief Specifies buffer type for render_image based on the desired bit depth.
enum class BufferType{
    UINT_8 = 0,
    UINT_16 = 1
};

enum class CameraType{
    PINHOLE = 0,
    THIN_LENS = 1
};

// ================================================================================
// Return ray color 
// ================================================================================
namespace renderer{
    /// @brief Maximum depth for the secondary rays.
    extern int MAX_DEPTH;
    /// @brief Background colour for the scene.
    extern EiVector3d background_color;
    /// @brief Maximum integer range used to multiply pixel colour values to get the desired bit depth.
    extern uint32_t max_code_range;
    // Alias for the renderer function pointer
    using RenderingFunction = void (*) (const EiVector3d& camera_center,
        const EiVector3d& pixel_00_center,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
        const TLAS& TLAS,
        const int image_height,
        const int image_width,
        const int number_of_samples,
        std::filesystem::path& output_filepath);

    extern RenderingFunction render_image;

    /**
     * @brief Returns a procedural sky color for a ray direction.
     * 
     * Produces a simple vertical white-to-blue gradient based on the y-component
     * of the ray direction.
     * 
     * Note: This has been used for a very long time as the default background (stemming
     * from Ray Tracing in One Weekend), but realistically, we probably do not need that
     * for a virtual laboratory.
     * Replaced with a solid background colour; feel free to remove it, or let users 
     * pick between this and a solid background, etc.
     * 
     * @param[in] ray (const Ray&) Input ray
     * 
     * @return (EiVector3d) RGB sky color corresponding to the ray direction.
     */
    inline EiVector3d ray_blue_sky(const Ray& ray){
        double a = 0.5 * (ray.direction(1) + 1.0);
        static EiVector3d white, blue;
        white << 1.0, 1.0, 1.0;
        blue << 0.5, 0.7, 1.0;
        return (1.0 - a) * white + a * blue;
    }

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
     * @param[in] TLAS (const &TLAS) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
     * 
     * @return (EiVector3d) A 3D, row-major Eigen vector storing the final colour of the pixel in the (r,g,b) format.
     * 
     */
    EiVector3d return_ray_color_stack(const Ray& primary_ray,
        const TLAS& TLAS,
        const SobolSampler& sampler);      // [SOBOL] Comment out to test MT19937

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
     * @param[in] matrix_pixel_spacing (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the vectors defining the horizontal and vertical spacing of the pixels in the viewport.
     *          They are defined towards the right, and downward.
     * @param[in] matrix_defocus_disc (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the horizontal and vertical defocus dics basis vectors for the thin lens approximation;
     *          it will be full of zeros if not DoF is not used.
     * @param[in] TLAS (const TLAS&) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] number_of_samples (const int) Number of samples used for anti-aliasing
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    template <RenderColor color, BufferType buffer_type, CameraType camera_type>
    void render_img(const EiVector3d& camera_center,
        const EiVector3d& pixel_00_center,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
        const TLAS& TLAS,
        const int image_height,
        const int image_width,
        const int number_of_samples,
        std::filesystem::path& output_filepath) {

        // DEV NOTES:
        // Forgive me father for I have sinned, but that was the only neat solution I could find for this that is
        // a) Compile time
        // b) Not requiring very many metatemplates (example here: https://www.reddit.com/r/cpp_questions/comments/1dtr4vo/how_do_create_a_conditional_type_based_on/)
        // c) Or rewriting this function twice separately for 8- and 16-bits
        // Note if we ever add more buffer types, however, we'd either have to stack these or do something else
        using BufferDType = std::conditional_t<buffer_type == BufferType::UINT_8, uint8_t, uint16_t>;

        std::vector<BufferDType> buffer;
        buffer.resize(image_width * image_height * 3); // Preallocate memory for the image buffer

        // Pull invariants out of the loops to avoid re-computing these
        const EiVector3d pixel_row_0 = matrix_pixel_spacing.row(0);
        const EiVector3d pixel_row_1 = matrix_pixel_spacing.row(1);
        const EiVector3d defocus_row_0 = matrix_defocus_disc.row(0);
        const EiVector3d defocus_row_1 = matrix_defocus_disc.row(1);
        const double color_scaling = 1.0 /number_of_samples; // Multiplication is faster than division, so we pre-divide it before looping

        // Progress bar - useful for higher anti-aliasing and/or refractive scenes
        std::string bar_title = "Processing scanlines:";
        ProgressBar pbar(bar_title, image_height);
        std::atomic<int> current_progress = 0;

        #pragma omp parallel for shared(stop_request) schedule(dynamic) 
        for (size_t j = 0; j < image_height; j++) {
            for (size_t i = 0; i < image_width; i++) {
                EiVector3d pixel_color = EiVector3d::Zero();
                // [SOBOL] One scramble value per pixel decorrelates the Sobol sequence between pixels (scrambled / randomized Sobol)
                // Derived deterministically from (i, j), so runs are reproducible
                 const unsigned long long pixel_scramble = sobol_pixel_scramble(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
                for (size_t k = 0; k < number_of_samples; k++) {
                    // Exit the main loop in rtmain when CTRL+C is pressed
                    if (stop_request) continue;

                    // [SOBOL] Sobol' point index = the sample number k within this pixel; same scramble for all samples of this pixel
                    SobolSampler sampler(static_cast<unsigned long long>(k), pixel_scramble);
                    // [SOBOL] Pixel anti-aliasing jitter from the reserved pixel dimensions (remapped from [0,1) to [-0.5, 0.5), much like we did for random_double)
                     const std::array<double,2> jitter = sampler.pixel_jitter();
                     double offset[2] = { jitter[0] - 0.5, jitter[1] - 0.5 };
                    //[MT19937 - LEGACY] white-noise AA jitter
                    //double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                    EiVector3d pixel_sample = pixel_00_center +
                        (i + offset[0]) * pixel_row_0 + (j + offset[1]) * pixel_row_1;
                        Ray current_ray;
                        if constexpr (camera_type == CameraType::THIN_LENS){
                            // [SOBOL] Thin lens consumes the reserved lens dimensions
                            current_ray = primary_ray_thin_lens(camera_center, pixel_sample, defocus_row_0, defocus_row_1, sampler);
                            //[MT19937 - LEGACY]
                            //current_ray = primary_ray_thin_lens(camera_center, pixel_sample, defocus_row_0, defocus_row_1);
                        }
                        else if constexpr (camera_type == CameraType::PINHOLE){
                            current_ray = primary_ray_pinhole(camera_center, pixel_sample);
                        }
                        //Clamp fireflies - optional, makes images less bright
                        //EiVector3d sample = return_ray_color_stack(current_ray, TLAS);
                        //double lum = 0.2126*sample.x() + 0.7152*sample.y() + 0.0722*sample.z();
                        //static constexpr double MAX_LUM = 10.0; // Tune per scene; hoist this out of the loop if using 
                        //if (lum > MAX_LUM) sample *= MAX_LUM / lum;
                        //pixel_color += sample;

                        // [SOBOL] Pass the per-path sampler into the path tracer
                        pixel_color += renderer::return_ray_color_stack(current_ray, TLAS, sampler);
                        //[MT19937 - LEGACY]
                        //pixel_color += renderer::return_ray_color_stack(current_ray, TLAS);

                
            }
                int px_idx = (i + j * image_width) * 3;
                // Divide by the number of samples to get the mean colour
                pixel_color = pixel_color * color_scaling;
                // Clamp each channel to [0,1]
                pixel_color = pixel_color.cwiseMax(0.0).cwiseMin(1.0); 
                if constexpr (color == RenderColor::GRAYSCALE) {
                    // Convert to a single-channel grayscale
                    const double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
                    // Scale to bytes
                    BufferDType gray_byte; // uint8_t or uint16_t
                    if constexpr (buffer_type == BufferType::UINT_8){
                        gray_byte = static_cast<BufferDType>(gray * 255.999);
                    }
                    else{
                        // Anything in the 8-16 bit range - we have pre-set the max_code_range to scale accordingly
                        gray_byte = static_cast<BufferDType>(gray * max_code_range + 0.5); 
                    }
                    buffer[px_idx] = gray_byte;
                    buffer[px_idx + 1] = gray_byte;
                    buffer[px_idx + 2] = gray_byte;
                }
                else if constexpr (color == RenderColor::COLOR) {
                    // Scale to bytes
                    if constexpr(buffer_type == BufferType::UINT_8){
                        pixel_color *= 255.999;
                        buffer[px_idx] = static_cast<uint8_t>(pixel_color.x());
                        buffer[px_idx + 1] = static_cast<uint8_t>(pixel_color.y());
                        buffer[px_idx + 2] = static_cast<uint8_t>(pixel_color.z());
                    }
                    else{  
                        pixel_color = pixel_color.cwiseMax(0.0).cwiseMin(1.0);
                        buffer[px_idx] = static_cast<uint16_t>(pixel_color.x() * max_code_range + 0.5);
                        buffer[px_idx + 1] = static_cast<uint16_t>(pixel_color.y() * max_code_range + 0.5);
                        buffer[px_idx + 2] = static_cast<uint16_t>(pixel_color.z() * max_code_range + 0.5);
                    }  
                }
            }
            
            // Update progress bar
            int progress = current_progress.fetch_add(1);
            if (omp_get_thread_num()==0) pbar.update(progress);
        }
        // Finish progress bar
        int progress = current_progress;
        pbar.finish();
        // Write the buffer in whatever output format we want
        outputwriter::save_image(buffer, image_height, image_width, output_filepath);
    };

    /**
     * @brief Setter for the MAX_DEPTH of the secondary ray bounces.
     * 
     * @param[in] max_depth (const int). Desired maximum depth. Higher is needed for refractive materials.
     */
    void set_depth(int max_depth);

     /**
     * @brief Setter for the background colour of the scene.
     * 
     * @param[in] color (const EiVector3d&) Desired background colour as an RGB triplet in the [0,1] range.
     */
    void set_background(const EiVector3d& color);

    /**
     * @brief Sets the maximum integer value based on the desired bit-depth in the output image.
     * This way we can store 8/10/12-bit depth images in 16-bit TIFF without scaling.
     * 
     * @param bit_depth (BitDepth) Desired bit-depth of the output image.
     */
    void set_max_code_range(const BitDepth bit_depth);

    /// @brief Picks the rendering function based on the grayscale setting, bit-depth, and output format.
    void set_rendering_function(const bool grayscale,
        const BitDepth bit_depth,
        const OutputFormat output_format);
}

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
    const std::filesystem::path output_filepath);

#endif // RTRENDER_H