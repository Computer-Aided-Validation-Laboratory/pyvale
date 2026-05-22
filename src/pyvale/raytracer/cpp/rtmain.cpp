// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>


// C++ header files
#include <cmath>
#include "./Eigen/Dense"
#include <filesystem>
#include <iostream>
#include <array>
#include <vector>
//#include <chrono>
#include <valgrind/callgrind.h>

// common_cpp header files
#include "../../common_cpp/Eigen/Dense"

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"
#include "rtcolorsampling.h"

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
    const std::vector<double>& scene_refractive_indices,
    const std::vector<int>& scene_mesh_priorities,
    const std::vector<int>& scene_mesh_object_types,
    const int texture_sampler,
    const int shading_type,
    const bool grayscale_flag) {


    //CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();
    // Use std::filesystem so it always constructs the path properly for the running OS
    std::filesystem::path output_filepath;
    std::string filename; // Output image file
    
    // Set the texture sampling algorithm based on the passed value
    texsampler::set(TextureSampler(texture_sampler));

    // Get the refractive index of the scene (typically air, but in case it is not)
    const int last_index = scene_refractive_indices.size() - 1;
    const float scene_ri = scene_refractive_indices[last_index]; // Scene RI is stored at the last position always

    // Render in colour or in grayscale; grayscale being the default
    void (*render_function_ptr)(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath) = &render_ppm_image<RenderColor::GRAYSCALE>;


    if (grayscale_flag == false){
        render_function_ptr = &render_ppm_image<RenderColor::COLOR>;
    }

    for (int timestep = 0; timestep < timestep_count; ++timestep){
        //TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, timestep, timestep_count); // target stack-based DoD implementation
        //std::chrono::time_point t1_build = std::chrono::high_resolution_clock::now();
        //TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, scene_uvs, scene_textures, scene_surface_types, timestep, timestep_count);
        TLAS current_TLAS = build_acceleration_structures(scene_coords_expanded, scene_normals_expanded, scene_face_colors, materials, scene_uvs, scene_textures, scene_surface_types, scene_refractive_indices, scene_mesh_priorities, scene_mesh_object_types, shading_type, timestep, timestep_count);
        //std::chrono::time_point t2_build = std::chrono::high_resolution_clock::now();

        
        // Iterate over all cameras and render an image for each
        for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
            EiVector3d camera_center = camera_centers[camera_idx];
            EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
            Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];
            Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_defocus_disc = matrix_defocus_discs[camera_idx];
            // Create the filepath for the rendered images
            filename = "rtimage_" + std::to_string(timestep) + "_cam" + std::to_string(camera_idx) + ".ppm"; // Output images in format rtimage_1_cam1 etc.
            output_filepath = output_directory;
            output_filepath.append(filename);
            std::cout << "Rendering frame " << (timestep+1) << "/" << timestep_count << std::endl;
            //std::chrono::time_point t1_render = std::chrono::high_resolution_clock::now();
            //CALLGRIND_START_INSTRUMENTATION;
            render_function_ptr(camera_center, pixel_00_center, matrix_pixel_spacing, matrix_defocus_disc, current_TLAS, image_height, image_width, number_of_samples, scene_ri, output_filepath);
            //render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, matrix_defocus_disc, current_TLAS, image_height, image_width, number_of_samples, scene_ri, output_filepath);
            //mock_ray_shoot(camera_center, pixel_00_center, matrix_pixel_spacing, matrix_defocus_disc, current_TLAS, image_height, image_width, number_of_samples, scene_ri, output_filepath);
            //CALLGRIND_STOP_INSTRUMENTATION;
            //std::chrono::time_point t2_render = std::chrono::high_resolution_clock::now();
            
            //std::chrono::duration t_render = std::chrono::duration_cast<std::chrono::milliseconds>(t2_render - t1_render);
            //std::cout << "Render time: " << t_render.count() << " ms \n";
        }

            //std::chrono::duration t_build = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_build - t1_build);
            //std::cout << "AS build time: " << t_build.count() << " ns \n";
    }
    //CALLGRIND_STOP_INSTRUMENTATION;
}