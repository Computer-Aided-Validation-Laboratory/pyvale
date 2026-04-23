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
//#include <valgrind/callgrind.h>

// common_cpp header files
#include "../../common_cpp/Eigen/Dense"

// raytracer header files
#include "rteigentypes.h"
#include "rtrender.h"

namespace nb = nanobind;

void render_scene(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_uvs,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_textures,
    const std::vector<int>& scene_surface_types) {


    //CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();
    // Use std::filesystem so it always constructs the path properly for the running OS
    std::filesystem::path output_filepath;
    std::string filename; // Output image file
    

    //std::chrono::time_point t1_d = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timestep_count; ++timestep){
        //TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, timestep, timestep_count); // target stack-based DoD implementation
        TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, scene_uvs, scene_textures, scene_surface_types, timestep, timestep_count);
        
        // Iterate over all cameras and render an image for each
        for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
            EiVector3d camera_center = camera_centers[camera_idx];
            EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
            Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];
            // Create the filepath for the rendered images
            filename = "rtimage_" + std::to_string(timestep) + "_cam" + std::to_string(camera_idx) + ".ppm"; // Output images in format rtimage_1_cam1 etc.
            output_filepath = output_directory;
            output_filepath.append(filename);
            std::cout << "Rendering frame " << (timestep+1) << "/" << timestep_count << std::endl;
            render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, test_TLAS, image_height, image_width, number_of_samples, output_filepath);
        }
    }

    //std::chrono::time_point t2_d = std::chrono::high_resolution_clock::now();
    //std::chrono::duration t_d = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_d - t1_d);
    //std::cout << "Iterative, DoD approach duration: " << t_d.count() << "ns \n";

    
        
    //CALLGRIND_STOP_INSTRUMENTATION;
}


/*
void render_scene_color(const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_directory,
    const int timestep_count,
    const std::vector<nb::DRef<EiVector3d>> camera_centers,
    const std::vector<nb::DRef<EiVector3d>> pixel_00_centers,
    const std::vector<nb::DRef<Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>>> matrix_pixel_spacings,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_coords_expanded,
    const std::vector<nb::ndarray<const double, nb::c_contig>>& scene_face_colors) {


    //CALLGRIND_START_INSTRUMENTATION;
    size_t num_cameras = camera_centers.size();
    // Use std::filesystem so it always constructs the path properly for the running OS
    std::filesystem::path output_filepath;
    std::string filename; // Output image file
    

    //std::chrono::time_point t1_d = std::chrono::high_resolution_clock::now();
    for (int timestep = 0; timestep < timestep_count; ++timestep){
        TLAS test_TLAS = build_acceleration_structures(scene_coords_expanded, scene_face_colors, timestep, timestep_count); // target stack-based DoD implementation
        
        // Iterate over all cameras and render an image for each
        for (size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
            EiVector3d camera_center = camera_centers[camera_idx];
            EiVector3d pixel_00_center = pixel_00_centers[camera_idx];
            Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor> matrix_pixel_spacing = matrix_pixel_spacings[camera_idx];
            // Create the filepath for the rendered images
            filename = "rtimage_" + std::to_string(timestep) + "_cam" + std::to_string(camera_idx) + ".ppm"; // Output images in format rtimage_1_cam1 etc.
            output_filepath = output_directory;
            output_filepath.append(filename);
            std::cout << "Rendering frame " << (timestep+1) << "/" << timestep_count << std::endl;
            render_ppm_image(camera_center, pixel_00_center, matrix_pixel_spacing, test_TLAS, image_height, image_width, number_of_samples, output_filepath);
        }
    }

    //std::chrono::time_point t2_d = std::chrono::high_resolution_clock::now();
    //std::chrono::duration t_d = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_d - t1_d);
    //std::cout << "Iterative, DoD approach duration: " << t_d.count() << "ns \n";

    
        
    //CALLGRIND_STOP_INSTRUMENTATION;
}
    */