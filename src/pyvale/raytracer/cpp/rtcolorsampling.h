// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
// STD header files
#include <vector>

// ray tracer header files
#include "rteigentypes.h"
#include "rtbvh.h"


// Getter for (R,G,B) values for the intersected surface element if it uses solid colour
inline EiVector3d get_face_color(Eigen::Index min_row_idx,
    const std::vector<double>& face_color) {
    // Get values to colour the intersected face
    double c1 = face_color[min_row_idx* NODE_COORDINATES];
    double c2 = face_color[min_row_idx * NODE_COORDINATES + 1];
    double c3 = face_color[min_row_idx * NODE_COORDINATES + 2];
    EiVector3d face_color_vec;
    //face_color_vec << 0.5, 0.5, 0.5;
    face_color_vec << c1, c2, c3;
    return face_color_vec;
}

inline int get_face_material(Eigen::Index minRowIndex,
    const std::vector<int>& material) {
    // Get material type of the intersected face
    int m1 = material[minRowIndex * 3];

    return m1;
}

// Getter for (u,v) coordinates for the intersected surface element to interpolate texture
inline void get_face_uvs(Eigen::Index min_row_idx,
    const std::vector<double>& face_uvs,
    int element_node_count,
    double* out_element_uvs) { // Pointer, so we can pass an array depending on element node count without creating separate functions for every case
    // Get uv values of the intersected face
    int base_idx = min_row_idx * UV_COORDINATES * element_node_count;
    std::array<double, 2> element_node_uvs;
    
    // DEBUG PRINTS
    //std::cerr << "get_face_uvs" << std::endl;
    //std::cerr << "\t base_idx: " << base_idx << std::endl;
    //std::cerr << "\t min_row_idx: " << min_row_idx << std::endl;

    //std::cerr << "Face colors for this node: " << std::endl;
    //for (int i = 0; i < face_uvs.size(); i++){
    //    std::cerr << face_uvs[i]  << std::endl;
    //}
    //std::cerr << "Element uvs for node " << std::endl;

    // Find (u,v) for each node in the mesh element and write it in the passed output array
    for (int i = 0; i < element_node_count; i++){
        out_element_uvs[0 + i * 2] = face_uvs[base_idx + i * UV_COORDINATES]; // Element node u
        out_element_uvs[1 + i * 2] = face_uvs[base_idx + i * UV_COORDINATES + 1]; // Element node v
        //std::cerr << "\t " << i << " : " << out_element_uvs[0 + i * 2] << " (access idx: " << base_idx + i * UV_COORDINATES + 0 << "), " << out_element_uvs[1 + i * 2] << std::endl;
    }
    //std::cerr << std::endl;
}

// Scoped enum, so will not implicitly convert to int
// Ensure that these match the enum in Python. Integers used to avoid using strings in C-interface
enum class TextureSampler{
    NEAREST_NEIGHBOUR = 0,
    LANCZOS_2 = 1,
    LANCZOS_3 = 2,
    CATMULL_ROM = 3,
    MITCHELL_NETRAVALI = 4,
    BSPLINE = 5,
    QUINTIC_SPLINE = 6
};

namespace texsampler{

    // Pointer to the selected function
    extern EiVector3d (*sample_texture)(const Texture& texture, const EiArray2d& uvs);
    
    // Sampler function declarations

    // Greyscale nearest neighbour sampling
    EiVector3d sample_greyscale_nearest_neighbour(const Texture& texture,
        const EiArray2d& uvs);

    // Kernels (coefficient calculators) for the cubic filters used in the template below to avoid repeating code unnecessarily while avoiding runtime overhead

    inline double kernel_lanczos2(const double x);
    inline double kernel_lanczos3(const double x);
    inline double kernel_catmull_rom(const double x);
    inline double kernel_mitchell_netravali(const double x);
    inline double kernel_bspline(const double x);
    inline double kernel_quintic_spline(const double x);
    
    // TO DO: Delete remaining versions after discussing the profiling/timing results; use version 2 for now as on average, it was best

    // Template parameters are the kernel functions above and loop_start and loop_end for different neighbourhood sizes, so we can cover quintic spline in the same template
    // Version 2: Do not separate x and y, but process whole rows at once
    template <double (*kernel_function)(double), const int loop_start, const int loop_end>
    EiVector3d sample_greyscale(const Texture& texture, const EiArray2d& uvs) {
        // Retrieve values
        const int height = texture.height;
        const int width = texture.width;
        const double u = uvs(0);
        const double v = uvs(1);

        // Texture grid coordinates based on (u,v) of where in the texture the ray hit
        const double texel_x = u * width - 0.5;
        const double texel_y = v * height - 0.5; 

        // Indices of the texel whose center is the closest to the texture hit point. Lanczos window is centered around those
        const int center_x = static_cast<int>(std::round(texel_x));
        const int center_y = static_cast<int>(std::round(texel_y));

        constexpr int neighbourhood_span = loop_end - loop_start + 1;
         // 1. Precompute the x-data for the whole row
        double weights_x[neighbourhood_span];
        int indices_x[neighbourhood_span];
        double sum_weight_x = 0.0;

        for (int i = 0; i < neighbourhood_span; ++i) {
            int offset_x = loop_start + i;
            indices_x[i] = std::clamp(center_x + offset_x, 0, width - 1);
            weights_x[i] = kernel_function(texel_x - static_cast<double>(center_x + offset_x));
            sum_weight_x += weights_x[i];
        }

        double color = 0.0; // Single value that will have to be stashed into EiVector3d at the end
        double sum_weight_y = 0.0;

        // 2. Go over the y-axis and process the whole rows
        for (int offset_y = loop_start; offset_y <= loop_end; ++offset_y) {
            // Calculate Y-properties for this row
            int sample_y = std::clamp(center_y + offset_y, 0, height - 1);
            double distance_y = texel_y - static_cast<double>(center_y + offset_y);
            double weight_y = kernel_function(distance_y);
            
            sum_weight_y += weight_y;

            double row_color = 0.0;
            int current_row_offset = sample_y * width;
            
            // This inner loop acts as a 1D dot product. 
            // Because 'neighbourhood_span' is a compile-time constant, compilers should vectorise this into a single SIMD instruction - but check if this is the case
            for (int i = 0; i < neighbourhood_span; ++i) {
                row_color += weights_x[i] * texture.data[current_row_offset + indices_x[i]];
            }
            
            // Accumulate the row's contribution
            color += weight_y * row_color;
        }

        // Total weight calculation
        double total_weight = sum_weight_x * sum_weight_y; // Sum of weights over all neighbours used for normalisation
        // Normalise (or set to 0 if very small) - important near edges
        const double color_normalised = (std::abs(total_weight) > 1e-12) ? (color / total_weight) : 0.0;
        EiVector3d output;
        output << color_normalised, color_normalised, color_normalised;

        return output;
    }

  /*  
    //VERSION 1: Potentially optimised by doing 1D passes for x and y
    template <double (*kernel_function)(double), const int loop_start, const int loop_end>
    EiVector3d sample_greyscale(const Texture& texture, const EiArray2d& uvs) {
        // Retrieve values
        const int height = texture.height;
        const int width = texture.width;
        const double u = uvs(0);
        const double v = uvs(1);

        // Texture grid coordinates based on (u,v) of where in the texture the ray hit
        const double texel_x = u * width - 0.5;
        const double texel_y = v * height - 0.5; 

        // Indices of the texel whose center is the closest to the texture hit point. Lanczos window is centered around those
        const int center_x = static_cast<int>(std::round(texel_x));
        const int center_y = static_cast<int>(std::round(texel_y));

        constexpr int neighbourhood_span = loop_end - loop_start + 1;
        // Create arrays for rows and columns to be processed separately so we can avoid nested loops
        double weights_x[neighbourhood_span];
        double weights_y[neighbourhood_span];
        int indices_x[neighbourhood_span];
        int row_offsets[neighbourhood_span];

        double sum_weight_x = 0.0;
        double sum_weight_y = 0.0;
        
        // 1D pass for X: Precompute clamped X indices and X weights
        for (int i = 0; i < neighbourhood_span; ++i) {
            int offset_x = loop_start + i;
            indices_x[i] = std::clamp(center_x + offset_x, 0, width - 1);
            
            double distance_x = texel_x - static_cast<double>(center_x + offset_x);
            weights_x[i] = kernel_function(distance_x);
            sum_weight_x += weights_x[i];
        }

        // 1D pass for Y: Precompute clamped Y row offsets and Y weights
        for (int j = 0; j < neighbourhood_span; ++j) {
            int offset_y = loop_start + j;
            int clamped_y = std::clamp(center_y + offset_y, 0, height - 1);
            row_offsets[j] = clamped_y * width;
            
            double distance_y = texel_y - static_cast<double>(center_y + offset_y);
            weights_y[j] = kernel_function(distance_y);
            sum_weight_y += weights_y[j];
        }

        // 2D Pass: Process whole rows to accumulate color
        double color = 0.0; // Single value that will have to be stashed into EiVector3d at the end
        
        for (int j = 0; j < neighbourhood_span; ++j) {
            double row_color = 0.0;
            int current_row_offset = row_offsets[j];
            
            // Inner loop does very little so hopefully can be SIMD-ed
            for (int i = 0; i < neighbourhood_span; ++i) {
                row_color += weights_x[i] * texture.data[current_row_offset + indices_x[i]];
            }
            
            // Multiply the accumulated row by the Y weight
            color += weights_y[j] * row_color;
        }

        // Total weight calculation
        double total_weight = sum_weight_x * sum_weight_y; // Sum of weights over all neighbours used for normalisation
        // Normalise (or set to 0 if very small) - important near edges
        const double color_normalised = (std::abs(total_weight) > 1e-12) ? (color / total_weight) : 0.0;
        EiVector3d output;
        output << color_normalised, color_normalised, color_normalised;

        return output;
    }

    //VERSION 0: Unoptimised, nested loops, processing each (x,y) separately
    template <double (*kernel_function)(double), const int loop_start, const int loop_end>
    EiVector3d sample_greyscale(const Texture& texture, const EiArray2d& uvs) {
        // Retrieve values
        const int height = texture.height;
        const int width = texture.width;
        const double u = uvs(0);
        const double v = uvs(1);

        // Texture grid coordinates based on (u,v) of where in the texture the ray hit
        const double texel_x = u * width - 0.5;
        const double texel_y = v * height - 0.5; 

        // Indices of the texel whose center is the closest to the texture hit point. Lanczos window is centered around those
        const int center_x = static_cast<int>(std::round(texel_x));
        const int center_y = static_cast<int>(std::round(texel_y));

        double color = 0.0; // Single value that will have to be stashed into EiVector3d at the end
        double total_weight = 0.0; // Sum of weights over all neighbours used for normalisation

        // Declare outside of the loop to avoid reallocating (although this should not be a worry in modern C++)
        int sample_x, sample_y;
        double sample_weight, distance_x, distance_y;
        
        for (int offset_y = loop_start; offset_y <= loop_end; ++offset_y) {
            // Texel index of the currently processed neighbour texel (found via center_texel + offset); hoisted out of the inner loop to avoid recalculating unnecessarily
            sample_y = std::clamp(center_y + offset_y, 0, height - 1);
            // Find the y- distance to the neighbour texel (unclamped for the mathematical stencil points)
            distance_y = texel_y - static_cast<double>(center_y + offset_y);
            for (int offset_x = loop_start; offset_x <= loop_end; ++offset_x) {
                
                // Same as above for x
                sample_x = std::clamp(center_x + offset_x, 0, width  - 1); // Clamp for texture access
                distance_x = texel_x - static_cast<double>(center_x + offset_x); 

                // Find the weight contributed by this neighbour texel to the final interpolated result
                sample_weight = kernel_function(distance_x) * kernel_function(distance_y);

                // Colour is a weighted colour sum
                // texel_value = texture.data[sample_y * width + sample_x]
                color += sample_weight * texture.data[sample_y * width + sample_x];
                total_weight += sample_weight;

            }
        }
        // Normalise (or set to 0 if very small) - important near edges
        const double color_normalised = (std::abs(total_weight) > 1e-12) ? (color / total_weight) : 0.0;
        EiVector3d output;
        output << color_normalised, color_normalised, color_normalised;

        return output;
    }
   */
    // Setter for the current function
    void set(TextureSampler sampler_type);
}