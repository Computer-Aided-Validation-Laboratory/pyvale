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
    
    // Template parameters are the kernel functions above and loop_start and loop_end for different neighbourhood sizes, so we can cover quintic spline in the same template
    template <double (*kernel_function)(double), int loop_start, int loop_end>
    EiVector3d sample_greyscale(const Texture& texture, const EiArray2d& uvs) {
        // Retrieve values
        int height = texture.height;
        int width = texture.width;
        double u = uvs(0);
        double v = uvs(1);

        // Texture grid coordinates based on (u,v) of where in the texture the ray hit
        double texel_x = u * width - 0.5;
        double texel_y = v * height - 0.5; 

        // Indices of the texel whose center is the closest to the texture hit point. Lanczos window is centered around those
        int center_x = static_cast<int>(std::round(texel_x));
        int center_y = static_cast<int>(std::round(texel_y));

        double color = 0.0; // Single value that will have to be stashed into EiVector3d at the end
        double total_weight = 0.0; // Sum of weights over all neighbours used for normalisation

        // Pre-declare outside of the loop to avoid re-creating them multiple times
        int sample_x, sample_y;
        double sample_weight, distance_x, distance_y;
        
        for (int offset_y = loop_start; offset_y <= loop_end; ++offset_y) {
            for (int offset_x = loop_start; offset_x <= loop_end; ++offset_x) {
                // Texel indices of the currently processed neighbour texel (found via center_texel + offset)
                sample_x = std::clamp(center_x + offset_x, 0, width  - 1); // Clamp for texture access
                sample_y = std::clamp(center_y + offset_y, 0, height - 1);
                
                // Find the x- and y- distances to the neighbour texel (unclamped for the mathematical stencil points)
                distance_x = texel_x - static_cast<double>(center_x + offset_x); 
                distance_y = texel_y - static_cast<double>(center_y + offset_y);

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

    // Setter for the current function
    void set(TextureSampler sampler_type);
}