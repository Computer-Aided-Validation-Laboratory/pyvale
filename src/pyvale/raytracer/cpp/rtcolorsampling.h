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
    extern int lower_boundary, upper_boundary; // Boundaries for loops going over texel neighbourhoods

    // Sampler function declarations
    // Greyscale nearest neighbour sampling
    EiVector3d sample_texture_nearest_neighbour(const Texture& texture,
        const EiArray2d& uvs);

    // Greyscale Lanczos 3
    EiVector3d sample_texture_lanczos3(const Texture& texture,
        const EiArray2d& uvs);

    EiVector3d sample_texture_CatmullRom(const Texture& texture,
        const EiArray2d& uvs);

    EiVector3d sample_texture_MitchellNetravali(const Texture& texture,
        const EiArray2d& uvs);

    EiVector3d sample_texture_Bspline(const Texture& texture,
        const EiArray2d& uvs);

    EiVector3d sample_texture_quin_spline(const Texture& texture,
        const EiArray2d& uvs);


    // Setter for the current function
    void set(TextureSampler sampler_type);
}



