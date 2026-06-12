// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <limits>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"

// Forward declaration (incomplete types) so we can use them in function pointers in BLAS while avoiding circular dependencies
struct RayState;

// Struct size [bytes]: 6 x 24 + 3 x 8 + 3 x 4 + 1 = 181 bytes
struct HitRecord {
    EiVector3d point_intersection {EiVector3d::Zero()}; // Where ray intersects the mesh element
    EiVector3d normal_surface {EiVector3d::Zero()}; // Geometric normal (might not be needed; tbd)
    EiVector3d normal_shading {EiVector3d::Zero()}; // Shading normal
    EiVector3d elem_interp_coords {EiVector3d::Zero()}; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVector3d face_color {EiVector3d::Zero()}; // 3D color - already sampled from texture or solid surface color. Albedo for diffuse/specular materials, absorption for refractive ones
    EiVector3d emission{ EiVector3d::Zero() };     // light source
    double t {std::numeric_limits<double>::infinity()};
    double refractive_index {1.0003};
    double thickness {1.0};
    double ray_offset {0.0};
    void (*ray_material_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color) {nullptr}; // Pointer to the function determining the interaction between the ray and the mesh material
    // Uncomment the below 2 lines if deciding to go for switch-based dispatcj in return_ray_color
    //ObjectType object_type;
    //int material;
    int hit_blas_idx; // ID of the intersected BLAS
    uint8_t hit_blas_priority; // Priority of the intersected BLAS

    inline void normalize_and_flip_normals(const Ray& ray){
        // Normalizes normals and flips the geometric normal so that it points against the incoming ray
        // Used previously always after getting out of the intersection; but now our normals are always normalised, so deprecated
        normal_shading.stableNormalize(); // Stable normalize reduces risk of under- and over- flow
        normal_surface.stableNormalize();
        if (ray.direction.dot(normal_surface) > 0.0) {
            normal_surface = -normal_surface; // Flip normal if it hits the back face
        }
    }

    inline void align_normals(){
        // Ensures that the geometric and shading normals point in the same direction
        if (normal_shading.dot(normal_surface) < 0.0){
            normal_shading = -normal_shading;
        }
    }
};