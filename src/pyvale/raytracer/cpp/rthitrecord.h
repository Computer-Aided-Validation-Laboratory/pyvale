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

enum MaterialType : int { 
    NOT_DEFINED = 0,
    DIFFUSE = 1, 
    SPECULAR = 2, 
    REFRACTIVE = 3,
    UNLIT = 4 
};

// Forward declaration (incomplete types) so we can use them in function pointers in BLAS while avoiding circular dependencies
struct RayState;

// Struct size [bytes]: 5 x 24 + 1 x 8 + 1 x 4 = 132 bytes
struct HitRecord {
    EiVector3d point_intersection {EiVector3d::Zero()}; 
    EiVector3d normal_surface {EiVector3d::Zero()}; // Geometric normal (might not be needed; tbd)
    EiVector3d normal_shading {EiVector3d::Zero()}; // Shading normal
    EiVector3d elem_interp_coords {EiVector3d::Zero()}; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVector3d face_color {EiVector3d::Zero()}; // 3D color - already sampled from texture or solid surface color, albedo
    EiVector3d emission{ EiVector3d::Zero() };     // light source
    double t {std::numeric_limits<double>::infinity()};
    void (*ray_material_ptr)(const RayState& current_state, const HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color) {nullptr}; // Pointer to the function determining the interaction between the ray and the mesh material
    //MaterialType material{ NOT_DEFINED }; // int 
};

inline void set_face_normal(const Ray& ray, EiVector3d& normal_surface) {
    // Normalises the surface normal at the intersection point and determines which way the ray hits the object. Flips the normal if it hits the back face
    normal_surface = normal_surface.normalized();
    if (ray.direction.dot(normal_surface) > 0.0) {
        normal_surface = -normal_surface; // Flip normal if it hits the back face
    }
};
