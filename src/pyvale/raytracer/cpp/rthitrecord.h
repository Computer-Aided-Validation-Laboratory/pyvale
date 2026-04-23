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

struct HitRecord {
    EiVector3d point_intersection {EiVector3d::Zero()};
    EiVector3d normal_surface {EiVector3d::Zero()};
    EiVector3d elem_interp_coords {EiVector3d::Zero()}; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVector3d face_color {EiVector3d::Zero()}; // 3D color - already sampled from texture or solid surface color
    double t {std::numeric_limits<double>::infinity()};
};

inline void set_face_normal(const Ray& ray, EiVector3d& normal_surface) {
    // Normalises the surface normal at the intersection point and determines which way the ray hits the object. Flips the normal if it hits the back face
    normal_surface = normal_surface.normalized();
    if (ray.direction.dot(normal_surface) > 0.0) {
        normal_surface = -normal_surface; // Flip normal if it hits the back face
    }
};