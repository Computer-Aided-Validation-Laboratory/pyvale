// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTRAY_H
#define RTRAY_H

// STD header files
#include <limits>

// ray tracer header files
#include "rteigentypes.h"

/**
 * @brief Struct storing the data for each ray.
 */
struct Ray {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    EiVector3d origin; // (x,y,z) coordinates
    EiVector3d direction; // (x,y z) coordinates of the ray direction vector; typically, this will be normalised
    double t_min {-std::numeric_limits<double>::infinity()}; // Smallest t value for this ray; we start primary rays at -infinity, then use this to offset secondary rays to avoid self-intersection
    double t_max{ std::numeric_limits<double>::infinity() }; // Largest t value for this ray; start at infinity and look for smaller values

    // Constructor
    Ray() = default;
    Ray(const EiVector3d origin_, const EiVector3d direction_) : origin(origin_), direction(direction_) {};
};

/**
 * @brief Finds the value of the ray at a given distance t.
 * 
 * @param[in] t (const double) Distance from the origin used to multiply the direction vector.
 * @param[in] ray (const Ray&) Struct representing given ray.
 * 
 * @return (EiVector3d) Row-major 3D vector evaluating the ray equation: ray(t) = origin + t * direction.
 */
inline EiVector3d ray_at_t(const double t, const Ray& ray) {
    return ray.origin + t * ray.direction;
};

#endif // RTRAY_H