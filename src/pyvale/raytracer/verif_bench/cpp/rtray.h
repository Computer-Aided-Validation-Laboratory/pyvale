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
#include "rtmathutils.h"

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

// [SOBOL QMC]
inline Ray primary_ray_thin_lens(const EiVector3d& camera_center,
    const EiVector3d& pixel_sample,
    const EiVector3d& defocus_row_0,
    const EiVector3d& defocus_row_1,
    const SobolSampler& sampler){
        std::array<double, 2> defocus_disc_offset = sobol_point_in_unit_disk(sampler);
        EiVector3d defocus_disc_sample = defocus_disc_offset[0] * defocus_row_0 + defocus_disc_offset[1] * defocus_row_1;
        EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
        EiVector3d ray_direction = (pixel_sample - ray_origin).stableNormalized(); // ray direction in thin lens approx

        return { ray_origin, ray_direction.stableNormalized() }; 
};

/*
// [MT19937 - LEGACY] Original thin-lens primary ray (rejection-sampled disk)
// Retained for MT19937-vs-Sobol comparison. To switch back, restore this signature (drop the SobolSampler argument) and the matching call in
// render_img (rtrender.h)
inline Ray primary_ray_thin_lens(const EiVector3d& camera_center,
    const EiVector3d& pixel_sample,
    const EiVector3d& defocus_row_0,
    const EiVector3d& defocus_row_1){
        std::array<double, 2> defocus_disc_offset = point_in_unit_disk();
        EiVector3d defocus_disc_sample = defocus_disc_offset[0] * defocus_row_0 + defocus_disc_offset[1] * defocus_row_1;
        EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
        EiVector3d ray_direction = (pixel_sample - ray_origin).stableNormalized(); // ray direction in thin lens approx

        return { ray_origin, ray_direction.stableNormalized() }; 
};
*/


inline Ray primary_ray_pinhole(const EiVector3d& camera_center,
    const EiVector3d& pixel_sample){
        EiVector3d ray_direction = pixel_sample - camera_center;
        // ray_origin = camera_center for pinhole
        return { camera_center, ray_direction.stableNormalized() }; 
    }


#endif // RTRAY_H