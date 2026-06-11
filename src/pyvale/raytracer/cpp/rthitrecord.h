// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTHITRECORD_H
#define RTHITRECORD_H

// STD header files
#include <limits>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"

// Forward declaration (incomplete types) so we can use them in function pointers in BLAS while avoiding circular dependencies
struct RayState;

/**
 * @brief Struct storing the data from ray-element intersection (provided it is a hit and t_intersection < t stored here).
 * 
 * The data is stored for a SINGLE element, so while a BLAS_Node intersected might contain e.g., 3 QUAD9's, HitRecord would
 * keep the data only for a single QUAD9 - the one that was specifically intersected.
 */
// Struct size [bytes]: 6 x 24 + 3 x 8 + 2 x 4 + 1 = 177 bytes
struct HitRecord {
    EiVector3d point_intersection {EiVector3d::Zero()}; // Where ray intersects the mesh element
    EiVector3d normal_surface {EiVector3d::Zero()}; // Geometric normal vector
    EiVector3d normal_shading {EiVector3d::Zero()}; // Shading normal vector
    EiVector3d elem_interp_coords {EiVector3d::Zero()}; // E.g., barycentric coordinates for TRI3, bilinear interpolation coords for QUAD4
    EiVector3d face_color {EiVector3d::Zero()}; // 3D color - already sampled from texture or solid surface color. Albedo for diffuse/specular materials; absorption coefficient (sigma_a) for refractive ones
    EiVector3d emission{ EiVector3d::Zero() }; // Light source, which we currently don't have beyond ambient illumination, so keep at 0
    double t {std::numeric_limits<double>::infinity()};
    double refractive_index {1.0003}; // RI of the intersected element
    double thickness {1.0}; // Thickness (in world units) of the intersected element; used for refractive SHELL elements only
    void (*ray_material_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color, const double offset) {nullptr}; // Pointer to the function determining the interaction between the ray and the mesh material
    // Uncomment the below 2 lines if deciding to go for switch-based dispatch in return_ray_color_stack
    //ObjectType object_type;
    //int material;
    int hit_blas_idx; // Index of the intersected BLAS (as stored in TLAS)
    uint8_t hit_blas_priority; // Priority/nestedness of the intersected BLAS if it's a dielectric

    /**
     * @brief Normalises the shading and geometric normals, and flips the geometric normal so it points against the incoming ray.
     * 
     * @param[in] ray (const Ray&) Incoming ray to evaluate the normal direction against.
     */
    inline void normalize_and_flip_normals(const Ray& ray){
        normal_shading.stableNormalize(); // Stable normalize reduces risk of under- and over- flow
        normal_surface.stableNormalize();
        if (ray.direction.dot(normal_surface) > 0.0) {
            normal_surface = -normal_surface; // Flip normal if it hits the back face
        }
    }

    /**
     * @brief Aligns the direction of the shading and geometric normals.
     */
    inline void align_normals(){
        // Ensures that the geometric and shading normals point in the same direction.
        if (normal_shading.dot(normal_surface) < 0.0){
            normal_shading = -normal_shading;
        }
    }
};

#endif // RTHITRECORD_H