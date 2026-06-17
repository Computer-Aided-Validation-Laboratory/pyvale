// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

// raytracer header files
#include "rtrender.h"
#include "rthitrecord.h"
#include "rtrayintersection.h"
#include "rtmathutils.h"

void ray_diffuse(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset){
    // Secondary ray is randomly scattered from the hit point
    // Depends on: Incident ray direction
    // Use non-uniform Lambertian distribution weighed by cos of the angle between the indicent ray and surface normal. Scattering is more likely close to the normal.

    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    EiVector3d next_accumulated_color = current_state.accumulated_color.cwiseProduct(albedo); // Pre-calculate the baseline for the next bounce
    
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    const EiVector3d normal_shade = intersection_record.normal_shading; // Shading normal

    // Generate orthonormal basis (Hughes-Moller method)
    //EiVector3d b1 =
    //    ((fabs(normal_shade.x()) > 0.1 ? EiVector3d(0,1,0) : EiVector3d(1,0,0))
    //    .cross(normal_shade)).normalized();
    //EiVector3d b2 = normal_shade.cross(u);

    // Generate orthonormal basis (Duff et al. from Pixar method) (https://jcgt.org/published/0006/01/01/paper-lowres.pdf)
    // Determine the sign of the z-component using std::copysign to avoid branching
    // If n.z >= 0, sign is 1.0; else -1.0.
    double sign = std::copysign(1.0, normal_shade.z());
    
    // Calculate intermediate values
    double a = -1.0 / (sign + normal_shade.z());
    double b = normal_shade.x() * normal_shade.y() * a;
    
    // Generate the two perpendicular tangent vectors
    EiVector3d b1 = EiVector3d(1.0 + sign * normal_shade.x() * normal_shade.x() * a, sign * b, -sign * normal_shade.x());
    EiVector3d b2 = EiVector3d(b, sign + normal_shade.y() * normal_shade.y() * a, -normal_shade.y());

    // Cosine-weighted hemisphere sampling
    double r1 = 2 * M_PI * (random_double());
    double r2 = random_double();
    double r2s = sqrt(r2);
    
    const EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal

    Ray ray_new;
    ray_new.origin = intersection_record.point_intersection + normal_geo * offset;
   
    EiVector3d direction_scatter = (b1 * cos(r1) * r2s + b2 * sin(r1) * r2s + normal_shade * sqrt(1 - r2));
    EiVector3d new_dir = direction_scatter.stableNormalized();
    if (new_dir.squaredNorm() < 0.5){
        new_dir = normal_shade; // Degenerate fallback - reuse normal
    }
    ray_new.direction = new_dir;
    // Below is another safeguard (just like offsetting the spawned ray origin)
    // Technically, you should be able to use one OR another, but many production renderers use both
    ray_new.t_min = SPAWNED_T_MIN_BASE * std::max(1.0, intersection_record.point_intersection.norm());

    /*
    std::cerr << "DIFFUSE" << std::endl;
    std::cerr << "\tShading normal: " << normal_shade.x() << ", " << normal_shade.y() << ", " << normal_shade.z() << std::endl;
    std::cerr << "\tGeometric normal: " << normal_geo.x() << ", " << normal_geo.y() << ", " << normal_geo.z() << std::endl;
    std::cerr << "\tScattering direction: " << direction_scatter.x() << ", " << direction_scatter.y() << ", " << direction_scatter.z() << std::endl;
    std::cerr << "\tPoint of intersection: " << p.x() << ", " << p.y() << ", " << p.z() << std::endl; 
    */
    
    //stack.emplace_back(ray_new, next_accumulated_color, intersection_record.refractive_index, current_state.depth + 1);
    stack.emplace_back(ray_new, next_accumulated_color, current_state.interior_list, current_state.depth + 1, current_state.interior_count);
}

void ray_specular(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset){
    // Secondary ray traced in the direction about the normal
    // Depends on: angle between the viewing direction and the surface normal
    const EiVector3d p = intersection_record.point_intersection; // Point of intersection
    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal
    EiVector3d normal_shade = intersection_record.normal_shading; // Shading normal

    EiVector3d next_accumulated_color = current_state.accumulated_color.cwiseProduct(albedo); // Pre-calculate the baseline for the next bounce
    EiVector3d ray_direction = current_state.ray.direction;

    EiVector3d reflected = ray_direction - 2 * ray_direction.dot(normal_shade) * normal_shade;
    
    if (reflected.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
        reflected = ray_direction - 2 * ray_direction.dot(normal_geo) * normal_geo;
    }
    Ray ray_new;
    ray_new.origin = intersection_record.point_intersection + normal_geo * offset;
    ray_new.direction = reflected.stableNormalized();
    ray_new.t_min = SPAWNED_T_MIN_BASE * std::max(1.0, intersection_record.point_intersection.norm());

    /* std::cerr << "SPECULAR" << std::endl;
    std::cerr << "\tShading normal: " << normal_shade.x() << ", " << normal_shade.y() << ", " << normal_shade.z() << std::endl;
    std::cerr << "\tGeometric normal: " << normal_geo.x() << ", " << normal_geo.y() << ", " << normal_geo.z() << std::endl;
    std::cerr << "Reflected ray direction: " << reflected.x() << ", " << reflected.y() << ", " << reflected.z() << std::endl;
    std::cerr << "\tPoint of intersection: " << intersection_record.point_intersection.x() << ", " << intersection_record.point_intersection.y() << ", " << intersection_record.point_intersection.z() << std::endl; */

    //stack.emplace_back(ray_new, next_accumulated_color, intersection_record.refractive_index, current_state.depth + 1);
    stack.emplace_back(ray_new, next_accumulated_color, current_state.interior_list, current_state.depth + 1, current_state.interior_count);
}

// We don't really need most these arguments, but this is to match the function pointer signature to avoid having a switch in the rendering loop
void ray_unlit(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset){

    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.face_color);
    return;
}