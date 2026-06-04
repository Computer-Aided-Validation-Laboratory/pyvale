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
#include "rtmaterials.h"

static constexpr int MAX_DEPTH = 50; // Max depth for the secondary rays

// Radiance with refractive materials - but we could make this into a separate option if refractive materials are present in the scene to avoid needing to branch into true/false hits if not necessary?
// This case would also have its own separate HitRecord, RayState structs since we could carry less data and fit more of those into cache lines
EiVector3d return_ray_color_stack(const Ray& primary_ray,
    const double scene_ri,
    const TLAS& TLAS){

    EiVector3d total_color = EiVector3d::Zero();
    //std::vector<RayState> stack; // Not thread safe
    thread_local std::vector<RayState> stack; // Thread safe
    stack.clear();
    stack.reserve(MAX_DEPTH);
    stack.emplace_back(primary_ray, scene_ri);

    void (*ray_material_interaction_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color); // Pointer to the function determining the interaction between the ray and the mesh material

    while(!stack.empty()){
        RayState current_state = stack.back();
        stack.pop_back();
        const Ray& current_ray = current_state.ray;

        HitRecord intersection_record; // Create HitRecord struct
        // Look for the intersection for this ray
        IntersectionOutput intersection;
        const bool hit_anything = intersect_TLAS(current_ray, TLAS, intersection, intersection_record);

        EiVector3d absorption = EiVector3d::Zero(); // Set default absorption to 0.0 = clear medium
        const bool has_medium = current_state.interior_count > 0; // Check if our ray has traversed any media that could attenuate the accumulated colour
        if (has_medium){
            absorption = find_top_absorption(&current_state.interior_list[0], current_state.interior_count, EiVector3d::Zero());
        }
        const bool has_absorption = absorption.x() > 0.0 || absorption.y() > 0.0 || absorption.z() > 0.0; // Save ourselves having to compute exponentials if there is no absorption
        
        if (!hit_anything) {
            if (has_absorption) {
                // Apply huge distance - ray travels into the sky, so we account for the light lost as the ray travels from inside of some absorbing medium out of the scene
                apply_absorption(current_state.accumulated_color, absorption, 1e30);
            }
            total_color += current_state.accumulated_color.cwiseProduct(ray_blue_sky(current_ray)); // Sky/background colour
            continue; // Early termination - no bounces here anyway
        }

        
        if (has_absorption){
            // Note that since we store t from the ray equation ray(t) = origin_vector + t * direction_vector, t = (intersection_record.point_intersection - current_ray.origin).norm() (this has been tested within the code, too)
            //double path_length = intersection_record.point_intersection.norm();
            double path_length = intersection_record.t;
            apply_absorption(current_state.accumulated_color, absorption, path_length);
        }
       
        // Assign what happens with the secondary rays based on the material pointer
        ray_material_interaction_ptr = intersection_record.ray_material_ptr;
        
        // Handle nested dielectrics - if using function pointer approach
        // Classify nested dielectrics if material is refractive
        if (intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SOLID> || intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SHELL>) {

            int hit_idx = intersection_record.hit_blas_idx;
            int hit_priority = intersection_record.hit_blas_priority;
            double hit_ri = intersection_record.refractive_index;
            // Get and compare the max priority currently surrounding the ray (or min value of int, if the list is empty)
            int top_idx = interior_highest_priority_idx(&current_state.interior_list[0], current_state.interior_count);
            int top_priority;

            if (top_idx < 0){
                top_priority = std::numeric_limits<int>::min();
            }
            else {
                top_priority = current_state.interior_list[top_idx].priority;
            }
            
            // Check if it is a true hit
            if (!(current_state.interior_count == 0 || hit_priority >= top_priority)){
                //std::cerr << "Inside interior count check" << std::endl;
                // False hit: priority of hit object < max priority in interior list (Schmidt's algorithm for nested volumes)
                // => Do not shade; re-cast the ray from hit point in the same direction by pushing a new RayState whose origin is the current hit point
                RayState next_state = current_state;
                interior_toggle(&next_state.interior_list[0], next_state.interior_count, hit_idx, hit_priority, hit_ri, intersection_record.face_color);
                // Offset the ray minimnally to avoid self-intersecting - much like we do for all secondary rays
                /*
                const double offset = std::numeric_limits<double>::epsilon() * 10.0 *
                    std::max({std::fabs(intersection_record.point_intersection.x()),
                    std::fabs(intersection_record.point_intersection.y()),
                    std::fabs(intersection_record.point_intersection.z())});*/
                const double offset = intersection_record.ray_offset;
                next_state.ray.origin = intersection_record.point_intersection + offset * current_ray.direction;
                next_state.ray.direction = current_ray.direction;
                next_state.ray.t_min = current_ray.t_min;
                next_state.ray.t_max = std::numeric_limits<double>::infinity();
                // DO NOT INCREMENT DEPTH - false hits are invisible according to the paper, so they do not affect the ray bounce count or energy
                stack.push_back(next_state);
                continue;
            }
        }
        // Explicit depth limit with ambient fallback
        if (current_state.depth >= MAX_DEPTH) {
            // Add a fallback ambient color to compensate for truncated energy 
            // Avoids the "plain black shadows" caused by zero light return
            EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2; 
            total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission + ambient_fallback);
            continue; 
        }
        
        
        EiVector3d albedo = intersection_record.face_color;
        if (current_state.depth > MAX_DEPTH/2) { // Start early termination if we are at least halfway through the maximum allowed depth
            // Russian roulette early termination
            // Clamp to prevent infinite loops (p=1.0) and division by zero (p=0.0)
            double p = std::clamp(albedo.maxCoeff(), 0.05, 0.95);
            if (random_double() > p){  // Note: for multi-threading this will have to be replaced with thread_local generator
            //if ((double)rand() / RAND_MAX > p){ // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
                //return emitted;
                total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission);
                continue;
            }
            albedo /= p;
        }

        // True hit: priority of hit object >= max priority in interior list OR the list is empty
        // Shade normally
        // Process ray and update the stack and total color based on the material of the intersected mesh
        // FUNCTION POINTER VARIANT
        ray_material_interaction_ptr(current_state, intersection_record, albedo, stack, total_color);

        // SWITCH DISPATCH VARIANT
        // Requires updating rtbvh.cpp and .h, rthitrecord, rtrayintersection (intersect_BLAS) to store material & object_type data
        // During profiling in May 2026, this showed marginally fewer L1 instruction cache misses (0-0.01 percent point difference in all test runs), BUT the average runtime was 1.58% worse
        // This was not enough to rule out this approach definitely, so while the function pointer approach was retained as the default, this is kept commented out in case
        /*
        switch (intersection_record.material) {
            case UNLIT: {
            ray_unlit(current_state, intersection_record, albedo, stack, total_color);
                break;
            }
            case DIFFUSE: { // Diffuse
                ray_diffuse(current_state, intersection_record, albedo, stack, total_color);
                break;
            }
            case SPECULAR: {// Specular (mirror)
                ray_specular(current_state, intersection_record, albedo, stack, total_color);
                break;
            }
            case REFRACTIVE: {// Refraction (dielectric)
                // Check for false hit
                int hit_idx = intersection_record.hit_blas_idx;
                int hit_priority = intersection_record.hit_blas_priority;
                double hit_ri = intersection_record.refractive_index;
                // Get and compare the max priority currently surrounding the ray (or min value of int, if the list is empty)
                int top_idx = interior_highest_priority_idx(&current_state.interior_list[0], current_state.interior_count);
                int top_priority;

                if (top_idx < 0){
                    top_priority = std::numeric_limits<int>::min();
                }
                else {
                    top_priority = current_state.interior_list[top_idx].priority;
                }
                
                // Check if it is a true hit
                if (!(current_state.interior_count == 0 || hit_priority >= top_priority)){
                    // False hit: priority of hit object < max priority in interior list (Schmidt's algorithm for nested volumes)
                    // => Do not shade; re-cast the ray from hit point in the same direction by pushing a new RayState whose origin is the current hit point
                    RayState next_state = current_state;
                    interior_toggle(&next_state.interior_list[0], next_state.interior_count, hit_idx, hit_priority, hit_ri, intersection_record.face_color);
                    // Offset the ray minimnally to avoid self-intersecting - much like we do for all secondary rays
                    const double offset = std::numeric_limits<double>::epsilon() * 10.0 *
                        std::max({std::fabs(intersection_record.point_intersection.x()),
                        std::fabs(intersection_record.point_intersection.y()),
                        std::fabs(intersection_record.point_intersection.z())});
                    next_state.ray.origin = intersection_record.point_intersection + offset * current_ray.direction;
                    next_state.ray.direction = current_ray.direction;
                    next_state.ray.t_min = current_ray.t_min;
                    next_state.ray.t_max = std::numeric_limits<double>::infinity();
                    // DO NOT INCREMENT DEPTH - false hits are invisible according to the paper, so they do not affect the ray bounce count or energy
                    stack.push_back(next_state);
                    continue;
                }
                if (intersection_record.object_type == ObjectType::SOLID){
                    ray_refractive<ObjectType::SOLID>(current_state, intersection_record, albedo, stack, total_color);
                }
                else{
                    ray_refractive<ObjectType::SHELL>(current_state, intersection_record, albedo, stack, total_color);
                }
                break;
            }
        }*/
        
    } // Stack while loop
    return total_color;
} 

/*
// Previous version without nested refractive materials or Beer-Lambert
EiVector3d return_ray_color_stack_nr(const Ray& primary_ray, const double scene_ri, const TLAS& TLAS){
    EiVector3d total_color = EiVector3d::Zero();
    std::vector<RayState> stack;
    stack.reserve(MAX_DEPTH);
    stack.emplace_back(primary_ray, scene_ri);
    void (*ray_material_interaction_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color); // Pointer to the function determining the interaction between the ray and the mesh material

    while(!stack.empty()){
        RayState current_state = stack.back();
        stack.pop_back();
        current_state.ray.direction.stableNormalize();
        Ray current_ray = current_state.ray;

        HitRecord intersection_record; // Create HitRecord struct
        // Look for the first intersection for this ray
        IntersectionOutput intersection;
        intersect_TLAS(current_ray, TLAS, intersection, intersection_record);

        ray_material_interaction_ptr = intersection_record.ray_material_ptr;
        //intersection_record.temp_flat_shading(); // Temporary function to swap shading normal with geometric normal and test flat shading before it is implemented as its own separate option

        if (intersection_record.t == std::numeric_limits<double>::infinity()) {
            //const EiVector3d blue_sky(0.5, 0.5, 0.5);
            const EiVector3d blue_sky = ray_blue_sky(current_ray); // Early termination - no bounces here anyway
            total_color += current_state.accumulated_color.cwiseProduct(blue_sky);
            continue;
        }
        
        EiVector3d emitted = intersection_record.emission;
        EiVector3d albedo = intersection_record.face_color;
        
        // Explicit depth limit with ambient fallback
        if (current_state.depth >= MAX_DEPTH) {
        //    // Add a fallback ambient color to compensate for truncated energy 
            // Avoids the "plain black shadows" caused by zero light return
            EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2; 
            total_color += current_state.accumulated_color.cwiseProduct(emitted + ambient_fallback);
            continue; 
        }
        
        if (current_state.depth > MAX_DEPTH/2) { // Start early termination if we are at least halfway through the maximum allowed depth
            // Russian roulette early termination
            double p = std::max({albedo.x(), albedo.y(), albedo.z()});
            // Clamp to prevent infinite loops (p=1.0) and division by zero (p=0.0)
            p = std::clamp(p, 0.05, 0.95); 
            if (random_double() > p){  // Note: for multi-threading this will have to be replaced with thread_local generator
            //if ((double)rand() / RAND_MAX > p){ // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
                //return emitted;
                total_color += current_state.accumulated_color.cwiseProduct(emitted);
                continue;
            }
            albedo /= p;
        }
        // Process ray and update the stack and total color based on the material of the intersected mesh
        ray_material_interaction_ptr(current_state, intersection_record, albedo, stack, total_color);
    } // Stack while loop
    return total_color;
} 
*/

/*
// This a new radiance function with lighting
EiVector3d return_ray_color_new(const Ray& ray,
                           const TLAS& TLAS,
                           int depth = 0) {

    HitRecord rec; // Create HitRecord struct
    // Look for intersection
    IntersectionOutput intersection;
    intersect_TLAS(ray, TLAS, intersection, rec);

    // rec.material = DIFFUSE;

    // Blue sky gradient
    if (rec.t == std::numeric_limits<double>::infinity() 
        || rec.material == NOT_DEFINED) {
        double a = 0.5 * (ray.direction(1) + 1.0);
        static EiVector3d white, blue;
        white << 1.0, 1.0, 1.0;
        blue << 0.5, 0.7, 1.0;
        return (1.0 - a) * white + a * blue;
    }

    set_face_normal(ray, rec.normal_surface);

    EiVector3d emitted = rec.emission;
    EiVector3d albedo = rec.face_color;

    // Russian roulette
    double p = std::max({albedo.x(), albedo.y(), albedo.z()});
    if (depth > 5) {
        if ((double)rand() / RAND_MAX > p)
            return emitted;
        albedo /= p;
    }

    switch (rec.material) {

    case UNLIT: {
        return rec.face_color;

    }

    case DIFFUSE: { // Diffuse
        EiVector3d w = rec.normal_surface;

        EiVector3d u =
            ((fabs(w.x()) > 0.1 ? EiVector3d(0,1,0) : EiVector3d(1,0,0))
            .cross(w)).normalized();

        EiVector3d v = w.cross(u);

        double r1 = 2 * M_PI * ((double)rand() / RAND_MAX);
        double r2 = (double)rand() / RAND_MAX;
        double r2s = sqrt(r2);

        EiVector3d d =
            (u * cos(r1) * r2s +
             v * sin(r1) * r2s +
             w * sqrt(1 - r2)).normalized();

        Ray ray_new;
        ray_new.origin = rec.point_intersection;
        ray_new.direction = d;

        return emitted + albedo.cwiseProduct(
            return_ray_color_new(ray_new, TLAS, depth + 1)
        );
    }

    case SPECULAR: {// Specular (mirror)
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(rec.normal_surface) * rec.normal_surface;
        
        Ray ray_new;
        ray_new.origin = rec.point_intersection;
        ray_new.direction = reflected;

        return emitted + albedo.cwiseProduct(
            return_ray_color_new(ray_new, TLAS, depth + 1)
        );
    }

    case REFRACTIVE: {// Refraction (dielectric)
        EiVector3d n = rec.normal_surface;
        EiVector3d nl = (ray.direction.dot(n) < 0) ? n : -n;
    
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(n) * n;
    
        Ray reflected_ray;
        reflected_ray.origin = rec.point_intersection;
        reflected_ray.direction = reflected;
    
        bool into = ray.direction.dot(nl) < 0; // entering or exiting
    
        double ri_surrounding = 1.0;   // air
        double ri_material = 1.5;   // glass
        double ri_ratio = into ? ri_surrounding / ri_material : ri_material / ri_surrounding;
    
        double dot_incidence = ray.direction.dot(nl);
        double cos_transmission2 = 1 - ri_ratio * ri_ratio * (1 - dot_incidence * dot_incidence);
    
        // Total internal reflection
        if (cos_transmission2 < 0) {
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflected_ray, TLAS, depth + 1)
            );
        }
    
        EiVector3d dir_transmission =
            (ray.direction * ri_ratio -
             n * ((into ? 1 : -1) * (dot_incidence * ri_ratio + sqrt(cos_transmission2)))).normalized();
    
        // Schlick approximation
        double a = ri_material - ri_surrounding;
        double b = ri_material + ri_surrounding;
        double R0 = (a * a) / (b * b);
    
        double c = 1 - (into ? -dot_incidence : dir_transmission.dot(n));
        double reflectance = R0 + (1 - R0) * c * c * c * c * c;
        double transmittance = 1 - reflectance;
    
        // Russian roulette between reflection and refraction
        double P = 0.25 + 0.5 * reflectance;
        double P_reflect = reflectance / P;
        double P_transmit = transmittance / (1 - P);
    
        if (depth > 2) {
            if ((double)rand() / RAND_MAX < P) {
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(reflected_ray, TLAS, depth + 1) * P_reflect
                );
            } else {
                Ray refracted_ray;
                refracted_ray.origin = rec.point_intersection;
                refracted_ray.direction = dir_transmission;
    
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(refracted_ray, TLAS, depth + 1) * P_transmit
                );
            }
        } else {
            Ray refracted_ray;
            refracted_ray.origin = rec.point_intersection;
            refracted_ray.direction = dir_transmission;
    
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflected_ray, TLAS, depth + 1) * reflectance +
                return_ray_color_new(refracted_ray, TLAS, depth + 1) * transmittance
            );
        }
    }
    }

    return emitted;
}
*/

/*
// Original, no-shading function
EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS) {

    HitRecord intersection_record; // Create HitRecord struct
    // Look for intersection
    IntersectionOutput intersection;
    intersect_TLAS(ray, TLAS, intersection, intersection_record);

    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        //std::cout << "Coloring..." << std::endl;
        set_face_normal(ray, intersection_record.normal_surface);
        // Color interpolated for a triangle
        //return intersection_record.elem_interp_coords(0) * intersection_record.face_color + intersection_record.elem_interp_coords(1) * intersection_record.face_color + intersection_record.elem_interp_coords(2) * intersection_record.face_color;
        return intersection_record.face_color; // To test quads without any special coloring for now
    }
    // Blue sky gradient
    return ray_blue_sky(ray);
}
*/

void mock_ray_shoot(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath) {
    // Shoot a single mock ray to see what happens - helpful in debugging

    Ray mock_ray;
    //mock_ray.origin = EiVector3d(0.0, 0.0, 410.0); wedge tests
    //mock_ray.direction = EiVector3d(0.15963, -0.0311445, -0.986686);
    mock_ray.origin = EiVector3d(0.0, 0.0, 410.0); //normals tests
    mock_ray.direction = EiVector3d(0.0814686, 0.171913, -0.981738);
    EiVector3d pixel_color = EiVector3d::Zero();
    pixel_color += return_ray_color_stack(mock_ray, scene_ri, TLAS);
    std::cerr << "Final color: " << pixel_color.x() << ", " << pixel_color.y() << ", " << pixel_color.z() << std::endl;
}