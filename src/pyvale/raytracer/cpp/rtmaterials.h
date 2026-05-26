// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <vector>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"
#include "rtmathutils.h"

static constexpr int INTERIOR_LIST_MAX = 10; // Size of the interior list. Should be sufficient unless user  embeds many volumes within one another 

// Struct size 8 + 2 x 4 = 16 bytes
struct InteriorEntry{
    EiVector3d absorption {EiVector3d::Zero()}; // (length unit)^-1, gives the tint/colour of the medium. (0.0, 0.0, 0.0) = clear
    double refractive_index {1.0003}; // Refractive index of the medium
    int priority {-1}; // Priority - tells us the ordering of nested volumes in the scene (e.g.,lower number = higher priority)
    int blas_idx {-1}; // Index of the corresponding BLAS, so we know which mesh is intersected - used to index into TLAS

    // Constructor
    InteriorEntry() = default;
    InteriorEntry(double ri_): refractive_index(ri_) {};
    InteriorEntry(EiVector3d absorption_, double ri_, int priority_, int blas_idx_) : absorption(absorption_), refractive_index(ri_), priority(priority_), blas_idx(blas_idx_) {};
};


/*
// Struct to store ray data in the stack-based shader
// Size: 48 + 24 + 16 x 10 + 8 + 4 x 2 =  248 bytes
struct RayState{
    Ray ray;
    EiVector3d accumulated_color {EiVector3d(1.0, 1.0, 1.0)}; // Accumulated multipliers (albedo, Fresnel terms, etc.)
    double outer_refractive_index {1.0003}; // Refractive index of the material where the ray originates; might be removed later, depending on how the interior list works
    int depth {0};

    //Constructors
    // First two are for the initial state with primary rays
    RayState(Ray ray_): ray(ray_) {};
    RayState(Ray ray_, double outer_refractive_index_): ray(ray_), outer_refractive_index(outer_refractive_index_) {};
    RayState(Ray ray_, EiVector3d accumulated_color_, double outer_refractive_index_, int depth_):
        ray(ray_),
        accumulated_color(accumulated_color_),
        outer_refractive_index(outer_refractive_index_),
        depth(depth_) {};
};

*/

// Struct to store ray data in the stack-based shader
// Size: 48 + 24 + 16 x 10 + 8 + 4 x 2 =  248 bytes
struct RayState{
    Ray ray;
    EiVector3d accumulated_color {EiVector3d(1.0, 1.0, 1.0)}; // Accumulated multipliers (albedo, Fresnel terms, etc.)
    std::array<InteriorEntry, INTERIOR_LIST_MAX> interior_list {}; // List of objects entered by ray
    double outer_refractive_index {1.0003}; // Refractive index of the material where the ray originates; might be removed later, depending on how the interior list works
    int depth {0};
    int interior_count {0}; // 0 => Ray is in the ambient medium (whatever fills the scene, with RI = scene_ri)

    //Constructors
    // First two are for the initial state with primary rays
    RayState(Ray ray_): ray(ray_) {};
    RayState(Ray ray_, double outer_refractive_index_): ray(ray_), outer_refractive_index(outer_refractive_index_) {};
    RayState(Ray ray_, EiVector3d accumulated_color_, std::array<InteriorEntry, INTERIOR_LIST_MAX> interior_list_, double outer_refractive_index_, int depth_, int interior_count_):
        ray(ray_),
        accumulated_color(accumulated_color_),
        interior_list(interior_list_),
        outer_refractive_index(outer_refractive_index_),
        depth(depth_),
        interior_count(interior_count_) {};
};


inline int interior_find(const InteriorEntry* interior_list, int count, int blas_idx){
    for(int i = 0; i < count; i++){
        if (interior_list[i].blas_idx == blas_idx){
            return i;
        }
    }
    return -1;
};

inline int interior_highest_priority_idx(const InteriorEntry* interior_list, int count){
    int highest_priority_idx = -1;
    int best_priority = std::numeric_limits<int>::min();
    for(int i = 0; i < count; i++){
        if (interior_list[i].priority > best_priority){
            best_priority = interior_list[i].priority;
            highest_priority_idx = i;
        }
    }
    return highest_priority_idx;
};

inline bool interior_toggle(InteriorEntry* interior_list,
    int& count, // We update this value, hence pass by reference
    int blas_idx,
    int priority,
    double refractive_index,
    const EiVector3d& absorption){
    // Add or remove the entry (toggle) for BLAS_ID.
    //Returns true if ray entered the object (add antry), false if exited (remove entry)

    int idx = interior_find(interior_list, count, blas_idx);
    if (idx >= 0){ // Exit: swap-erase
        interior_list[idx] = interior_list[count - 1];
        --count;
        return false;
    }
    else { // Enter
        interior_list[count].absorption = absorption;
        interior_list[count].refractive_index = refractive_index;
        interior_list[count].priority = priority;
        interior_list[count].blas_idx = blas_idx;
        ++count;
        return true;
    }
};

inline double find_top_ri(const InteriorEntry* interior_list,
    int interior_count,
    double fallback_ri){
    // Finds the top refractive index in the interior list
    // fallback_ri = scene_ri, which we set if the list is empty
    int current_highest_idx = interior_highest_priority_idx(interior_list, interior_count);
    return (current_highest_idx < 0) ? fallback_ri : interior_list[current_highest_idx].refractive_index;
};

inline EiVector3d find_top_absorption(const InteriorEntry* interior_list,
    int interior_count,
    const EiVector3d& fallback) { //
    // Finds the top refractive index in the interior list
    // fallback_ri = scene_ri
    int idx = interior_highest_priority_idx(interior_list, interior_count);
    return (idx < 0) ? fallback : interior_list[idx].absorption;
}

inline EiVector3d ray_blue_sky(const Ray& ray){
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}

void ray_diffuse(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_specular(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

    /*
void ray_refractive(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);
    */

void ray_unlit(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);

void ray_undefined(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color);


inline void apply_absorption(EiVector3d& accumulated_color, const EiVector3d& absorption, const double path_length){
    // Applies the absorption from the Beer Lambert law; path length is the travelled distance over some segment
    EiVector3d segment_transmission;
    segment_transmission << std::exp(-absorption(0) * path_length), std::exp(-absorption(1) * path_length), std::exp(-absorption(2) * path_length);
    // Include absorption into the transmitted/refracted throughput
    accumulated_color = accumulated_color.cwiseProduct(segment_transmission);
};

// Implementation with thickness for thin shell 
template <ObjectType object_type>
void ray_refractive(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){
    // Secondary ray may reflect or refract
    // Depends on: surface normal, refractive indices, sometimes wavelength
    //EiVector3d emitted = intersection_record.emission;
    EiVector3d attenuation(1.0, 1.0, 1.0); // Instead of albedo as for refractive materials, we absorb nothing and we want to make sure that is the case
    // Data stored in albedo can be used later for tinting, though, so we keep the interface
    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    // Pre-calculate the baseline for the next bounces
    // We split the values as in the Beer-Lambert implementation they will differ
    EiVector3d next_accumulated_color_reflected = current_state.accumulated_color.cwiseProduct(attenuation); 
    EiVector3d next_accumulated_color_refracted = next_accumulated_color_reflected;
    const EiVector3d p = intersection_record.point_intersection; // Point of intersection
    //const double OFFSET = OFFSET_SHADOW * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    const double OFFSET = std::numeric_limits<double>::epsilon() * 10.0 * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    const double spawned_ray_t_min = 1e-4 * std::max(1.0, intersection_record.point_intersection.norm()); // t_min for the spawned secondary rays 
    EiVector3d ray_direction = current_state.ray.direction;
   

    // Ensure normals always point against the incident ray - for the nested volume case, we can flip normals like this;
    // for regular case, we cannot as we need the dot product to determine bool into to figure out if ray enters or exits; here we use priorities
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal
    EiVector3d normal_shade = intersection_record.normal_shading; //  // Shading normal; use for Physics to dictate how light behaves

    // ri_from = refractive index of the highest priority entry in the original list OR scene_ri if empty (highest idx = -1 -> Empty)
    double scene_ri = current_state.outer_refractive_index;
    double ri_from = find_top_ri(&current_state.interior_list[0], current_state.interior_count, scene_ri);
    
    // Build the refraction ray's interior list = parent list toggled by the current object
    std::array<InteriorEntry, INTERIOR_LIST_MAX> refracted_list;
    int refracted_count = current_state.interior_count;
    for (int i = 0; i < current_state.interior_count; i++){
        refracted_list[i] = current_state.interior_list[i];
    }
   
    double ri_to = scene_ri; // Fallback assignment in case something breaks
    double thickness = 0.0; // Used for thin shell only

    if constexpr (object_type == ObjectType::SHELL) {
        // Thin shell: We don't toggle volumes as we stay in the same bulk medium
        // We use the shell RI only as an effective interface RI so the pane can still bend rays
        ri_to = intersection_record.refractive_index;
        thickness = intersection_record.thickness;
    }
    else if constexpr (object_type == ObjectType::SOLID){
        // Solid volume boundary: toggle interior membership
        const int hit_idx = intersection_record.hit_blas_idx;
        const int hit_priority = intersection_record.hit_blas_priority;
        const double hit_ri = intersection_record.refractive_index;
        const EiVector3d hit_absorption = intersection_record.face_color;

        interior_toggle(&refracted_list[0], refracted_count, hit_idx, hit_priority, hit_ri, hit_absorption);

        // Check whether ray enters or exits the object
        //If it is in the current interior list, the ray exits
        //const int existing_idx = interior_find(&current_state.interior_list[0], current_state.interior_count, hit_idx);

        // For reflected ray, the list is the copy of the parent list
        // 'to-index' = ri of the highest-priority entry in the REFRACTION ray's list, or scene_ri if empty
        // NB: when entering the hit object, this is just hit_ri if hit_priority is the new max; the formula handles both cases uniformly
        ri_to = find_top_ri(&refracted_list[0], refracted_count, scene_ri); 
    }


    double cos_theta_i = std::clamp(-ray_direction.dot(normal_shade), 0.0, 1.0); // To avoid floating point errors
    EiVector3d reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_shade; // Reflection direction
    reflected_dir.stableNormalize();

    if (reflected_dir.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
        reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_geo; 
    }

    double ri_ratio = ri_from / ri_to; // In the nested implementation, these are already correct by construction
    double sin2_theta_t = ri_ratio * ri_ratio * (1.0 - cos_theta_i * cos_theta_i); // Sin^2 of the transmission angle
    
    // Total internal reflection; should not occur for a thin shell
    if (sin2_theta_t > 1.0) {
        Ray reflected_ray;
        reflected_ray.origin = intersection_record.point_intersection + normal_geo * OFFSET; // Push secondary rays slightly off the surface to remove the shadow acne
        reflected_ray.direction = reflected_dir;
        reflected_ray.t_min = spawned_ray_t_min;
        
        stack.emplace_back(reflected_ray, next_accumulated_color_reflected, current_state.interior_list, scene_ri, current_state.depth + 1, current_state.interior_count);
        return;
    }
    
    // Schlick's approximation
    double a = ri_to - ri_from;
    double b = ri_to + ri_from;
    double R0 = (a * a) / (b * b);

    // Use cosine of the medium with the lower index of refraction
    double cos_theta_t = sqrt(1.0 - sin2_theta_t);
    double c = 1 - (ri_from <= ri_to ? cos_theta_i : cos_theta_t);

    double reflectance = 0.0;
    if constexpr (object_type == ObjectType::SHELL){
        // In a thin shell with thickness, we have double-interface
        // In ray-tracing we ignore interfecence for very thin films and use standard thin dielectric closed form from PBRT
        double R = R0 + (1 - R0) * (c * c * c * c * c);
        reflectance = (2.0 * R) / (1.0 + R); // Closed form double-interface for a shell with thickness
    }
    else if constexpr (object_type == ObjectType::SOLID){
        double reflectance = R0 + (1 - R0) * (c * c * c * c * c);
    }

    double transmittance = 1 - reflectance;

    // Define new rays
    Ray reflected_ray;
    reflected_ray.origin = intersection_record.point_intersection + normal_geo * OFFSET; // Push back into incident medium (i.e., off the surface)
    reflected_ray.direction = reflected_dir;
    reflected_ray.t_min = spawned_ray_t_min;

    Ray refracted_ray;
    refracted_ray.direction = ri_ratio * ray_direction + (ri_ratio * cos_theta_i - cos_theta_t) * normal_shade; // Transmitted/refracted direction
    refracted_ray.direction.stableNormalize();
    refracted_ray.t_min = spawned_ray_t_min;

    if constexpr (object_type == ObjectType::SHELL) {
        // Thin shell: do not move into a new volume; keep the stack unchanged
        // Offset slightly along the refracted direction to avoid immediately rehitting the same triangle.
        //refracted_ray.origin = intersection_record.point_intersection + refracted_ray.direction * OFFSET;
        EiVector3d in_slab_dir = refracted_ray.direction; // Refracted direction in-slab
        double cos_t_abs = std::max(1e-8, std::abs(in_slab_dir.dot(-normal_shade)));
        //std::cerr << "Cos t abs: " << cos_t_abs << std::endl;
        double thickness = intersection_record.thickness;
        double path_in_slab = thickness / cos_t_abs; // Distance travelled inside the shell with given thickness
        //std::cerr << "Path in slab: " << path_in_slab << std::endl;
        EiVector3d exit_point = intersection_record.point_intersection + in_slab_dir * path_in_slab;
        //std::cerr << "Exit point: " << exit_point << std::endl;
        
        // Beer-Lambert law for the slab traversal. We consider it per channel
        const EiVector3d absorption = intersection_record.face_color; // sigma_a in Beer-Lambert law used for volumetric absorption to determine the tint
        apply_absorption(next_accumulated_color_refracted, absorption, path_in_slab);
        refracted_ray.origin = exit_point - normal_geo * OFFSET;
        refracted_ray.direction = ray_direction; // Parallel slabs cancel angular deflection, so the ougoing direction = incident direction; already normalised at creation

    }
    else if constexpr (object_type == ObjectType::SOLID){
         // Solid volume: push into the transmitted medium
        refracted_ray.origin = intersection_record.point_intersection - normal_geo * OFFSET; // Push forward into new medium (i.e., into the surface)
    }

    // Russian roulette between reflection and refraction
    if (current_state.depth > 2) {
        double P = 0.25 + 0.5 * reflectance; // Reflection's chance of surviving
        if (random_double() < P){ // Note: for multi-threading this will have to be replaced with thread_local generator
        //if ((double)rand() / RAND_MAX < P) { // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
            // Reflection works the same way for shells and solids
            double P_reflect = reflectance / P; // Adjust original reflectance based on P
            stack.emplace_back(reflected_ray, next_accumulated_color_reflected * P_reflect, current_state.interior_list, scene_ri, current_state.depth + 1, current_state.interior_count);
            return;
        }
        else {
            double P_transmit = transmittance / (1.0 - P); // Adjust original transmittance based on P
            if constexpr (object_type == ObjectType::SHELL) {
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit, current_state.interior_list,
                    scene_ri, current_state.depth + 1, current_state.interior_count);
            }
            else if (object_type == ObjectType::SOLID){
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit, refracted_list,
                    scene_ri, current_state.depth + 1, refracted_count);
            }
            return;
        }
    } 
    else {
        // Push both rays
        stack.emplace_back(reflected_ray, next_accumulated_color_reflected * reflectance, current_state.interior_list, scene_ri, current_state.depth + 1, current_state.interior_count);
        if constexpr (object_type == ObjectType::SHELL) {
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance, current_state.interior_list,
                scene_ri, current_state.depth + 1, current_state.interior_count);
        }
        else if constexpr (object_type == ObjectType::SOLID){
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance, refracted_list,
                scene_ri, current_state.depth + 1, refracted_count);
        }
        return;
    }
}