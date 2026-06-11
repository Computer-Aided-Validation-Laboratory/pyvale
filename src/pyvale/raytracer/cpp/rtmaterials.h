// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTMATERIALS_H
#define RTMATERIALS_H

// STD header files
#include <vector>
#include <cstdint>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"
#include "rtmathutils.h"

// Base offset used for the t_min of of spawned rays. Used to reduce self-intersections when spawning secondary rays from a surface.
static constexpr double SPAWNED_T_MIN_BASE = 1e-7; 

// ================================================================================
// Interior list for nested dielectrics
// ================================================================================

// Use uint8_t for everything related as we would need 255 nested meshes to exceed this quantity. Highly unlikely to occur
// Size of the interior list. Should be sufficient unless user embeds many volumes within one another
static constexpr uint8_t INTERIOR_LIST_MAX = 10;


/**
 * @brief Stores one interior medium entry for nested dielectric tracking.
 * 
 * Contains the optical properties and scene identifiers of one medium currently
 * occupied by the ray.
 */
// Struct size: 4 x 8 + 2 x 4 = 40 bytes
struct InteriorEntry{
    EiVector3d absorption {EiVector3d::Zero()}; // Absorption coefficient of the medium (sigma_a) in (length unit)^-1. Gives the tint/colour of the medium; (0.0, 0.0, 0.0) = clear
    double refractive_index {1.0003}; // Refractive index of the medium
    int priority {-1}; // Nesting priority - tells us the ordering of nested volumes in the scene (e.g., higher number = more nested)
    int blas_idx {-1}; // Index of the corresponding BLAS, so we know which mesh is intersected - used to index into TLAS

    // Constructors
    InteriorEntry() = default;
    /**
     * @brief Constructs an interior entry with only a refractive index.
     * 
     * @param[in] ri_ (double) Refractive index of the medium
     */
    InteriorEntry(double ri_): refractive_index(ri_) {};
    /**
     * @brief Constructs a fully specified interior entry.
     * 
     * @param[in] absorption_ (EiVector3d) Absorption coefficient of the medium
     * @param[in] ri_ (double) Refractive index of the medium
     * @param[in] priority_ (int) Nesting priority of the medium
     * @param[in] blas_idx_ (int) BLAS index of the corresponding object
     */
    InteriorEntry(EiVector3d absorption_, double ri_, int priority_, int blas_idx_) : absorption(absorption_), refractive_index(ri_), priority(priority_), blas_idx(blas_idx_) {};
};

/**
 * @brief Finds an interior entry by BLAS index.
 * 
 * Searches the active interior list for an entry corresponding to the given BLAS.
 * 
 * @param[in] interior_list (const InteriorEntry*) Pointer to the interior list
 * @param[in] count (uint8_t) Number of active entries in the list
 * @param[in] blas_idx (int) BLAS index to search for
 * 
 * @return (int) Index of the matching entry, or -1 if not found.
 */
inline int interior_find(const InteriorEntry* interior_list, uint8_t count, int blas_idx){
    for(int i = 0; i < count; i++){
        if (interior_list[i].blas_idx == blas_idx){
            return i;
        }
    }
    return -1;
};

/**
 * @brief Finds the index of the highest-priority active interior entry.
 * 
 * Used to determine which medium currently dominates the ray state when
 * multiple nested dielectric volumes are present.
 * 
 * @param[in] interior_list (const InteriorEntry*) Pointer to the interior list
 * @param[in] count (uint8_t) Number of active entries in the list
 * 
 * @return (int) Index of the highest-priority entry, or -1 if the list is empty.
 */
inline int interior_highest_priority_idx(const InteriorEntry* interior_list,
    uint8_t count){

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

/**
 * @brief Finds the refractive index of the top active interior medium.
 * 
 * If the interior list is empty, returns the fallback refractive index,
 * typically that of the ambient scene medium (scene_ri).
 * 
 * @param[in] interior_list (const InteriorEntry*) Pointer to the interior list
 * @param[in] interior_count (uint8_t) Number of active entries in the list
 * @param[in] fallback_ri (double) Refractive index to return if the list is empty
 * 
 * @return (double) Refractive index of the top active medium, or fallback_ri if none exists.
 */
inline double interior_top_ri(const InteriorEntry* interior_list,
    uint8_t interior_count,
    double fallback_ri){

    int current_highest_idx = interior_highest_priority_idx(interior_list, interior_count);
    return (current_highest_idx < 0) ? fallback_ri : interior_list[current_highest_idx].refractive_index;
};

/**
 * @brief Finds the absorption coefficient of the top active interior medium.
 * 
 * If the interior list is empty, returns zero absorption.
 * 
 * @param[in] interior_list (const InteriorEntry*) Pointer to the interior list
 * @param[in] interior_count (uint8_t) Number of active entries in the list
 * 
 * @return (EiVector3d) Absorption coefficient of the top active medium, or zero if none exists.
 */
inline EiVector3d interior_top_absorption(const InteriorEntry* interior_list,
    uint8_t interior_count) { //

    EiVector3d fallback(0.0, 0.0, 0.0); 
    int idx = interior_highest_priority_idx(interior_list, interior_count);
    return (idx < 0) ? fallback : interior_list[idx].absorption;
}

/**
 * @brief Toggles membership of a BLAS in the active interior list.
 * 
 * If the BLAS (with given index) is already present, it is removed, corresponding to the ray
 * exiting the object.
 * Otherwise, a new entry is added, corresponding to the
 * ray entering the object.
 * 
 * @param[in,out] interior_list (InteriorEntry*) Pointer to the interior list
 * @param[in,out] count (uint8_t&) Number of active entries in the list
 * @param[in] blas_idx (int) BLAS index of the object
 * @param[in] priority (int) Nesting priority of the object
 * @param[in] refractive_index (double) Refractive index of the object
 * @param[in] absorption (const EiVector3d&) Absorption coefficient of the object
 * 
 * @return (bool) True if the ray entered the object, false if it exited.
 */
inline bool interior_toggle(InteriorEntry* interior_list,
    uint8_t& count, // We update this value, hence pass by reference
    int blas_idx,
    int priority,
    double refractive_index,
    const EiVector3d& absorption){


    int idx = interior_find(interior_list, count, blas_idx);
    if (idx >= 0){ // Exit: swap-erase
        interior_list[idx] = interior_list[count - 1];
        --count;
        return false;
    }
    else { // Enter
        interior_list[count] = InteriorEntry(absorption, refractive_index, priority, blas_idx);
        //interior_list[count].absorption = absorption;
        //interior_list[count].refractive_index = refractive_index;
        //interior_list[count].priority = priority;
        //interior_list[count].blas_idx = blas_idx;
        ++count;
        return true;
    }
};

// ================================================================================
// Ray state struct
// ================================================================================

/**
 * @brief Stores the full tracing state of one ray in the stack-based renderer.
 * 
 * Includes the ray geometry, accumulated payload, dielectric interior state,
 * recursion depth, and ambient medium information.
 */
// Size: 48 + 24 + 16 x 10 + 8 + 2 + 1 =  243 bytes
struct RayState{
    Ray ray; // Current ray
    EiVector3d accumulated_color {EiVector3d(1.0, 1.0, 1.0)}; // Accumulated payload multipliers (albedo, Fresnel terms, etc.)
    std::array<InteriorEntry, INTERIOR_LIST_MAX> interior_list {}; // List of objects entered by ray
    double scene_ri {1.0003}; // Refractive index of the ambient medium; without nested volumes, use this to store the ri_from
    uint16_t depth {0}; // Current ray depth
    // Even without recursion, most ray tracers don't seem to have more than 50 bounces of depth, so we are being very generous here (max. value of 65535)
    uint8_t interior_count {0}; // Number of active entries in interior_list. 0 => Ray is in the ambient medium (whatever fills the scene, with RI = scene_ri)

    //Constructors
    // First two are for the initial state with primary rays
    /**
     * @brief Constructs a ray state from a primary ray
     * 
     * @param[in] ray_ (Ray) Input ray.
     */
    RayState(Ray ray_): ray(ray_) {};
    /**
     * @brief Constructs a ray state from a primary ray and ambient refractive index.
     * 
     * @param[in] ray_ (Ray) Input ray
     * @param[in] scene_ri_ (double) Refractive index of the ambient medium
     */
    RayState(Ray ray_, double scene_ri_): ray(ray_), scene_ri(scene_ri_) {};
    /**
     * @brief Constructs a fully specified ray state.
     * 
     * @param[in] ray_ (Ray) Current ray
     * @param[in] accumulated_color_ (EiVector3d) Current accumulated payload
     * @param[in] interior_list_ (std::array<InteriorEntry, INTERIOR_LIST_MAX>) Active interior media list
     * @param[in] scene_ri_ (double) Refractive index of the ambient medium
     * @param[in] depth_ (uint16_t) Current ray depth
     * @param[in] interior_count_ (uint8_t) Number of active entries in interior_list
     */
    RayState(Ray ray_, EiVector3d accumulated_color_, std::array<InteriorEntry, INTERIOR_LIST_MAX> interior_list_, double scene_ri_, uint16_t depth_, uint8_t interior_count_):
        ray(ray_),
        accumulated_color(accumulated_color_),
        interior_list(interior_list_),
        scene_ri(scene_ri_),
        depth(depth_),
        interior_count(interior_count_) {};
};

// ================================================================================
// Material functions
// ================================================================================

/**
 * @brief Returns a procedural sky color for a ray direction.
 * 
 * Produces a simple vertical white-to-blue gradient based on the y-component
 * of the ray direction.
 * 
 * @param[in] ray (const Ray&) Input ray
 * 
 * @return (EiVector3d) RGB sky color corresponding to the ray direction.
 */
inline EiVector3d ray_blue_sky(const Ray& ray){
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}

/**
 * @brief Computes diffuse material response and spawns a scattered ray.
 * 
 * Adds emitted radiance, updates throughput with albedo, and generates a
 * cosine-weighted random hemisphere direction around the shading normal.
 * 
 * @param[in] current_state (const RayState&) Current ray state
 * @param[in,out] intersection_record (HitRecord&) Intersection data for the hit point
 * @param[in] albedo (const EiVector3d&) Surface albedo
 * @param[in,out] stack (std::vector<RayState>&) Ray stack to which spawned rays are appended
 * @param[in,out] total_color (EiVector3d&) Accumulated output color
 * @param[in] offset (double) Spatial offset used when spawning the secondary ray
 */
void ray_diffuse(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset);

/**
 * @brief Computes specular material response and spawns a reflected ray.
 * 
 * Adds emitted radiance, updates throughput with albedo, and reflects the
 * incident ray about the shading normal, with a fallback to the geometric
 * normal if needed.
 * 
 * @param[in] current_state (const RayState&) Current ray state
 * @param[in,out] intersection_record (HitRecord&) Intersection data for the hit point
 * @param[in] albedo (const EiVector3d&) Surface albedo
 * @param[in,out] stack (std::vector<RayState>&) Ray stack to which spawned rays are appended
 * @param[in,out] total_color (EiVector3d&) Accumulated output color
 * @param[in] offset (double) Spatial offset used when spawning the secondary ray
 */
void ray_specular(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset);

/**
 * @brief Computes response for an unlit material.
 * 
 * Terminates the path at the current hit and adds the face color directly
 * to the accumulated output.
 * 
 * @param[in] current_state (const RayState&) Current ray state
 * @param[in,out] intersection_record (HitRecord&) Intersection data for the hit point
 * @param[in] albedo (const EiVector3d&) Unused parameter, present for signature consistency
 * @param[in,out] stack (std::vector<RayState>&) Unused ray stack, present for signature consistency
 * @param[in,out] total_color (EiVector3d&) Accumulated output color
 * @param[in] offset (double) Unused parameter, present for signature consistency
 */
void ray_unlit(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset);

/**
 * @brief Fallback material response for undefined materials.
 * 
 * Terminates the path and adds a procedural sky color based on the current
 * ray direction.
 * 
 * @param[in] current_state (const RayState&) Current ray state
 * @param[in,out] intersection_record (HitRecord&) Intersection data for the hit point
 * @param[in] albedo (const EiVector3d&) Unused parameter, present for signature consistency
 * @param[in,out] stack (std::vector<RayState>&) Unused ray stack, present for signature consistency
 * @param[in,out] total_color (EiVector3d&) Accumulated output color
 * @param[in] offset (double) Unused parameter, present for signature consistency
 */
void ray_undefined(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset);

// ================================================================================
// Everything for refractive materials
// ================================================================================

/**
 * @brief Applies Beer-Lambert absorption to the accumulated payload.
 * 
 * Computes per-channel transmission over the given path length using the
 * supplied absorption coefficient and multiplies it into accumulated_color.
 * 
 * @param[in,out] accumulated_color (EiVector3d&) Throughput to be attenuated
 * @param[in] absorption (const EiVector3d&) Absorption coefficient per color channel
 * @param[in] path_length (double) Travelled distance through the absorbing medium
 */
inline void apply_absorption(EiVector3d& accumulated_color, const EiVector3d& absorption, const double path_length){
    // Path length is the travelled distance over some segment
    EiVector3d segment_transmission;
    segment_transmission << std::exp(-absorption(0) * path_length), std::exp(-absorption(1) * path_length), std::exp(-absorption(2) * path_length);
    // Include absorption into the transmitted/refracted throughput
    accumulated_color = accumulated_color.cwiseProduct(segment_transmission);
};

/**
 * @brief Calculates the reflectance using unpolarised Fresnel equations.
 * 
 * More accurate than Schlick's approximation, but slower.
 * 
 * @param[in] ri_from (const double) Refractive index of the incident medium
 * @param[in] ri_to (const double) Refractive index of the refracted medium
 * @param[in] cos_theta_i (const double) Angle of incidence
 * @param[in] cos_theta_t (const double) Angle of refraction
 * @return (double) Reflectance (for solids)
 */
static inline double reflectance_fresnel(const double ri_from,
    const double ri_to,
    const double cos_theta_i,
    const double cos_theta_t){
    // Unpolarised Fresnel reflectance; more accurate, but slower

    const double n1_cti = ri_from * cos_theta_i;
    const double n2_ctt = ri_to * cos_theta_t;
    const double n1_ctt = ri_from * cos_theta_t;
    const double n2_cti = ri_to * cos_theta_i;

    const double rs_num = n1_cti - n2_ctt;
    const double rs_den = n1_cti + n2_ctt;
    const double rp_num = n1_ctt - n2_cti;
    const double rp_den = n1_ctt + n2_cti;

    const double Rs = (rs_num / rs_den) * (rs_num / rs_den);
    const double Rp = (rp_num / rp_den) * (rp_num / rp_den);
    // Reflectance for solids; a value we use for shells 
    return 0.5 * (Rs + Rp);
}

/**
 * @brief Calculates the reflectance using Schlick's approximation.
 * 
 * Faster than Fresnel equations, not as accurate.
 * 
 * @param[in] ri_from (const double) Refractive index of the incident medium
 * @param[in] ri_to (const double) Refractive index of the refracted medium
 * @param[in] cos_theta_i (const double) Angle of incidence
 * @param[in] cos_theta_t (const double) Angle of refraction
 * @return (double) Reflectance (for solids)
 */
static inline double reflectance_schlick(const double ri_from,
    const double ri_to,
    const double cos_theta_i,
    const double cos_theta_t){
    // Schlick's approximation
    const double a = ri_to - ri_from;
    const double b = ri_to + ri_from;
    const double R0 = (a * a) / (b * b);
    const double c = 1 - (ri_from <= ri_to ? cos_theta_i : cos_theta_t); // Again, in nested dielectrics this replaces the "if into" check
    // Reflectance for solids; a value we use for shells 
    return R0 + (1 - R0) * (c * c * c * c * c);
}

/**
 * @brief Computes refractive material response and spawns reflected and/or transmitted rays.
 * 
 * Handles both solid dielectric volumes and thin shells. Supports nested
 * dielectric tracking, Schlick Fresnel reflectance, total internal reflection,
 * Beer-Lambert absorption, and Russian roulette path termination.
 * 
 * @tparam object_type (ObjectType) Type of refractive object, e.g. SOLID or SHELL.
 * 
 * @param[in] current_state (const RayState&) Current ray state
 * @param[in,out] intersection_record (HitRecord&) Intersection data for the hit point
 * @param[in] albedo (const EiVector3d&) Currently unused input kept for interface consistency
 * @param[in,out] stack (std::vector<RayState>&) Ray stack to which spawned rays are appended
 * @param[in,out] total_color (EiVector3d&) Accumulated output color
 * @param[in] offset (double) Spatial offset used when spawning secondary rays
 */
template <ObjectType object_type>
void ray_refractive(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset){
    // Secondary ray may reflect or refract
    // Depends on: surface normal, refractive indices, sometimes wavelength (TO DO: ADD WAVELENGTHS)
    // This is a bit convoluted with shell/solid split and nested dielectrics, so this is sectioned

    //---------------------------------------------------------------------------------
    // 1. Retrieve data and pre-calculate baseline for the next bounces
    //---------------------------------------------------------------------------------

    EiVector3d attenuation(1.0, 1.0, 1.0); // Instead of albedo as for refractive materials, we absorb nothing and we want to make sure that is the case
    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection

    // We split the colour values as in the Beer-Lambert implementation they will differ for reflected and refracted rays
    EiVector3d next_accumulated_color_reflected = current_state.accumulated_color.cwiseProduct(attenuation); 
    EiVector3d next_accumulated_color_refracted = next_accumulated_color_reflected;
    const EiVector3d p_intersect = intersection_record.point_intersection; // Point of intersection
    const double spawned_ray_t_min = SPAWNED_T_MIN_BASE * std::max(1.0, p_intersect.norm()); // t_min for the spawned secondary rays 
    EiVector3d ray_direction = current_state.ray.direction;
    uint8_t interior_count = current_state.interior_count;
    std::array<InteriorEntry, INTERIOR_LIST_MAX> interior_list = current_state.interior_list;

    // Ensure normals always point against the incident ray - for the nested volume case, we can flip normals just like this - we keep track of what is being exited/entered
    // Otherwise, this needed to be handled with the dot product to determine bool into to figure out if ray enters or exits
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal
    EiVector3d normal_shade = intersection_record.normal_shading; // Shading normal; use for physics to dictate how light behaves
    
    // Build the refracted ray's interior list = parent list toggled by the current object (we note that we enter/exit a new volume)
    // Reflected ray uses the parent's interior list (we stay in the same volume)
    std::array<InteriorEntry, INTERIOR_LIST_MAX> refracted_list;
    uint8_t refracted_count = interior_count;
    for (int i = 0; i < interior_count; i++){
        refracted_list[i] = interior_list[i];
    }

    //---------------------------------------------------------------------------------
    // 2. Handle refractive indices
    //---------------------------------------------------------------------------------
   
    // ri_from = RI of the medium the ray is currently in = top of CURRENT list
    // top = of the highest priority entry in the original list OR scene_ri if empty (highest idx = -1 => empty)
    double scene_ri = current_state.scene_ri;
    double ri_from = interior_top_ri(&interior_list[0], current_state.interior_count, scene_ri);

    // ri_to = RI of the medium the refracted ray will be in; value depends if it's shell or solid
    double ri_to = scene_ri; // Fallback assignment in case something breaks

    if constexpr (object_type == ObjectType::SHELL) {
        // Thin shell: Don't toggle volumes as we stay in the same bulk medium
        // Use the shell RI only as the effective interface RI so the pane can still bend rays
        ri_to = intersection_record.refractive_index;
    }
    else if constexpr (object_type == ObjectType::SOLID){
        // Solid volume boundary: toggle interior membership
        const int hit_idx = intersection_record.hit_blas_idx;
        const int hit_priority = intersection_record.hit_blas_priority;
        const double hit_ri = intersection_record.refractive_index;
        const EiVector3d hit_absorption = intersection_record.face_color;

        interior_toggle(&refracted_list[0], refracted_count, hit_idx, hit_priority, hit_ri, hit_absorption);

        // ri_to = RI of the highest-priority entry in the REFRACTED ray's list, or scene_ri if empty
        // When entering the hit object, this is just hit_ri if hit_priority is the new max; the formula handles both cases uniformly
        ri_to = interior_top_ri(&refracted_list[0], refracted_count, scene_ri); 

        // Beer-Lambert - this technically shouldn't be here with nested dielectrics, but keeping it here in case
        /*
        // Check if we entered or left the volume (-1 => not in the list => we are entering)
        bool into = interior_find(&interior_list[0], interior_count, hit_idx) >= 0 ? false : true;
        if (!into){
            // Ray is exiting: it has travelled intersection_record.t through the current medium
            const EiVector3d current_absorption = interior_top_absorption(
                &interior_list[0], interior_count);
            const bool has_absorption = current_absorption.x() > 0.0
                                     || current_absorption.y() > 0.0
                                     || current_absorption.z() > 0.0;
            if (has_absorption) {
                apply_absorption(next_accumulated_color_reflected, current_absorption, intersection_record.t);
                apply_absorption(next_accumulated_color_refracted, current_absorption, intersection_record.t);
            }
        }*/
    }

    //---------------------------------------------------------------------------------
    // 3. Calculate Fresnel reflections
    //---------------------------------------------------------------------------------

    double cos_theta_i = std::clamp(-ray_direction.dot(normal_shade), 0.0, 1.0); // To avoid floating point errors
    EiVector3d reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_shade; 
    reflected_dir.stableNormalize();
    
    if (reflected_dir.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
       double cos_theta_geo = std::max(0.0, -ray_direction.dot(normal_geo));
        reflected_dir = ray_direction + 2.0 * cos_theta_geo * normal_geo;
        reflected_dir.stableNormalize();
    }

    double ri_ratio = ri_from / ri_to; // In the nested implementation, these are already correct by construction; no need for "if into" check
    double sin2_theta_t = ri_ratio * ri_ratio * (1.0 - cos_theta_i * cos_theta_i); // Sin^2 of the transmission angle
    
    // Check for Total Internal Reflection (TIR)
    // Only for solid; in shell apprroximation it should never occur
    if constexpr (object_type == ObjectType::SOLID){
        if (sin2_theta_t > 1.0) {
            Ray reflected_ray;
            reflected_ray.origin = p_intersect + normal_geo * offset; // Push secondary rays slightly off the surface to remove the shadow acne
            reflected_ray.direction = reflected_dir;
            reflected_ray.t_min = spawned_ray_t_min;
            
            // TIR — reflected ray travels in the SAME medium, so keep the parent list
            stack.emplace_back(reflected_ray, next_accumulated_color_reflected,
                interior_list, scene_ri, current_state.depth + 1, interior_count);
            return;
        }
    }
    // Use cosine of the medium with the lower index of refraction
    double cos_theta_t = 0.0;
    if constexpr (object_type == ObjectType::SHELL){
        // TIR does not occur for SHELLs so we skip the return path when sin2_theta_t > 1.0 (possible close to grazing angles)
        // In this case we'd have 1.0 - sin2_theta_t < 0 => Sqrt of that is imaginary => We clamp it
        cos_theta_t = std::sqrt(std::max(0.0, 1.0 - sin2_theta_t));
    }
    else if constexpr (object_type == ObjectType::SOLID){
        // No need to clamp - we won't reach this point if sin2_theta_t > 1.0
        cos_theta_t = std::sqrt(1.0 - sin2_theta_t);
    }

    // TO DO:
    // Add a switch in Python to let the users decide between Fresnel (accuracy) and speed (Schlick)

    // Calculate reflectance (for solids; for shells, we need to adjust it)
    //double reflectance = reflectance_schlick(ri_from, ri_to, cos_theta_i, cos_theta_t); 
    double reflectance = reflectance_fresnel(ri_from, ri_to, cos_theta_i, cos_theta_t); 
    if constexpr (object_type == ObjectType::SHELL){
    // Thin shell with thickness => Double-interface
    // In ray-tracing we ignore interfecence for very thin films and use closed form double-interface dielectric from PBRT
    reflectance = (2.0 * reflectance) / (1.0 + reflectance); // reflectance
    }  
        
    double transmittance = 1 - reflectance;

    //---------------------------------------------------------------------------------
    // 4. Define new rays
    //---------------------------------------------------------------------------------
    Ray reflected_ray;
    reflected_ray.origin = p_intersect + normal_geo * offset; // Push back into incident medium (i.e., off the surface)
    reflected_ray.direction = reflected_dir;
    reflected_ray.t_min = spawned_ray_t_min;

    Ray refracted_ray;
    refracted_ray.direction = ri_ratio * ray_direction + (ri_ratio * cos_theta_i - cos_theta_t) * normal_shade;
    refracted_ray.direction.stableNormalize();
    refracted_ray.t_min = spawned_ray_t_min;

    // Set the origin of the refracted ray
    if constexpr (object_type == ObjectType::SHELL) {
        // Thin shell: do not move into a new volume; keep the stack unchanged
        // Offset slightly along the refracted direction to avoid immediately rehitting the same mesh element
        EiVector3d in_slab_dir = refracted_ray.direction; // Refracted direction in-slab
        double cos_t_abs = std::abs(in_slab_dir.dot(-normal_shade));
        // At grazing angles the slab model breaks down — we cap the path to avoid launching exit_point arbitrarily far from the surface
        constexpr double COS_SHELL_MIN = 0.01; // ~89.4 degrees — beyond this, slab approximation is invalid
        if (cos_t_abs < COS_SHELL_MIN) {
            // Near-grazing: skip slab offset entirely, treat as zero-thickness
            refracted_ray.origin = p_intersect - normal_geo * offset;
            // Some engines offset along the refracted direction - in my tests, this was worse, but keeping it here as an option
            // refracted_ray.origin = p_intersect + refracted_ray.direction * offset;
            // Still apply absorption with a capped path to avoid energy injection
            double path_in_slab = intersection_record.thickness / COS_SHELL_MIN;
            apply_absorption(next_accumulated_color_refracted, intersection_record.face_color, path_in_slab);
        } else {
            double path_in_slab = intersection_record.thickness / cos_t_abs;
            // Here face_color is sigma_a (absorption coeff.) in Beer-Lambert law used for volumetric absorption to determine the tint
            apply_absorption(next_accumulated_color_refracted, intersection_record.face_color, path_in_slab); 
            EiVector3d exit_point = p_intersect + in_slab_dir * path_in_slab;
            refracted_ray.origin = exit_point - normal_geo * offset;
            // Some engines offset along the refracted direction - in my tests, this was worse, but keeping it here as an option
            //refracted_ray.origin = exit_point + refracted_ray.direction * offset;
        }
        refracted_ray.direction = ray_direction; // Parallel slabs cancel angular deflection, so the ougoing direction = incident direction; already normalised at creation
    }
    else if constexpr (object_type == ObjectType::SOLID){
         // Solid volume: push into the transmitted medium
        refracted_ray.origin = p_intersect - normal_geo * offset; // Push forward into new medium (i.e., into the surface)
        // Some engines offset along the refracted direction - in my tests, this was worse, but keeping it here as an option
        //refracted_ray.origin = intersection_record.point_intersection + refracted_ray.direction * offset;
    }

    //---------------------------------------------------------------------------------
    // 5. Russian roulette (depth > 2) or push both (depth <= 2)
    //---------------------------------------------------------------------------------
    if (current_state.depth > 2) {
        //double P = 0.25 + 0.5 * reflectance; // <- This was giving nonsensical and overshot ray energy whenver reflectance was >= 0.5 (visible when we had multiple bounces)
        double P = std::clamp(reflectance, 0.1, 0.9); // Reflection's chance of surviving; 0.1 to prevent division by 0, 0.9 to give transmission a chance to survive
        if (random_double() < P){ 
            // Reflection works the same way for shells and solids
            double P_reflect = reflectance / P; // Adjust original reflectance based on P
            stack.emplace_back(reflected_ray, next_accumulated_color_reflected * P_reflect, interior_list,
                scene_ri, current_state.depth + 1, interior_count);
            return;
        }
        else {
            double P_transmit = transmittance / (1.0 - P); // Adjust original transmittance based on P
            if constexpr (object_type == ObjectType::SHELL) {
                // Shell: medium unchanged, parent list and RI preserved
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit,
                    interior_list, scene_ri, current_state.depth + 1, interior_count);
            }
            else if constexpr (object_type == ObjectType::SOLID){
                // Solid: ray enters/exits a volume, use the toggled list
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit,
                    refracted_list, scene_ri, current_state.depth + 1, refracted_count);
            }
            return;
        }
    } 
    else { // Push both rays
        stack.emplace_back(reflected_ray, next_accumulated_color_reflected * reflectance, interior_list,
            scene_ri, current_state.depth + 1, interior_count);
        if constexpr (object_type == ObjectType::SHELL) {
            // Shell: medium unchanged, parent list and RI preserved
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance,
                interior_list, scene_ri, current_state.depth + 1, interior_count);
        }
        else if constexpr (object_type == ObjectType::SOLID){
            // Solid: ray enters/exits a volume, use the toggled list
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance,
                refracted_list, scene_ri, current_state.depth + 1, refracted_count);
        }
        return;
    }
}

// ================================================================================
// Previous versions of ray_refractive to help with debug/dev
// ================================================================================

// Implementation without nested dielectrics, i.e., pure Beer-Lambert (but we have InteriorEntry etc., because that was implemented first and the below was
// reverse engineered purely for troubleshooting)
/*
template <ObjectType object_type>
void ray_refractive(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color,
    const double offset){
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
    const double spawned_ray_t_min = SPAWNED_T_MIN_BASE * std::max(1.0, intersection_record.point_intersection.norm()); // t_min for the spawned secondary rays 
    EiVector3d ray_direction = current_state.ray.direction;
   
    
    bool into = ray_direction.dot(intersection_record.normal_surface) < 0;
    // Ensure normals always point against the incident ray
    intersection_record.align_normals();
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal
    normal_geo.stableNormalize();
    EiVector3d normal_shade = intersection_record.normal_shading; //  // Shading normal; use for Physics to dictate how light behaves
    normal_shade.stableNormalize();

    if (!into) {
        normal_geo = -normal_geo;
        normal_shade = -normal_shade;
    };
    

    std::array<InteriorEntry, INTERIOR_LIST_MAX> refracted_list;
    refracted_list = current_state.interior_list;
    uint8_t refracted_count = current_state.interior_count;
    double ri_from = current_state.scene_ri; 
    double ri_to = 1.0003; // Fallback assignment in case something breaks
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

        //interior_toggle(&refracted_list[0], refracted_count, hit_idx, hit_priority, hit_ri, hit_absorption);

        // Check whether ray enters or exits the object
        //If it is in the current interior list, the ray exits
        //const int existing_idx = interior_find(&current_state.interior_list[0], current_state.interior_count, hit_idx);

        // For reflected ray, the list is the copy of the parent list
        // to-index = ri of the highest-priority entry in the REFRACTION ray's list, or scene_ri if empty
        // NB: when entering the hit object, this is just hit_ri if hit_priority is the new max; the formula handles both cases uniformly
        //ri_to = interior_top_ri(&refracted_list[0], refracted_count, scene_ri); 
        ri_to = hit_ri;
    }


    double cos_theta_i = std::clamp(-ray_direction.dot(normal_shade), 0.0, 1.0); // To avoid floating point errors
    EiVector3d reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_shade; // Reflection direction
    reflected_dir.stableNormalize();
    
    if (reflected_dir.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
       double cos_theta_geo = std::max(0.0, -ray_direction.dot(normal_geo));
        reflected_dir = ray_direction + 2.0 * cos_theta_geo * normal_geo;
        reflected_dir.stableNormalize();
    }

    double ri_ratio = into ? ri_from / ri_to : ri_to / ri_from; 
    //double ri_ratio = ri_from / ri_to; // In the nested implementation, these are already correct by construction
    double sin2_theta_t = ri_ratio * ri_ratio * (1.0 - cos_theta_i * cos_theta_i); // Sin^2 of the transmission angle
    
    // Check for Total Internal Reflection (TIR) - only for solid; in shell apprroximation it should never occur
    if constexpr (object_type == ObjectType::SOLID){
        if (sin2_theta_t > 1.0) {
            Ray reflected_ray;
            reflected_ray.origin = intersection_record.point_intersection + normal_geo * offset; // Push secondary rays slightly off the surface to remove the shadow acne
            reflected_ray.direction = reflected_dir;
            reflected_ray.t_min = spawned_ray_t_min;
            
            stack.emplace_back(reflected_ray, next_accumulated_color_reflected, current_state.interior_list, ri_to, current_state.depth + 1, current_state.interior_count);
            return;
        }
    }

    // Schlick's approximation
    double a = ri_to - ri_from;
    double b = ri_to + ri_from;
    double R0 = (a * a) / (b * b);

    // Use cosine of the medium with the lower index of refraction
    double cos_theta_t = 0.0;
    if constexpr (object_type == ObjectType::SHELL){
        // TIR does not occur for SHELLs so we skip the return path when sin2_theta_t > 1.0 (possible close to grazing angles)
        // In this case we'd have 1.0 - sin2_theta_t < 0 => Sqrt of that is imaginary => We clamp it
        cos_theta_t = std::sqrt(std::max(0.0, 1.0 - sin2_theta_t));
    }
    else if constexpr (object_type == ObjectType::SOLID){
        // No need to clamp - we won't reach this point if sin2_theta_t > 1.0
        cos_theta_t = std::sqrt(1.0 - sin2_theta_t);
    }
    
    //double c = 1 - (ri_from <= ri_to ? cos_theta_i : cos_theta_t);
    double c = 1 - (into ? cos_theta_i : cos_theta_t);

    double reflectance = 0.0;
    if constexpr (object_type == ObjectType::SHELL){
        // In a thin shell with thickness, we have double-interface
        // In ray-tracing we ignore interfecence for very thin films and use standard thin dielectric closed form from PBRT
        double R = R0 + (1 - R0) * (c * c * c * c * c);
        reflectance = (2.0 * R) / (1.0 + R); // Closed form double-interface for a shell with thickness
    }
    else if constexpr (object_type == ObjectType::SOLID){
        reflectance = R0 + (1 - R0) * (c * c * c * c * c);
    }

    double transmittance = 1 - reflectance;

    // Define new rays
    Ray reflected_ray;
    reflected_ray.origin = intersection_record.point_intersection + normal_geo * offset; // Push back into incident medium (i.e., off the surface)
    reflected_ray.direction = reflected_dir;
    reflected_ray.t_min = spawned_ray_t_min;

    Ray refracted_ray;
    refracted_ray.direction = ri_ratio * ray_direction + (ri_ratio * cos_theta_i - cos_theta_t) * normal_shade; // Transmitted/refracted direction
    refracted_ray.direction.stableNormalize();
    refracted_ray.t_min = spawned_ray_t_min;


    if constexpr (object_type == ObjectType::SHELL) {
        // Thin shell: do not move into a new volume; keep the stack unchanged
        // Offset slightly along the refracted direction to avoid immediately rehitting the same triangle.
        //refracted_ray.origin = intersection_record.point_intersection + refracted_ray.direction * offset;
        EiVector3d in_slab_dir = refracted_ray.direction; // Refracted direction in-slab
        
        double cos_t_abs = std::abs(in_slab_dir.dot(-normal_shade));

        // At grazing angles the slab model breaks down — cap the path to avoid launching exit_point arbitrarily far from the surface
        constexpr double COS_SHELL_MIN = 0.01; // ~89.4 degrees — beyond this, slab approx invalid
        if (cos_t_abs < COS_SHELL_MIN) {
            // Near-grazing: skip slab offset entirely, treat as zero-thickness
            refracted_ray.origin = intersection_record.point_intersection - normal_geo * offset;
            refracted_ray.direction = ray_direction;
            // Still apply absorption with a capped path to avoid energy injection
            double path_in_slab = intersection_record.thickness / COS_SHELL_MIN;
            apply_absorption(next_accumulated_color_refracted, intersection_record.face_color, path_in_slab);
        } else {
            double path_in_slab = intersection_record.thickness / cos_t_abs;
            apply_absorption(next_accumulated_color_refracted, intersection_record.face_color, path_in_slab);
            EiVector3d exit_point = intersection_record.point_intersection + in_slab_dir * path_in_slab;
            refracted_ray.origin = exit_point - normal_geo * offset;
            refracted_ray.direction = ray_direction;
            double displacement = (exit_point - intersection_record.point_intersection).norm();

        }


    }
    else if constexpr (object_type == ObjectType::SOLID) {
        // Apply Beer-Lambert for the segment just traversed through this solid's interior.
        // This only has a physical meaning on the EXIT hit (ray leaving the medium), where intersection_record.t is the chord length through the material.
        // On the ENTRY hit, the ray hasn't yet travelled inside, so absorption = 0 (face_color should be zeroed on entry, or we gate on !into).
        if (!into) {
            // Ray is exiting — it has just travelled intersection_record.t through the solid
            const EiVector3d absorption = intersection_record.face_color;
            const bool has_absorption = absorption.x() > 0.0 || absorption.y() > 0.0 || absorption.z() > 0.0;
            if (has_absorption) {
                double path_length = intersection_record.t; // distance from entry point to this exit point
                apply_absorption(next_accumulated_color_reflected, absorption, path_length);
                apply_absorption(next_accumulated_color_refracted, absorption, path_length);
            }
        }

        refracted_ray.origin = intersection_record.point_intersection - normal_geo * offset;
    }

    constexpr double MAX_ENERGY = 10.0; // Tune to scene scale
    next_accumulated_color_refracted = next_accumulated_color_refracted.cwiseMin(EiVector3d(MAX_ENERGY, MAX_ENERGY, MAX_ENERGY));
    // Russian roulette between reflection and refraction
    if (current_state.depth > 2) {
        //double P = 0.25 + 0.5 * reflectance; // Reflection's chance of surviving
        double P = std::clamp(reflectance, 0.1, 0.9); // survival prob for reflection branch
        if (random_double() < P){ // Note: for multi-threading this will have to be replaced with thread_local generator
        //if ((double)rand() / RAND_MAX < P) { // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
            // Reflection works the same way for shells and solids
            double P_reflect = reflectance / P; // Adjust original reflectance based on P
            stack.emplace_back(reflected_ray, next_accumulated_color_reflected * P_reflect, current_state.interior_list, ri_from, current_state.depth + 1, current_state.interior_count);
            return;
        }
        else {
            double P_transmit = transmittance / (1.0 - P); // Adjust original transmittance based on P
            if constexpr (object_type == ObjectType::SHELL) {
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit, current_state.interior_list,
                    ri_to, current_state.depth + 1, current_state.interior_count);
            }
            else if constexpr (object_type == ObjectType::SOLID){
                stack.emplace_back(refracted_ray, next_accumulated_color_refracted * P_transmit, refracted_list,
                    ri_to, current_state.depth + 1, refracted_count);
            }
            return;
        }
    } 
    else {
        // Push both rays
        stack.emplace_back(reflected_ray, next_accumulated_color_reflected * reflectance, current_state.interior_list, ri_from, current_state.depth + 1, current_state.interior_count);

        if constexpr (object_type == ObjectType::SHELL) {
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance, current_state.interior_list,
                ri_to, current_state.depth + 1, current_state.interior_count);
        }
        else if constexpr (object_type == ObjectType::SOLID){
            stack.emplace_back(refracted_ray, next_accumulated_color_refracted * transmittance, refracted_list,
                ri_to, current_state.depth + 1, refracted_count);
        }
        return;
    }
}
*/

#endif // RTMATERIALS_H