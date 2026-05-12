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
#include "rtmaterials.h"

// New radiance function with lighting but iterative and refactored 
EiVector3d return_ray_color_stack(const Ray& primary_ray, const TLAS& TLAS){

    static constexpr int MAX_DEPTH = 60; // Max depth for the secondary rays
    EiVector3d total_color = EiVector3d::Zero();
    std::vector<RayState> stack;
    stack.reserve(MAX_DEPTH);
    stack.push_back({primary_ray, EiVector3d(1.0, 1.0, 1.0), 0});
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

        if (intersection_record.t == std::numeric_limits<double>::infinity()) {
            const EiVector3d blue_sky = ray_blue_sky(current_ray); // Early termination - no bounces here anyway
            total_color += current_state.accumulated_color.cwiseProduct(blue_sky);
            continue;
        }

        // Set and normalize surface and shading normals - had to move this inside the specific material functions as flipping here broke the logic for refractive materials
        //set_face_normal(current_ray, intersection_record.normal_surface);
        //set_face_normal(current_ray, intersection_record.normal_shading);

        EiVector3d emitted = intersection_record.emission;
        EiVector3d albedo = intersection_record.face_color;

        /*
        // Explicit depth limit with ambient fallback
        if (current_state.depth >= MAX_DEPTH) {
        //    // Add a fallback ambient color to compensate for truncated energy 
            // Avoids the "plain black shadows" caused by zero light return
            EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2; 
            total_color += current_state.accumulated_color.cwiseProduct(emitted + ambient_fallback);
            continue; 
        }*/
        
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

void render_ppm_image(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_filepath) {
    // Get camera parameters from the dict and cast it to Eigen types so it works with existing code; by reference to avoid copying data

    std::vector<uint8_t> buffer;
    buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)

    for (int j = 0; j < image_height; j++) {
        //std::cerr << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush << std::endl;
        for (int i = 0; i < image_width; i++) {
            EiVector3d pixel_color = EiVector3d::Zero();
            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * matrix_pixel_spacing.row(0) +
                    (j + offset[1]) * matrix_pixel_spacing.row(1);
                std::array<double, 2> defocus_disc_offset = point_in_unit_disk();
                EiVector3d defocus_disc_sample = defocus_disc_offset[0] * matrix_defocus_disc.row(0) + defocus_disc_offset[1] * matrix_defocus_disc.row(1);
                EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
                EiVector3d ray_direction = pixel_sample - ray_origin; // ray direction in thin lens approx
                //EiVector3d ray_origin = camera_center; // ray origin in pinhole camera mode
                //EiVector3d ray_direction = pixel_sample - camera_center; // ray direction in pinhole camera mode;
                //Ray current_ray{ ray_origin, ray_direction.normalized() };
                Ray current_ray{ ray_origin, ray_direction};
                //pixel_color += return_ray_color(current_ray, TLAS);
                //pixel_color += return_ray_color_new(current_ray, TLAS);
                pixel_color += return_ray_color_stack(current_ray, TLAS);
            }
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray / number_of_samples * 255.99);
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
        }
    }

    std::ofstream image_file;

    image_file.open(output_filepath);
    if (!image_file.is_open()) {
        std::cerr << "Failed to open the output file.\n";
        return;
    }

    image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
    image_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());

    image_file.close();
    std::cout << "\r Done. \n";
}