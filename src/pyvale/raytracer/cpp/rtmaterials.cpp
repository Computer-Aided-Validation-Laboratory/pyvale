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

static constexpr double OFFSET_SHADOW = 1e-2;

void ray_diffuse(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){
    // Secondary ray is randomly scattered from the hit point
    // Depends on: Incident ray direction
    // Use non-uniform Lambertian distribution weighed by cos of the angle between the indicent ray and surface normal. Scattering is more likely close to the normal.
    //EiVector3d emitted = intersection_record.emission;
    const EiVector3d p = intersection_record.point_intersection; // Point of intersection
    const double OFFSET = OFFSET_SHADOW * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    //const double OFFSET = std::numeric_limits<double>::epsilon() * 10.0 * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});

    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    EiVector3d next_accumulated_color = current_state.accumulated_color.cwiseProduct(albedo); // Pre-calculate the baseline for the next bounce
    
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    EiVector3d normal_shade = intersection_record.normal_shading; // Shading normal

    // Generate orthonormal basis (Hughes-Moller method)
    //EiVector3d b1 =
    //    ((fabs(normal_shade.x()) > 0.1 ? EiVector3d(0,1,0) : EiVector3d(1,0,0))
    //    .cross(normal_shade)).normalized();
    //EiVector3d b2 = normal_shade.cross(u);

    // Generate orthonormal basis (Duff et al. from Pixar method) (https://jcgt.org/published/0006/01/01/paper-lowres.pdf)
    // Determine the sign of the z-component using std::copysign to avoid branching
    // If n.z >= 0, sign is 1.0; else -1.0.
    double sign = std::copysign(1.0f, normal_shade.z());
    
    // Calculate intermediate values
    double a = -1.0f / (sign + normal_shade.z());
    double b = normal_shade.x() * normal_shade.y() * a;
    
    // Generate the two perpendicular tangent vectors
    EiVector3d b1 = EiVector3d(1.0f + sign * normal_shade.x() * normal_shade.x() * a, sign * b, -sign * normal_shade.x());
    EiVector3d b2 = EiVector3d(b, sign + normal_shade.y() * normal_shade.y() * a, -normal_shade.y());

    // Cosine-weighted hemisphere sampling
    //double r1 = 2 * M_PI * ((double)rand() / RAND_MAX);
    //double r2 = (double)rand() / RAND_MAX;
    double r1 = 2 * M_PI * (random_double());
    double r2 = random_double();
    double r2s = sqrt(r2);

    EiVector3d direction_scatter = (b1 * cos(r1) * r2s + b2 * sin(r1) * r2s + normal_shade * sqrt(1 - r2));
    //direction_scatter.stableNormalize();
    
    /*
    // Catch degenerate scatter direction (NaN prevention)
    if (direction_scatter.squaredNorm() < 1e-8) {
        direction_scatter = normal_shade;
    }
*/
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal

    Ray ray_new;
    ray_new.origin = intersection_record.point_intersection + normal_geo * OFFSET;
    //ray_new.origin = intersection_record.point_intersection + direction_scatter * OFFSET;
    ray_new.direction = direction_scatter;

    stack.push_back({ray_new, next_accumulated_color, current_state.depth + 1});
}

void ray_specular(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){
    // Secondary ray traced in the direction about the normal
    // Depends on: angle between the viewing direction and the surface normal
    //EiVector3d emitted = intersection_record.emission;
    const EiVector3d p = intersection_record.point_intersection; // Point of intersection
    //const double OFFSET = OFFSET_SHADOW * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    const double OFFSET = std::numeric_limits<double>::epsilon() * 10.0 * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    
    intersection_record.normalize_and_flip_normals(current_state.ray);
    intersection_record.align_normals();
    EiVector3d normal_geo = intersection_record.normal_surface; // Geometric normal
    EiVector3d normal_shade = intersection_record.normal_shading; // Shading normal

    EiVector3d next_accumulated_color = current_state.accumulated_color.cwiseProduct(albedo); // Pre-calculate the baseline for the next bounce
    EiVector3d ray_direction = current_state.ray.direction;

    EiVector3d reflected = ray_direction - 2 * ray_direction.dot(normal_shade) * normal_shade;
    //reflected.stableNormalize();
    
    if (reflected.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
        reflected = ray_direction - 2 * ray_direction.dot(normal_geo) * normal_geo;
    }
    Ray ray_new;
    ray_new.origin = intersection_record.point_intersection - normal_geo * OFFSET;
    ray_new.direction = reflected;
    ray_new.t_min = 1e-4 * std::max(1.0, intersection_record.point_intersection.norm());

    stack.push_back({ray_new, next_accumulated_color, current_state.depth + 1});
}

void ray_refractive(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){
    // Secondary ray may reflect or refract
    // Depends on: surface normal, refractive indices, sometimes wavelength
    //EiVector3d emitted = intersection_record.emission;
    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission); // Add emission for the current intersection
    EiVector3d next_accumulated_color = current_state.accumulated_color.cwiseProduct(albedo); // Pre-calculate the baseline for the next bounce
    const EiVector3d p = intersection_record.point_intersection; // Point of intersection
    //const double OFFSET = OFFSET_SHADOW * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
    const double OFFSET = std::numeric_limits<double>::epsilon() * 10.0 * std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())});
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

    //set_face_normal(current_state.ray, normal_shade); // Shade normal won't necessarily point in the same direction as geometric normal, so we can't move it inside the if statement above

    EiVector3d reflected = ray_direction - 2 * ray_direction.dot(normal_shade) * normal_shade;

    //double cos_theta_i = std::max(0.0, -ray_direction.dot(normal_shade)); // Cosine of the incident angle
    //double cos_theta_i = -ray_direction.dot(normal_shade); // Cosine of the incident angle
    double cos_theta_i = std::clamp(-ray_direction.dot(normal_shade), 0.0, 1.0); // To avoid floating point errors
    EiVector3d reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_shade; // Reflection direction
    //reflected_dir.stableNormalize();
    
    if (reflected_dir.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
        reflected_dir = ray_direction + 2.0 * cos_theta_i * normal_geo; 
    }

    double ri_surrounding = 1.0;   // Refractive index of the surrounding medium; for now hardcoded for air
    double ri_material = 1.5;   // Refractive index of the material/volume; for now hardcoded for glass
    double ri_ratio = into ? ri_surrounding / ri_material : ri_material / ri_surrounding; 

    double sin2_theta_t = ri_ratio * ri_ratio * (1.0 - cos_theta_i * cos_theta_i); // Sin^2 of the transmission angle

    /* // tbd which calculation gives better results
    EiVector3d r_out_perp = ri_ratio * (ray_direction + cos_theta_i * normal_shade); // Perpendicular component
    double r_out_perp_length_squared = r_out_perp.squaredNorm();

    // If the squared length is > 1.0, Total Internal Reflection occurs
    if (r_out_perp_length_squared > 1.0) {
        Ray reflected_ray;
        reflected_ray.origin = intersection_record.point_intersection + normal_geo * OFFSET; // Push secondary rays slightly off the surface to remove the shadow acne
        reflected_ray.direction = reflected_dir;
        
        stack.push_back({reflected_ray, next_accumulated_color, current_state.depth + 1});
        return;
    }

    // Calculate the parallel component
    EiVector3d r_out_parallel = -sqrt(fabs(1.0 - r_out_perp_length_squared)) * normal_shade;
    */

    
    // Total internal reflection
    if (sin2_theta_t > 1.0) {
        Ray reflected_ray;
        reflected_ray.origin = intersection_record.point_intersection + normal_geo * OFFSET; // Push secondary rays slightly off the surface to remove the shadow acne
        reflected_ray.direction = reflected_dir;
        reflected_ray.t_min = 1e-4 * std::max(1.0, intersection_record.point_intersection.norm());
        
        stack.push_back({reflected_ray, next_accumulated_color, current_state.depth + 1});
        return;
    }
    
    // Schlick's approximation
    double a = ri_material - ri_surrounding;
    double b = ri_material + ri_surrounding;
    double R0 = (a * a) / (b * b);

    // Use cosine of the medium with the lower index of refraction
    double cos_theta_t = sqrt(1.0 - sin2_theta_t);
    double c = 1 - (into ? cos_theta_i : cos_theta_t);
    //double c = 1 - (into ? -dot_incidence : -dir_transmission.dot(normal_shade));
    double reflectance = R0 + (1 - R0) * (c * c * c * c * c);
    double transmittance = 1 - reflectance;

    // Define new rays
    Ray reflected_ray;
    reflected_ray.origin = intersection_record.point_intersection + normal_geo * OFFSET; // Push back into incident medium (i.e., off the surface)
    reflected_ray.direction = reflected_dir;
    reflected_ray.t_min = 1e-4 * std::max(1.0, intersection_record.point_intersection.norm());

    Ray refracted_ray;
    refracted_ray.origin = intersection_record.point_intersection - normal_geo * OFFSET; // Push forward into new medium (i.e., into the surface)
    refracted_ray.direction = ri_ratio * ray_direction + (ri_ratio * cos_theta_i - cos_theta_t) * normal_shade; // Transmitted/refracted direction
    //refracted_ray.direction = r_out_perp + r_out_parallel;

    //refracted_ray.direction.stableNormalize();
    
    /*
    if (reflected_ray.direction.dot(normal_geo) < 0.0) { // If reflected ray points inside the geometry
        refracted_ray.direction = ri_ratio * ray_direction + (ri_ratio * cos_theta_i - cos_theta_t) * normal_geo;
    }
*/

    // Russian roulette between reflection and refraction
    if (current_state.depth > 2) {
        double P = 0.25 + 0.5 * reflectance; // Reflection's chance of surviving

        if (random_double() < P){ // Note: for multi-threading this will have to be replaced with thread_local generator
        //if ((double)rand() / RAND_MAX < P) { // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
            double P_reflect = reflectance / P; // Adjust original reflectance based on P
            stack.push_back({reflected_ray, next_accumulated_color * P_reflect, current_state.depth + 1});
            return;
        }
        else {
            double P_transmit = transmittance / (1.0 - P); // Adjust original transmittance based on P
            stack.push_back({refracted_ray, next_accumulated_color * P_transmit, current_state.depth + 1});
            return;
        }
    } 
    else {
        // Push both rays
        stack.push_back({reflected_ray, next_accumulated_color * reflectance, current_state.depth + 1});
        stack.push_back({refracted_ray, next_accumulated_color * transmittance, current_state.depth + 1});
        return;
    }
}

// We don't really need most these arguments, but this is to match the function pointer signature to avoid having a switch in the rendering loop
void ray_unlit(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){

    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.face_color);
    return;
}

void ray_undefined(const RayState& current_state,
    HitRecord& intersection_record,
    const EiVector3d& albedo,
    std::vector<RayState>& stack,
    EiVector3d& total_color){
    
    const EiVector3d blue_sky = ray_blue_sky(current_state.ray); // Early termination - no bounces here anyway
    total_color += current_state.accumulated_color.cwiseProduct(blue_sky);
    return;
}
