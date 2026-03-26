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

namespace nb = nanobind;

double speckle(double u, double v) {
    int iu = int(u * 1000);
    int iv = int(v * 1000);
    unsigned int seed = iu * 73856093 ^ iv * 19349663;
    seed = (seed << 13) ^ seed;
    return 0.5 * (1.0 + ((seed * (seed * seed * 15731 + 789221) + 1376312589) & 0x7fffffff) / double(0x7fffffff));
}

EiVector3d return_ray_color(const Ray& ray,
    const TLAS& TLAS) {

    HitRecord intersection_record; // Create HitRecord struct
    // Look for intersection
    IntersectionOutput intersection;
    intersect_TLAS(ray, TLAS, intersection, intersection_record);

    if (intersection_record.t != std::numeric_limits<double>::infinity()) { // Instead of keeping a bool hit_anything, check if t value has changed from the default
        //std::cout << "Coloring..." << std::endl;
        set_face_normal(ray, intersection_record.normal_surface);
        return intersection_record.barycentric_coordinates(0) * intersection_record.face_color + intersection_record.barycentric_coordinates(1) * intersection_record.face_color + intersection_record.barycentric_coordinates(2) * intersection_record.face_color;
    }

    // Blue sky gradient
    double a = 0.5 * (ray.direction(1) + 1.0);
    static EiVector3d white, blue;
    white << 1.0, 1.0, 1.0;
    blue << 0.5, 0.7, 1.0;
    return (1.0 - a) * white + a * blue;
}



// This a new radiance function with lighting
EiVector3d return_ray_color_new(const Ray& ray,
                           const TLAS& TLAS,
                           TextureProjector& texture_projector,
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
    
    double proj_scale = tan(degreesToRadians(texture_projector.proj_fov * 0.5));
    
    // Projector mapping
    EiVector3d xp = rec.point_intersection - texture_projector.proj_pos;
    
    // Build projector basis
    EiVector3d pw = texture_projector.proj_dir;
    EiVector3d pu = pw.cross(EiVector3d(0,1,0)).normalized();
    EiVector3d pv = pu.cross(pw);
    
    // Coordinates in projector space
    double z = xp.dot(pw);
    double u = xp.dot(pu) / (z * proj_scale);
    double v = xp.dot(pv) / (z * proj_scale);
    
    // Normalize to [0,1]
    u = 0.5 * (u + 1.0);
    v = 0.5 * (v + 1.0);
    
    // Sample speckle
    double pattern = 0.0;
    if (z > 0 && u >= 0 && u <= 1 && v >= 0 && v <= 1) {
        // pattern = speckle(u, v);
        pattern = texture_projector.apply_texture(u, v);
    }
    albedo = albedo * pattern;

    switch (rec.material) {

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
            return_ray_color_new(ray_new, TLAS, texture_projector, depth + 1)
        );
    }

    case SPECULAR: {// Specular (mirror)
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(rec.normal_surface) * rec.normal_surface;
        
        Ray ray_new;
        ray_new.origin = rec.point_intersection;
        ray_new.direction = reflected;

        return emitted + albedo.cwiseProduct(
            return_ray_color_new(ray_new, TLAS, texture_projector, depth + 1)
        );
    }

    case REFRACTIVE: {// Refraction (dielectric)
        EiVector3d n = rec.normal_surface;
        EiVector3d nl = (ray.direction.dot(n) < 0) ? n : -n;
    
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(n) * n;
    
        Ray reflRay;
        reflRay.origin = rec.point_intersection;
        reflRay.direction = reflected;
    
        bool into = ray.direction.dot(nl) < 0; // entering or exiting
    
        double nc = 1.0;   // air
        double nt = 1.5;   // glass
        double nnt = into ? nc / nt : nt / nc;
    
        double ddn = ray.direction.dot(nl);
        double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
    
        // Total internal reflection
        if (cos2t < 0) {
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflRay, TLAS, texture_projector, depth + 1)
            );
        }
    
        EiVector3d tdir =
            (ray.direction * nnt -
             n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).normalized();
    
        // Schlick approximation
        double a = nt - nc;
        double b = nt + nc;
        double R0 = (a * a) / (b * b);
    
        double c = 1 - (into ? -ddn : tdir.dot(n));
        double Re = R0 + (1 - R0) * c * c * c * c * c;
        double Tr = 1 - Re;
    
        // Russian roulette between reflection and refraction
        double P = 0.25 + 0.5 * Re;
        double RP = Re / P;
        double TP = Tr / (1 - P);
    
        if (depth > 2) {
            if ((double)rand() / RAND_MAX < P) {
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(reflRay, TLAS, texture_projector, depth + 1) * RP
                );
            } else {
                Ray refrRay;
                refrRay.origin = rec.point_intersection;
                refrRay.direction = tdir;
    
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(refrRay, TLAS, texture_projector, depth + 1) * TP
                );
            }
        } else {
            Ray refrRay;
            refrRay.origin = rec.point_intersection;
            refrRay.direction = tdir;
    
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflRay, TLAS, texture_projector, depth + 1) * Re +
                return_ray_color_new(refrRay, TLAS, texture_projector, depth + 1) * Tr
            );
        }
    }
    }

    return emitted;
}





void render_ppm_image(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const std::filesystem::path output_filepath,
    nb::ndarray<const double, nb::c_contig>& texture_img) {

    
    TextureProjector texture_projector;
    double* texture_img_ptr = const_cast<double*>(texture_img.data());
    int img_h = texture_img.shape(0);
    int img_w = texture_img.shape(1);
    // EiVector3d proj_pos(150, 50, 150);
    EiVector3d proj_pos(-10, 60, 150);
    EiVector3d proj_dir = (EiVector3d(0,0,0) - proj_pos).normalized();
    double proj_fov = 30.0;
    texture_projector.texture_img_ptr = texture_img_ptr;
    texture_projector.proj_pos = proj_pos;
    texture_projector.proj_dir = proj_dir;
    texture_projector.proj_fov = proj_fov;
    texture_projector.img_h = img_h;
    texture_projector.img_w = img_w;



    
    // Get camera parameters from the dict and cast it to Eigen types so it works with existing code; by reference to avoid copying data

    std::vector<uint8_t> buffer;
    buffer.reserve(image_width * image_height * 12); // Preallocate memory for the image buffer (conservatively)

    for (int j = 0; j < image_height; j++) {
        //std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush << std::endl;
        for (int i = 0; i < image_width; i++) {


            EiVector3d pixel_color = EiVector3d::Zero();
            EiVector3d pixel_color_new = EiVector3d::Zero();


            for (int k = 0; k < number_of_samples; k++) {
                double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                EiVector3d pixel_sample = pixel_00_center +
                    (i + offset[0]) * matrix_pixel_spacing.row(0) +
                    (j + offset[1]) * matrix_pixel_spacing.row(1);
                EiVector3d ray_direction = pixel_sample - camera_center;
                Ray current_ray{ camera_center, ray_direction.normalized() };
                
                
                // pixel_color_new += return_ray_color_new(current_ray, TLAS);
                // pixel_color += return_ray_color(current_ray, TLAS);
                pixel_color += return_ray_color_new(current_ray, TLAS, texture_projector);
            }
            
            // Convert to gray
            double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
            int gray_byte = int(gray / number_of_samples * 255.99);
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));
            buffer.push_back(static_cast<uint8_t>(gray_byte));

            
            // Keep  the colours
            // double r = pixel_color[0] / number_of_samples;
            // double g = pixel_color[1] / number_of_samples;
            // double b = pixel_color[2] / number_of_samples;
            
            // int r_byte = int(255.99 * r);
            // int g_byte = int(255.99 * g);
            // int b_byte = int(255.99 * b);
            
            // buffer.push_back(static_cast<uint8_t>(r_byte));
            // buffer.push_back(static_cast<uint8_t>(g_byte));
            // buffer.push_back(static_cast<uint8_t>(b_byte));


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