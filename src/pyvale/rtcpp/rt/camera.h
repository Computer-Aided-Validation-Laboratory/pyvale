#ifndef CAMERA_H
#define CAMERA_H
#include "geometry/hittable.h"
#include "util.h"
#include "pdf.h"
#include "vec3.h"
#include "colour.h"
#include "materials/material.h"
#include <ostream>
#include <omp.h>


class Camera {
    public:
        color background = color(0.7, 0.7, 0.7);
        int image_width = 200;
        int samples_per_pixel = 10;
        int max_depth = 5;

        double vfov = 90;
        point3 lookfrom = point3(0,0,0);   // Point camera is looking from
        point3 lookat   = point3(0,0,-1);  // Point camera is looking at
        vec3   vup      = vec3(0,1,0);     // Camera-relative "up" direction

        double defocus_angle = 0;  // Variation angle of rays through each pixel
        double focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus

        Camera() : Camera(point3(0,0,-1), point3(0,0,0), 40, 1, 0, 10, vec3(0, 1, 0), 500) {}

        Camera(
            point3 lookfrom,
            point3 lookat,
            double field_of_view, // vertical field-of-view in degrees
            double aspect_ratio,
            double aperture,
            double focal_length,
            vec3 vup,
            int image_width,
            double t0 = 0,
            double t1 = 0
        ) : lookfrom(lookfrom),
        lookat(lookat),
        vup(vup),
        image_width(image_width)
        {
            // auto theta = degrees_to_radians(field_of_view);
            // auto h = tan(theta/2);
            // auto screen_height = 2.0 * h;
            // auto screen_width = aspect_ratio * screen_height;

            // CameraFwd = normalize(lookat - lookfrom);
            // CameraRight = normalize(cross(CameraFwd, vup));
            // CameraUp = cross(CameraRight, CameraFwd);

            // origin = lookfrom;
            // horizontal = focal_length * screen_width * CameraRight;
            // vertical = focal_length * screen_height * CameraUp;
            // focalplane_lower_left_corner = origin - horizontal/2 - vertical/2 + focal_length*CameraFwd;

            // lens_radius = aperture / 2;
            // time0 = t0;
            // time1 = t1;
            // // fov = field_of_view;
            // ar = aspect_ratio;

            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;
    
            sqrt_spp = int(std::sqrt(samples_per_pixel));
            pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
            recip_sqrt_spp = 1.0 / sqrt_spp;
    
            center = lookfrom;
    
            // Determine viewport dimensions.
            auto theta = degrees_to_radians(vfov);
            auto h = std::tan(theta/2);
            auto viewport_height = 2 * h * focus_dist;
            auto viewport_width = viewport_height * (double(image_width)/image_height);
    
            // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
            w = normalize(lookfrom - lookat);
            u = normalize(cross(vup, w));
            v = cross(w, u);
    
            // Calculate the vectors across the horizontal and down the vertical viewport edges.
            vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
            vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge
    
            // Calculate the horizontal and vertical delta vectors from pixel to pixel.
            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;
    
            // Calculate the location of the upper left pixel.
            auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    
            // Calculate the camera defocus disk basis vectors.
            auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

        // Ray get_ray(double u, double v, int max_depth) const {
        //     vec3 rd = lens_radius * random_in_unit_disk(); 
        //     vec3 offset = CameraRight * rd.x() + CameraUp * rd.y();  //lens depth of field effect
        //     return Ray(
        //         origin + offset,
        //         normalize(focalplane_lower_left_corner + u * horizontal + v * vertical - origin - offset), max_depth,
        //         random_double(time0, time1)
        //     );
        // }

        // void render(const Hittable& world, const Hittable& lights, std::ostream& stream = std::cout) const {
        //     auto image_width = width_pixels;
        //     auto image_height = int(width_pixels * ar);
        //     // auto theta = degrees_to_radians(fov);
        //     // auto h = tan(theta/2);
        //     // auto screen_height = 2.0 * h;
        //     // auto screen_width = aspect_ratio * screen_height;
        //     double pixel_samples_scale = double(1.0 / samples_per_pixel);

        //     stream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        //     for (int j = 0; j < image_height; j++) {
        //         std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        //         for (int i = 0; i < image_width; i++) {
        //             color pixel_color(0,0,0);
        //             for (int sample = 0; sample < samples_per_pixel; sample++) {
        //                 Ray r = get_ray(i, j, max_depth);
        //                 pixel_color += ray_color(r, max_depth, world, lights);
        //             }
        //             write_color(stream, pixel_samples_scale * pixel_color);
        //         }
        //     }
    
        //     std::clog << "\rDone.                 \n";     
        // }
        void render(const Hittable& world, const Hittable& lights, std::ostream& stream = std::cout) {
            // initialize();
    
            stream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            std::vector<std::string> pixel_buffer(image_height * image_width);
    
            #pragma omp parallel for schedule(dynamic)
            for (int j = 0; j < image_height; j++) {
                #ifndef _OPENMP
                    // log progress if serial
                    std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
                #endif

                for (int i = 0; i < image_width; i++) {
                    color pixel_color(0,0,0);
                    for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                        for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                            Ray r = get_ray(i, j, s_i, s_j);
                            pixel_color += ray_color(r, max_depth, world, lights);
                        }
                    }

                    std::ostringstream ss;
                    write_color(ss, pixel_samples_scale * pixel_color);
                    pixel_buffer[j * image_width + i] = ss.str();
                }
            }
            
            // Shove all the buffer into the steam after being computed
            for (const auto& line : pixel_buffer) {
                stream << line;
            }

            std::clog << "\rDone.                 \n";
        }

    private:
        point3 origin;
        point3 focalplane_lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 CameraRight, CameraUp, CameraFwd;
        double lens_radius;
        double time0, time1;  // shutter open/close times
        // double fov;
        double ar;

        // int samples_per_pixel = 20;
        // int max_depth = 5;
        int    image_height;         // Rendered image height
        double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
        int    sqrt_spp;             // Square root of number of samples per pixel
        double recip_sqrt_spp;       // 1 / sqrt_spp
        point3 center;               // Camera center
        point3 pixel00_loc;          // Location of pixel 0, 0
        vec3   pixel_delta_u;        // Offset to pixel to the right
        vec3   pixel_delta_v;        // Offset to pixel below
        vec3   u, v, w;              // Camera frame basis vectors
        vec3   defocus_disk_u;       // Defocus disk horizontal radius
        vec3   defocus_disk_v;       // Defocus disk vertical radius


        Ray get_ray(int i, int j, int s_i, int s_j) const {
            // Construct a camera ray originating from the defocus disk and directed at a randomly
            // sampled point around the pixel location i, j for stratified sample square s_i, s_j.
    
            auto offset = sample_square_stratified(s_i, s_j);
            auto pixel_sample = pixel00_loc
                              + ((i + offset.x()) * pixel_delta_u)
                              + ((j + offset.y()) * pixel_delta_v);
    
            auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
            auto ray_direction = pixel_sample - ray_origin;
            auto ray_time = random_double();
    
            return Ray(ray_origin, ray_direction, max_depth, ray_time);
        }
    
        vec3 sample_square_stratified(int s_i, int s_j) const {
            // Returns the vector to a random point in the square sub-pixel specified by grid
            // indices s_i and s_j, for an idealized unit square pixel [-.5,-.5] to [+.5,+.5].
    
            auto px = ((s_i + random_double()) * recip_sqrt_spp) - 0.5;
            auto py = ((s_j + random_double()) * recip_sqrt_spp) - 0.5;
    
            return vec3(px, py, 0);
        }
    
        vec3 sample_square() const {
            // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
            return vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }
    
        vec3 sample_disk(double radius) const {
            // Returns a random point in the unit (radius 0.5) disk centered at the origin.
            return radius * random_in_unit_disk();
        }
    
        point3 defocus_disk_sample() const {
            // Returns a random point in the camera defocus disk.
            auto p = random_in_unit_disk();
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
        }

        color ray_color(const Ray& r, int depth, const Hittable& world, const Hittable& lights) const {
            // If we've exceeded the ray bounce limit, no more light is gathered.
            if (depth <= 0)
                return color(0,0,0);
    
            Hit_record rec;
    
            // If the ray hits nothing, return the background color.
            if (!world.hit(r, 0.001, infinity, rec))
                return background;
    
            scatter_record srec;

            color color_from_emission = rec.material_ptr->emitted(r, rec, rec.u, rec.v, rec.position);

            if (!rec.material_ptr->scatter(r, rec, srec))
                return color_from_emission;
            
            // srec.skip_pdf = true;

            if (srec.skip_pdf) {
                return srec.attenuation * ray_color(srec.skip_pdf_ray, depth-1, world, lights);
            }

            auto light_ptr = std::make_shared<Hittable_PDF>(lights, rec.position);
            Mixture_PDF p(light_ptr, srec.pdf_ptr);

            Ray scattered = Ray(rec.position, p.generate(), depth, r.get_time());
            auto pdf_value = p.value(scattered.get_direction());

            double scattering_pdf = rec.material_ptr->scattering_pdf(r, rec, scattered);

            color sample_color = ray_color(scattered, depth-1, world, lights);
            color color_from_scatter =
                (srec.attenuation * scattering_pdf * sample_color) / pdf_value;

            return color_from_emission + color_from_scatter;        

        }
};

#endif
