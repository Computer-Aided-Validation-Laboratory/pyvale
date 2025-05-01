#ifndef CAMERA_H
#define CAMERA_H
#include "geometry/hittable.h"
#include "util.h"
#include "pdf.h"
#include "vec3.h"
#include "color.h"
#include "materials/material.h"


class Camera {
    public:
        color background = color(0.7, 0.7, 0.7);
        int width_pixels = 200;

        Camera() : Camera(point3(0,0,-1), point3(0,0,0), 40, 1, 0, 10, vec3(0, 1, 0)) {}

        Camera(
            point3 lookfrom,
            point3 lookat,
            double field_of_view, // vertical field-of-view in degrees
            double aspect_ratio,
            double aperture,
            double focal_length,
            vec3 vup,
            double t0 = 0,
            double t1 = 0
        ) {
            auto theta = degrees_to_radians(field_of_view);
            auto h = tan(theta/2);
            auto screen_height = 2.0 * h;
            auto screen_width = aspect_ratio * screen_height;

            CameraFwd = normalize(lookat - lookfrom);
            CameraRight = normalize(cross(CameraFwd, vup));
            CameraUp = cross(CameraRight, CameraFwd);

            origin = lookfrom;
            horizontal = focal_length * screen_width * CameraRight;
            vertical = focal_length * screen_height * CameraUp;
            focalplane_lower_left_corner = origin - horizontal/2 - vertical/2 + focal_length*CameraFwd;

            lens_radius = aperture / 2;
            time0 = t0;
            time1 = t1;
            // fov = field_of_view;
            ar = aspect_ratio;
        }

        Ray get_ray(double u, double v, int max_depth) const {
            vec3 rd = lens_radius * random_in_unit_disk(); 
            vec3 offset = CameraRight * rd.x() + CameraUp * rd.y();  //lens depth of field effect
            return Ray(
                origin + offset,
                normalize(focalplane_lower_left_corner + u * horizontal + v * vertical - origin - offset), max_depth,
                random_double(time0, time1)
            );
        }

        void render(const Hittable& world, const Hittable& lights) const {
            auto image_width = width_pixels;
            auto image_height = int(width_pixels * ar);
            // auto theta = degrees_to_radians(fov);
            // auto h = tan(theta/2);
            // auto screen_height = 2.0 * h;
            // auto screen_width = aspect_ratio * screen_height;
            double pixel_samples_scale = double(1.0 / samples_per_pixel);

            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            for (int j = 0; j < image_height; j++) {
                std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
                for (int i = 0; i < image_width; i++) {
                    color pixel_color(0,0,0);
                    for (int sample = 0; sample < samples_per_pixel; sample++) {
                        Ray r = get_ray(i, j, max_depth);
                        pixel_color += ray_color(r, max_depth, world, lights);
                    }
                    write_color(std::cout, pixel_samples_scale * pixel_color);
                }
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

        int samples_per_pixel = 20;
        int max_depth = 5;


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
