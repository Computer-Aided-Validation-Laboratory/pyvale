#ifndef CAMERA_H
#define CAMERA_H
#include "util.h"


class Camera {
    public:
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

    private:
        point3 origin;
        point3 focalplane_lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 CameraRight, CameraUp, CameraFwd;
        double lens_radius;
        double time0, time1;  // shutter open/close times
};

#endif
