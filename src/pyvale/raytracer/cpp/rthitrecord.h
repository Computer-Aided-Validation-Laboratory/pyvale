#pragma once

#include <limits>
#include "rteigentypes.h"
#include "rtray.h"

struct HitRecord {
    // Hit record, which is called every time we test ray for an intersection. Ultimately stores the values of the closest hits
    double t{ std::numeric_limits<double>::infinity() };
    EiVector3d point_intersection{ EiVector3d::Zero() };
    EiVector3d normal_surface{ EiVector3d::Zero() };
    EiVector3d barycentric_coordinates{ EiVector3d::Zero() };
    EiVector3d face_color{ EiVector3d::Zero() };
};

inline void set_face_normal(const Ray& ray, EiVector3d& normal_surface) {
    // Normalises the surface normal at the intersection point and determines which way the ray hits the object. Flips the normal if it hits the back face
    normal_surface = normal_surface.normalized();
    if (ray.direction.dot(normal_surface) > 0.0) {
        normal_surface = -normal_surface; // Flip normal if it hits the back face
    }
}