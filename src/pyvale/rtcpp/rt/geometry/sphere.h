#ifndef SPHERE_H
#define SPHERE_H
#include "../util.h"

#include "hittable.h"


class Sphere: public Hittable  {
    public:
        Sphere(point3 cen, double r, shared_ptr<Material> m)
            : center(cen), radius(r), material_ptr(m) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;
        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const;
        virtual double pdf_value(const point3& o, const vec3& v) const;
        virtual vec3 random(const point3& o) const;

public:
    point3 center;
    double radius;
    shared_ptr<Material> material_ptr;

};

bool Sphere::bounding_box(double dis_min, double dis_max, aabb& output_box) const {
    output_box = aabb(
        center - vec3(radius, radius, radius),
        center + vec3(radius, radius, radius));
    return true;
}

bool Sphere::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    vec3 oc = r.get_origin() - center;
    auto a = r.get_direction().length_squared();
    auto half_b = dot(oc, r.get_direction());
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = half_b*half_b - a*c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);

        auto temp = (-half_b - root)/a;
        if (temp < dis_max && temp > dis_min) {
            hit.distance = temp;
            hit.position = r.at(hit.distance);
            vec3 outward_normal = (hit.position - center) / radius;
            hit.set_face_normal(r, outward_normal);
            hit.material_ptr = material_ptr;
            return true;
        }

        temp = (-half_b + root) / a;
        if (temp < dis_max && temp > dis_min) {
            hit.distance = temp;
            hit.position = r.at(hit.distance);
            vec3 outward_normal = (hit.position - center) / radius;
            hit.set_face_normal(r, outward_normal);
            hit.material_ptr = material_ptr;
            return true;
        }
    }
    return false;
}

double Sphere::pdf_value(const point3& o, const vec3& scatter_direction) const {
    Hit_record hit;

    double target_distance = (center - o).length();
    double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));

    if (dot(scatter_direction, normalize(center - o)) > cos_theta_max){
        return  1 / ((1 - cos_theta_max) * 2 * pi);
    }
    else {
        return 0;
    }


    
    auto solid_angle = 2 * pi * (1 - cos_theta_max);

    return  1 / solid_angle;
}

vec3 Sphere::random(const point3& o) const {
    

    //double radius2 = 82.5;
    //vec3 center2 = vec3(277, 555, 277);

    //std::cout << radius << " " << radius2 << " \n";
    //std::cout << center2.x() << " " << center2.y() << " " << center2.z() << " "  << " " << "\n";
    //std::cout << center.x() << " " << center.y() << " " << center.z() << " " << " " << "\n";


    double target_distance = (center - o).length();
    double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));

    onb uvw;
    uvw.build_from_w(center - o);
    double phi = random_double() * 2 * pi;
    double r2 = random_double();

    double z = 1. + r2 * (cos_theta_max - 1.);
    double x = cos(phi) * sqrt(1. - z * z);
    double y = sin(phi) * sqrt(1. - z * z);

    return uvw.local(vec3(x, y, z));
}


#endif
