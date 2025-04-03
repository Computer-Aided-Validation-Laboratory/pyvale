#ifndef PLANE_H
#define PLANE_H
#include "../util.h"

#include "hittable.h"


class Plane_xy: public Hittable {
    public:

        Plane_xy(
            double _x0, double _x1, double _y0, double _y1, double _z, shared_ptr<Material> mat
        ) : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_z), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
            return true;
        }

    public:
        shared_ptr<Material> mp;
        double x0, x1, y0, y1, k;
};

class Plane_xz: public Hittable {
    public:

        Plane_xz(
            double _x0, double _x1, double _z0, double _z1, double _y, shared_ptr<Material> mat
        ) : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_y), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = aabb(point3(x0,k-0.0001,z0), point3(x1, k+0.0001, z1));
            return true;
        }

        virtual double pdf_value(const point3& o, const vec3& v) const {
            Hit_record hit;
            if (!this->hit(Ray(o, v, 1, 0.), 0.0001, infinity, hit))
                return 0;

            double area = (x1 - x0) * (z1 - z0);
            double distance_squared = hit.distance * hit.distance;
            double cosine = (dot(-v, hit.normal));
            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);


            return clamp(distance_squared / (cosine * area), 0.01, 10000);
        }

        virtual vec3 random(const point3& origin) const {

            auto random_point = point3(random_double(x0, x1), k, random_double(z0, z1));
            return normalize(random_point - origin);
        }

        /*virtual double pdf_value(const point3& o, const vec3& scatter_direction) const {
            Hit_record hit;

            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);
            double radius = vec3((x0 - x1) / 2, 0, (z0 - z1) / 2).length();
            double target_distance = (center - o).length();
            double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));

            if (dot(scatter_direction, normalize(center - o)) > cos_theta_max) {
                return  1 / ((1 - cos_theta_max) * 2 * pi);
            }
            else {
                return 0;
            }



            auto solid_angle = 2 * pi * (1 - cos_theta_max);

            return  1 / solid_angle;
        }

        virtual vec3 random(const point3& o) const {
            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);
            double radius = vec3((x0 - x1) / 2, 0, (z0 - z1) / 2).length();
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


        }*/
    public:
        shared_ptr<Material> mp;
        double x0, x1, z0, z1, k;
};

class Plane_yz: public Hittable {
    public:

        Plane_yz(
            double _y0, double _y1, double _z0, double _z1, double _x, shared_ptr<Material> mat
        ) : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_x), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = aabb(point3(k-0.0001, y0, z0), point3(k+0.0001, y1, z1));
            return true;
        }

    public:
        shared_ptr<Material> mp;
        double y0, y1, z0, z1, k;
};

bool Plane_xy::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().z()) / r.get_direction().z();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto x = r.get_origin().x() + dis*r.get_direction().x();
    auto y = r.get_origin().y() + dis*r.get_direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;

    hit.u = (x-x0)/(x1-x0);
    hit.v = (y-y0)/(y1-y0);
    hit.distance = dis;
    auto outward_normal = vec3(0, 0, 1);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}

bool Plane_xz::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().y()) / r.get_direction().y();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto x = r.get_origin().x() + dis*r.get_direction().x();
    auto z = r.get_origin().z() + dis*r.get_direction().z();
    if (x < x0 - 0. || x > x1 + 0. || z < z0 - 0. || z > z1 + 0.)

        return false;

    hit.u = (x-x0)/(x1-x0);
    hit.v = (z-z0)/(z1-z0);
    hit.distance = dis;
    auto outward_normal = vec3(0, 1, 0);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}

bool Plane_yz::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().x()) / r.get_direction().x();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto y = r.get_origin().y() + dis*r.get_direction().y();
    auto z = r.get_origin().z() + dis*r.get_direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;

    hit.u = (y-y0)/(y1-y0);
    hit.v = (z-z0)/(z1-z0);
    hit.distance = dis;
    auto outward_normal = vec3(1, 0, 0);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}








#endif
