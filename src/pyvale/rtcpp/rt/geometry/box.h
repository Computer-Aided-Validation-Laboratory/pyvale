#ifndef BOX_H
#define BOX_H
#include "../util.h"

#include "plane.h"
#include "hittable_list.h"


class Box: public Hittable  {
    public:
        Box() {}
        Box(const point3& p0, const point3& p1, shared_ptr<Material> ptr);

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            output_box = aabb(box_min, box_max);
            return true;
        }
        virtual double pdf_value(const point3& o, const vec3& v) const {
            Hit_record hit;
            if (!this->hit(Ray(o, v, 1, 0.), 0.0001, infinity, hit))
                return 0;
            double x0 = box_min.x();
            double x1 = box_max.x();
            double z0 = box_min.z();
            double z1 = box_max.z();

            double area = (x1 - x0) * (z1 - z0);
            double distance_squared = hit.distance * hit.distance;
            double cosine = fabs(dot(v, hit.normal));

            return clamp(distance_squared / (cosine * area) , 0.01, 100. );
        }

        virtual vec3 random(const point3& origin) const {

            double x0 = box_min.x();
            double x1 = box_max.x();
            double z0 = box_min.z();
            double z1 = box_max.z();
            double k = (box_min.y() + box_max.y()) / 2;

            auto random_point = point3(random_double(x0, x1), k, random_double(z0, z1));
            return normalize(random_point - origin);
        }
    public:
        point3 box_min;
        point3 box_max;
        Hittable_list sides;
};


Box::Box(const point3& p0, const point3& p1, shared_ptr<Material> ptr) {
    box_min = p0;
    box_max = p1;

    sides.add(make_shared<Plane_xy>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.add(make_shared<Plane_xy>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.add(make_shared<Plane_xz>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.add(make_shared<Plane_xz>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.add(make_shared<Plane_yz>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.add(make_shared<Plane_yz>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool Box::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    return sides.hit(r, dis_min, dis_max, hit);
}


#endif
