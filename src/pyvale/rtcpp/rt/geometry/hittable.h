#ifndef HITTABLE_H
#define HITTABLE_H
#include "../util.h"


class Material;



struct Hit_record {
    point3 position;
    vec3 normal;
    shared_ptr<Material> material_ptr;
    double distance; //last ray collision distance
    bool orientation;
    double u;
    double v;
    
    inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        orientation = dot(r.get_direction(), outward_normal) < 0;
        normal = orientation == UPWARDS ? outward_normal :-outward_normal;
    }
};


class Hittable {
    public:
        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const = 0;
        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const = 0;
        virtual double pdf_value(const vec3& o, const vec3& v) const {
            return 0.0;
        }

        virtual vec3 random(const vec3& o) const {
            return vec3(1, 0, 0);
        }
};









#endif
