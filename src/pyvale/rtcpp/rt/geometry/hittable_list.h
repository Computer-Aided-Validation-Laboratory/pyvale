#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
#include "../util.h"

#include "hittable.h"

#include <memory>
#include <vector>


class Hittable_list: public Hittable  {
    public:
        Hittable_list() {}
        Hittable_list(shared_ptr<Hittable> object) { add(object); }

        void clear() { objects.clear(); }
        void add(shared_ptr<Hittable> object) { objects.push_back(object); }

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;
        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const;
        virtual double pdf_value(const point3& o, const vec3& v) const;
        virtual vec3 random(const vec3& o) const;

    public:
        point3 center;
        std::vector<shared_ptr<Hittable>> objects;
};


bool Hittable_list::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    Hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = dis_max;

    for (const auto& object : objects) {
        if (object->hit(r, dis_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.distance;
            hit = temp_rec;
        }
    }

    return hit_anything;
}

bool Hittable_list::bounding_box(double dis_min, double dis_max, aabb& output_box) const {
    if (objects.empty()) return false;

    aabb temp_box;
    bool first_box = true;

    for (const auto& object : objects) {
        if (!object->bounding_box(dis_min, dis_max, temp_box)) return false;
        output_box = first_box ? temp_box : get_surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}


double Hittable_list::pdf_value(const point3& o, const vec3& v) const {
    auto weight = 1.0 / objects.size();
    auto sum = 0.0;

    for (const auto& object : objects)
        sum += weight * object->pdf_value(o, v);

    return sum;
}


vec3 Hittable_list::random(const vec3& o) const {
    auto int_size = static_cast<int>(objects.size());
    return objects[random_int(0, int_size - 1)]->random(o);
}



#endif
