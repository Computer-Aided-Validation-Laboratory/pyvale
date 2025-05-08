#ifndef BVH_H
#define BVH_H

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

// #include <algorithm>


class BVH_node : public Hittable {
  public:
    BVH_node(Hittable_list list) : BVH_node(list.objects, 0, list.objects.size()) {
            // There's a C++ subtlety here. This constructor (without span indices) creates an
            // implicit copy of the hittable list, which we will modify. The lifetime of the copied
            // list only extends until this constructor exits. That's OK, because we only need to
            // persist the resulting bounding volume hierarchy.
        }

    BVH_node(std::vector<shared_ptr<Hittable>>& objects, size_t start, size_t end) {
        // Build the bounding box of the span of source objects.
        bbox = aabb::empty;
        for (size_t object_index=start; object_index < end; object_index++) {
            aabb other_box;
            objects[object_index]->bounding_box(0,0, other_box);
            bbox = aabb(bbox, other_box);
        }

        int axis = bbox.longest_axis();

        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                      : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            left = objects[start];
            right = objects[start+1];
        } else {
            std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

            auto mid = start + object_span/2;
            left = make_shared<BVH_node>(objects, start, mid);
            right = make_shared<BVH_node>(objects, mid, end);
        }
    }

    bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& rec) const override {
        if (!bbox.hit(r, dis_min, dis_max))
            return false;

        bool hit_left = left->hit(r, dis_min, dis_max, rec);
        bool hit_right = right->hit(r, dis_min, hit_left ? rec.distance : dis_max, rec);

        return hit_left || hit_right;
    }

    // aabb bounding_box() const override { return bbox; }
    bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
    { 
        output_box = bbox;
        return true;
    }

  private:
    shared_ptr<Hittable> left;
    shared_ptr<Hittable> right;
    aabb bbox;

    static bool box_compare(
        const shared_ptr<Hittable> a, const shared_ptr<Hittable> b, int axis_index
    ) {
        aabb a_bb = aabb();
        a->bounding_box(0,0, a_bb);
        aabb b_bb = aabb();
        b->bounding_box(0,0, b_bb);
        auto a_axis_interval = a_bb.axis_interval(axis_index);
        auto b_axis_interval = b_bb.axis_interval(axis_index);
        return std::get<0>(a_axis_interval) < std::get<0>(b_axis_interval);
    }

    static bool box_x_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
        return box_compare(a, b, 2);
    }
};


#endif