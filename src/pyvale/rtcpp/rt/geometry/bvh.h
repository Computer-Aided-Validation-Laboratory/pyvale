#ifndef BVH_H
#define BVH_H
#include "../util.h"
#include "hittable.h"

#include <algorithm>

//Splitting BVH Volumes:
// brief algorithm description:

// 1. create bvh_node(Hittable_list)
// 2. sort the Hittable_list comparing their surrounding boxes along a random axis 
// 3. put half in each subtree:  
//      left = bvh_node(subtree_left); 
//      right = bvh_node(subtree_right);
// 4. when subtree.size() <= 2 recursion end

class BVH_node : public Hittable  {
    public:
        BVH_node();

        BVH_node(Hittable_list& list, double dis_min, double dis_max)
            : BVH_node(list.objects, 0, list.objects.size(), dis_min, dis_max)
        {}

        // Function Overloading

        BVH_node(
            std::vector<shared_ptr<Hittable>>& objects,
            size_t start, size_t end, double dis_min, double dis_max);

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;
        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const;

    public:
        shared_ptr<Hittable> left;
        shared_ptr<Hittable> right;
        aabb Box;
};

// check if box_a is smaller than box_b in specified axis
inline bool box_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b, int axis) {
    aabb box_a;
    aabb box_b;

    if (!a->bounding_box(0,0, box_a) || !b->bounding_box(0,0, box_b))
        std::cerr << "No bounding box in BVH_node constructor.\n";

    return box_a.min().e[axis] < box_b.min().e[axis];
}


bool box_x_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
    return box_compare(a, b, 0);
}

bool box_y_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
    return box_compare(a, b, 1);
}

bool box_z_compare (const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
    return box_compare(a, b, 2);
}


BVH_node::BVH_node(
    std::vector<shared_ptr<Hittable>>& objects,
    size_t start, size_t end, double dis_min, double dis_max
) {
    // randomly choose an axis
    // 0 = we go for x axis,  1 = we go for y axis,   2 we go for z axis

    int axis = random_int(0,2);
    auto comparator = (axis == 0) ? box_x_compare
                    : (axis == 1) ? box_y_compare
                                  : box_z_compare;
    // number of objects
    size_t object_span = end - start;

    if (object_span == 1) {
        left = right = objects[start];        
        //end recursion
    } else if (object_span == 2) {
        if (comparator(objects[start], objects[start+1])) {
            left = objects[start];
            right = objects[start+1];
            //end recursion
        } else {
            left = objects[start+1];
            right = objects[start];
            //end recursion
        }
    } else {
        // sort objects comparing their surrounding boxes along choosen axis

        std::sort(objects.begin() + start, objects.begin() + end, comparator);
        
        // divide the sorted objects 
        auto mid = start + object_span/2;

        //recursively create bvh_nodes
        left = make_shared<BVH_node>(objects, start, mid, dis_min, dis_max);
        right = make_shared<BVH_node>(objects, mid, end, dis_min, dis_max);
    }

    aabb box_left, box_right;

    if (  !left->bounding_box (dis_min, dis_max, box_left)
       || !right->bounding_box(dis_min, dis_max, box_right)
    )
        std::cerr << "No bounding box in BVH_node constructor.\n";

    //get_surrounding_box defined in aabb.h
    Box = get_surrounding_box(box_left, box_right);
}


bool BVH_node::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    if (!Box.hit(r, dis_min, dis_max))
        return false;

    bool hit_left = left->hit(r, dis_min, dis_max, hit);
    bool hit_right = right->hit(r, dis_min, hit_left ? hit.distance : dis_max, hit);

    return hit_left || hit_right;
}


bool BVH_node::bounding_box(double dis_min, double dis_max, aabb& output_box) const {
    output_box = Box;
    return true;
}


#endif
