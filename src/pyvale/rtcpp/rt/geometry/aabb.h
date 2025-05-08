#ifndef AABB_H
#define AABB_H
// #include "util.h"
#include "ray.h"
#include "vec3.h"
#include <cassert>
#include <cmath>
#include <stdexcept>

// #include <endian.h>


class aabb {
    public:
        aabb() {}
        aabb(const point3& a, const point3& b) { _min = a; _max = b; }
        aabb(const aabb& box0, const aabb& box1) {
            double min_x = std::min(box0.min().x(), box1.min().x());
            double min_y = std::min(box0.min().y(), box1.min().y());
            double min_z = std::min(box0.min().z(), box1.min().z());
            double max_x = std::max(box0.max().x(), box1.max().x());
            double max_y = std::max(box0.max().y(), box1.max().y());
            double max_z = std::max(box0.max().z(), box1.max().z());
            _min = vec3(min_x, min_y, min_z);
            _max = vec3(max_x, max_y, max_z);
        }

        point3 min() const {return _min; }
        point3 max() const {return _max; }

        bool hit(const Ray& r, double dis_min, double dis_max) const {
            for (int a = 0; a < 3; a++) {
                auto t0 = fmin((_min[a] - r.get_origin()[a]) / r.get_direction()[a],
                               (_max[a] - r.get_origin()[a]) / r.get_direction()[a]);
                auto t1 = fmax((_min[a] - r.get_origin()[a]) / r.get_direction()[a],
                               (_max[a] - r.get_origin()[a]) / r.get_direction()[a]);
                dis_min = fmax(t0, dis_min);
                dis_max = fmin(t1, dis_max);
                if (dis_max <= dis_min)
                    return false;
            }
            return true;
        }

        double area() const {
            auto a = _max.x() - _min.x();
            auto b = _max.y() - _min.y();
            auto c = _max.z() - _min.z();
            return 2*(a*b + b*c + c*a);
        }

        int longest_axis() const {
            auto a = _max.x() - _min.x();
            auto b = _max.y() - _min.y();
            auto c = _max.z() - _min.z();
            if (a > b && a > c)
                return 0;
            else if (b > c)
                return 1;
            else
                return 2;
        }

        const std::pair<int, int> axis_interval(int n) const {
            assert((n >= 0) && (n < 3));
            if (n == 0) return {_min.x(), _max.x()};
            if (n == 1) return {_min.y(), _max.y()};
            if (n == 2) return {_min.z(), _max.z()};
            else throw std::invalid_argument("Received argument out of bounds");
        }

        static const aabb empty, universe;

    private:
        point3 _min;
        point3 _max;
};

aabb get_surrounding_box(aabb box0, aabb box1) {
    vec3 small(fmin(box0.min().x(), box1.min().x()),
               fmin(box0.min().y(), box1.min().y()),
               fmin(box0.min().z(), box1.min().z()));

    vec3 big  (fmax(box0.max().x(), box1.max().x()),
               fmax(box0.max().y(), box1.max().y()),
               fmax(box0.max().z(), box1.max().z()));

    return aabb(small,big);
}


aabb operator+(const aabb& bbox, const vec3& offset) {
    return aabb(bbox.min() + offset, bbox.max() + offset);
}

aabb operator+(const vec3& offset, const aabb& bbox) {
    return bbox + offset;
}

const aabb aabb::empty    = aabb(point3(+INFINITY,+INFINITY,+INFINITY), point3(-INFINITY,-INFINITY,-INFINITY));
const aabb aabb::universe = aabb(point3(-INFINITY,-INFINITY,-INFINITY), point3(+INFINITY,+INFINITY,+INFINITY));

#endif
