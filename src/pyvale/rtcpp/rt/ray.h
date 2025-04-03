#ifndef RAY_H
#define RAY_H
#include "vec3.h"


class Ray
{
    public:
        Ray() {};
        Ray(const point3& origin, const vec3& direction,int depth, double time)
            : origin(origin), direction(direction), depth(depth), tm(time)
        {}

        point3 get_origin() const  { return origin; }
        vec3 get_direction() const { return direction; }
        double get_time() const    { return tm; }
        int get_depth() const    { return depth; }

        point3 at(double distance) const {
            return origin + distance*direction;
        }

    public:
        point3 origin;
        vec3 direction;
        double tm;
        int depth;
};

#endif
