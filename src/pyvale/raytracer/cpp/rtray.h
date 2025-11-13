#pragma once

#include <limits>
#include "rteigentypes.h"

struct Ray {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    EiVector3d origin;
    EiVector3d direction;
    double t_min;
    double t_max{ std::numeric_limits<double>::infinity() };
};

inline EiVector3d ray_at_t(const double t, const Ray& ray) {
    return ray.origin + t * ray.direction;
};
// return direction.normalized(); // for normalizing ray direction; can keep it inline, just have it here so I don't forget it's an option