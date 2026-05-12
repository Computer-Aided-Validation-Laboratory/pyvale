// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

// STD header files
#include <limits>

// ray tracer header files
#include "rteigentypes.h"

struct Ray {
    //EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // Required for structures using Eigen members
    EiVector3d origin;
    EiVector3d direction;
    double t_min {1e-8};
    double t_max{ std::numeric_limits<double>::infinity() };
};

inline EiVector3d ray_at_t(const double t, const Ray& ray) {
    return ray.origin + t * ray.direction;
};