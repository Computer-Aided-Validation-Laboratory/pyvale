// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once // Header guard instead of ifndef

// STD header files
#include <cmath>
#include <random>
#define _USE_MATH_DEFINES

// raytracer header files
#include "rteigentypes.h"

inline double degreesToRadians(double angleDeg) {
    return angleDeg * M_PI / 180;
}

inline double clip(double number, double lower_boundary, double upper_boundary){
    return std::max(lower_boundary, std::min(number, upper_boundary));
}

static std::uniform_real_distribution<double> distribution(0.0, 1.0);
static std::mt19937 generator; // No seed
//static std::mt19937 generator(123456u); // With seed to get deterministic results for regression tests. Guaranteed to get the same sequence for the same seed via the C++ standard
inline double random_double() {
    return distribution(generator);
}