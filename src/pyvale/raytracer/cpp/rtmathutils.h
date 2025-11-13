#pragma once // Header guard instead of ifndef

#include <cmath>
#include <random>
#define _USE_MATH_DEFINES
#include <corecrt_math_defines.h> // Otherwise M_PI doesn't work in VS. cmath alone works in CLion, though
#include "rteigentypes.h"

inline double degreesToRadians(double angleDeg) {
    return angleDeg * M_PI / 180;
}

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}