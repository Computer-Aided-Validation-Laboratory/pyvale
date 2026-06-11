// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTMATHUTILS_H
#define RTMATHUTILS_H

// STD header files
#include <cmath>
#include <random>
#define _USE_MATH_DEFINES

// raytracer header files
#include "rteigentypes.h"

/**
 * @brief Converts angle value from degrees to radians.
 * 
 * @param[in] angle_deg (double) Value in degrees to convert.
 * @return (double) Passed value converted to radians.
 */
inline double degreesToRadians(const double angle_deg) {
    return angle_deg * M_PI / 180;
}

/**
 * @brief Generates a random double in the range [0, 1).
 * 
 * Thread-safe implementation using thread-local storage for both the
 * random number generator and distribution. By default, the generator
 * is not explicitly seeded, resulting in implementation-defined behavior.
 * 
 * @return (double) Random value uniformly distributed in [0, 1).
 */
inline double random_double() {
    thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    thread_local std::mt19937 generator; // No seed
    //static std::mt19937 generator(123456u); // With seed to get deterministic results for regression tests. Guaranteed to get the same sequence for the same seed via the C++ standard
    //thread_local std::mt19937 generator(std::random_device{}()); // If we want to ensure each thread gets unique sequence
    return distribution(generator);
}

/**
 * @brief Generates a random double in the range [-1, 1).
 * 
 * Thread-safe implementation using thread-local storage for both the
 * random number generator and distribution. Primarily intended for
 * sampling coordinates within a square domain.
 * 
 * @return (double) Random value uniformly distributed in [-1, 1).
 */
static inline double random_double_disk() {
    thread_local std::uniform_real_distribution<double> distribution2(-1.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution2(generator);
}

/**
 * @brief Generates a random 2D point inside the unit disk.
 * 
 * Uses rejection sampling by generating points in the square [-1, 1] × [-1, 1]
 * and accepting only those that lie within the unit circle. This produces a
 * uniform distribution over the disk, used for the camera defocus disc.
 * 
 * @return (std::array<double,2>) A 2D point (x,y) such that x^2 + y^2 < 1.
 */
static inline std::array<double,2> point_in_unit_disk(){
    while (true){
        std::array<double,2> offset = { random_double_disk(), random_double_disk()};
        if (offset[0] * offset[0] + offset[1] * offset[1] < 1.0){
            return offset;
        }
    }
}

#endif // RTMATHUTILS_H