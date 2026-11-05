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
#include <array>
#define _USE_MATH_DEFINES

// raytracer header files
#include "rteigentypes.h"
#include "rtsobolsampler.h" // [SOBOL] Quati-Monte Carlo sampler

/**
 * @brief Converts angle value from degrees to radians.
 * 
 * @param[in] angle_deg (double) Value in degrees to convert.
 * @return (double) Passed value converted to radians.
 */
inline double degreesToRadians(const double angle_deg) {
    return angle_deg * M_PI / 180;
}

// ================================================================================
// Monte Carlo sampling functions
// ================================================================================

/**
 * @brief [MT19937 - LEGACY] Generates a random double in the range [0, 1).
 * 
 * Thread-safe implementation using thread-local storage for both the
 * random number generator and distribution. By default, the generator
 * is not explicitly seeded, resulting in implementation-defined behavior.
 * 
 * This uses Mersenne-Twister => Monte-Carlo algorithm
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
 * @brief [MT19937 - LEGACY] Generates a random double in the range [-1, 1).
 * 
 * Thread-safe implementation using thread-local storage for both the
 * random number generator and distribution. Primarily intended for
 * sampling coordinates within a square domain.
 * 
 * This uses Mersenne-Twister => Monte-Carlo algorithm
 * 
 * @return (double) Random value uniformly distributed in [-1, 1).
 */
static inline double random_double_disk() {
    thread_local std::uniform_real_distribution<double> distribution2(-1.0, 1.0);
    thread_local std::mt19937 generator;
    return distribution2(generator);
}

/**
 * @brief [MT19937 - LEGACY] Generates a random 2D point inside the unit disk.
 * 
 * Uses rejection sampling by generating points in the square [-1, 1] × [-1, 1]
 * and accepting only those that lie within the unit circle. This produces a
 * uniform distribution over the disk, used for the camera defocus disc.
 * 
 * This uses Mersenne-Twister => Monte-Carlo algorithm
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

// ================================================================================
// Quasi-Monte Carlo functions
// ================================================================================\

/**
 * @brief [SOBOL] Maps [−1,1]^2 bijectively onto the unit disk with no rejections and no bias
 * 
 * Also known as Shirley's concentric disk map.
 * This is used to replace point_in_unit_disk() in thin lens camera model
 * when sampling with Sobol (Quasi-Monte Carlo).
 * Rejection sampling would break the structure of Sobol otherwise.
 * 
 * @param a (double) First sobol sample
 * @param b (double) Second sobol sample
 */
inline std::array<double, 2> concentric_disk_sample(double a, double b) {
    if (a == 0.0 && b == 0.0){
        return {0.0, 0.0};
    }
    double r, theta;
    if (std::abs(a) > std::abs(b)) {
        r = a;
        theta = (M_PI / 4.0) * (b / a);
    } else {
        r = b;
        theta = (M_PI / 2.0) - (M_PI / 4.0) * (a / b);
    }
    return { r * std::cos(theta), r * std::sin(theta) };
}

/**
 * @brief [SOBOL] Generates a 2D point inside the unit disk from a Sobol' sample.
 *
 * QMC-correct replacement for the rejection-based point_in_unit_disk(). Reads
 * the lens dimensions of the supplied SobolSampler, remaps the two [0,1) values
 * to [-1,1], and feeds them through Shirley's concentric map (no rejection, no
 * bias), which preserves the low-discrepancy structure of the sequence.
 *
 * @param[in] sampler (const SobolSampler&) Per-path Sobol' sampler for this pixel sample
 * @return (std::array<double,2>) A 2D point (x,y) inside the unit disk
 */
static inline std::array<double,2> sobol_point_in_unit_disk(const SobolSampler& sampler){
    const std::array<double,2> s = sampler.lens_sample();   // (u, v) in [0,1)
    const double a = 2.0 * s[0] - 1.0;                       // remap to [-1,1]
    const double b = 2.0 * s[1] - 1.0;
    return concentric_disk_sample(a, b);
}

#endif // RTMATHUTILS_H