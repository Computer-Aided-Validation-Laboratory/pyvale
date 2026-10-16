// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

/**
 * Sobol Quasi-Monte Carlo (QMC) sampler integration
 * 
 * Uses Leonhard Gruenschloss' double-precision random-access Sobol' generator
 * (https://github.com/lgruen/sobol/tree/main/double-precision), which uses the
 * Joe-Kuo direction numbers (new-joe-kuo-6.21201)
 * 
 * Available dimensions: 1024 
 * 
 * -----------------------------------------------------------------------------------
 * DIMENSION ALLOCATION EXPLAINED
 * -----------------------------------------------------------------------------------
 *      dim 0, 1: pixel anti-aliasing jitter (for x, y)
 *      dim 2, 3: thin-lens defocus disk
 *      dim SOBOL_DIM_BOUNCE_BASE (4) + SOBOL_DIMS_PER_BOUNCE * d ...: bounce d
 *              +0 (first dimension): Russian roulette/Fresnel reflect or transmit/path survival (scalar decision)
 *              +1, +2 (second dimension): 2D hemisphere for the diffuse scatter (if it happens)
 * The Joe-Kuo 6.21201 table is specifically constructed for good high-dimensional
 * projections, so using genuine Sobol' dimensions for all 50 bounces (rather than
 * padding) is the recommended, accuracy-maximizing choice
 * 
 * -----------------------------------------------------------------------------------
 * WHY PER-PATH
 * -----------------------------------------------------------------------------------
 * - MT19937 - We can just draw the next value from the generator, regardless of what we do
 * - Sobol - For this to be low-discrepancy, we need to fix one dimension per path (e.g,
 * anti-aliasing is one path, ray bounce is another) and draw randomly from this fixed
 * dimension.
 * If we just did what we do with MT19937, the QMC structure would collapse and we would
 * get worse outcome than with pure MC.
 * So here we circumvent that by assigning a dimension range for each path, then the
 * sample number (k) for a given pixel selects the point in the Sobol sequence.
 * 
 * The generator comes with a "scramble" argument, which we use for per-pixel decorrelation
 * Each pixel derives a deterministic 52-bit scramble from its (x, y) coordinates, so different
 * pixels realize different sequences (no cross-pixel structure) while remaining reproducible run-to-run.
 * 
 * -----------------------------------------------------------------------------------
 * FILES TO MODIFY TO REVERT BACK TO MONTE CARLO:
 * -----------------------------------------------------------------------------------
 * rtray.h
 * rtmaterials.cpp and header
 * rtrender.cpp and header
 * rtmathutils.h (not necessary unless deleting permamently)
 */


#ifndef RTSOBOLSAMPLER_H
#define RTSOBOLSAMPLER_H

#include <cstdint>
#include <array>

#include "sobol/sobol.h" // lgruen/sobol double-precision header (namespace sobol)


// ================================================================================
// Dimension layout constants
// ================================================================================

/// @brief First Sobol' dimension reserved for the pixel anti-aliasing jitter
static constexpr unsigned SOBOL_DIM_PIXEL = 0u;   // uses dims 0, 1
/// @brief First Sobol' dimension reserved for the thin-lens defocus disk sample
static constexpr unsigned SOBOL_DIM_LENS = 2u;    // uses dims 2, 3
/// @brief First Sobol' dimension reserved for the bounce sub-sequence
static constexpr unsigned SOBOL_DIM_BOUNCE_BASE = 4u;
/// @brief Number of Sobol' dimensions consumed per bounce (decision + scatter pair) = 1 + 2 = 3
static constexpr unsigned SOBOL_DIMS_PER_BOUNCE = 3u;
/// @brief Offset (within a bounce's block) of the scalar decision dimension; i.e, we stay within the same dimension
static constexpr unsigned SOBOL_BOUNCE_OFF_DECISION = 0u;
/// @brief Offset (within a bounce's block) of the first BSDF-scatter dimension (2D pair)
static constexpr unsigned SOBOL_BOUNCE_OFF_SCATTER = 1u;

/**
 * @brief Returns the first Sobol' dimension reserved for a given bounce depth.
 *
 * Bounce d consumes the contiguous block
 * [base + SOBOL_DIMS_PER_BOUNCE*d, base + SOBOL_DIMS_PER_BOUNCE*d + SOBOL_DIMS_PER_BOUNCE).
 * The first is the scalar decision dimension (Russian roulette / Fresnel branch), 
 * then the second pairs with it to make the 2D diffuse hemisphere sample. SOBOL_BOUNCE_OFF_SCATTER
 * is the first of the hemisphere pair.
 *
 * @param[in] depth (unsigned) Current ray depth (0 = primary ray's first hit)
 * @return (unsigned) First Sobol' dimension for this bounce
 */
inline constexpr unsigned sobol_bounce_dim(unsigned depth){
    return SOBOL_DIM_BOUNCE_BASE + SOBOL_DIMS_PER_BOUNCE * depth;
}

// ================================================================================
// Per-path Sobol' sampler
// ================================================================================

// Struct size: 2 x 8 = 16 bytes
/**
 * @brief Lightweight per-path Sobol' sampler
 *
 * Holds the Sobol' point index for one pixel sample (the sample number k) and a
 * per-pixel scramble value used to decorrelate pixels. Pulling a value requires
 * an explicit dimension so the QMC structure is preserved; convenience helpers
 * are provided for the fixed allocations above
 *
 * We carry this in RayState through the bounce stack
 */
struct SobolSampler{
    unsigned long long index {0ULL};    ///< Point index in the sequence (the pixel's sample number k)
    unsigned long long scramble {0ULL}; ///< Per-pixel scramble for randomized/scrambled Sobol'

    SobolSampler() = default;

    /**
     * @brief Constructs a sampler for one pixel sample
     * @param[in] index_ (unsigned long long) Sobol' point index (sample number k)
     * @param[in] scramble_ (unsigned long long) Per-pixel scramble value
     */
    SobolSampler(unsigned long long index_, unsigned long long scramble_)
        : index(index_), scramble(scramble_) {}

    /**
     * @brief Draws one Sobol' value from a specific dimension
     *
     * @param[in] dimension (unsigned) Sobol' dimension to read. MUST be < 1024
     *            and MUST be unique per logical draw along the path.
     * @return (double) Quasi-random value in [0, 1)
     */
    inline double get(unsigned dimension) const {
        return sobol::sample(index, dimension, scramble);
    }

    /**
     * @brief Draws the 2D pixel anti-aliasing jitter sample
     * @return (std::array<double,2>) (x, y) in [0, 1)
     */
    inline std::array<double, 2> pixel_jitter() const {
        return { get(SOBOL_DIM_PIXEL), get(SOBOL_DIM_PIXEL + 1) };
    }

    /**
     * @brief Draws the 2D thin-lens defocus disk sample (pre concentric map).
     * @return (std::array<double,2>) (u, v) in [0, 1)
     */
    inline std::array<double, 2> lens_sample() const {
        return { get(SOBOL_DIM_LENS), get(SOBOL_DIM_LENS + 1) };
    }

    /**
     * @brief Draws the scalar decision value (RR / Fresnel branch) for a bounce.
     * @param[in] depth (unsigned) Current ray depth
     * @return (double) Decision value in [0, 1)
     */
    inline double bounce_decision(unsigned depth) const {
        return get(sobol_bounce_dim(depth)+ SOBOL_BOUNCE_OFF_SCATTER);
    }

    /**
     * @brief Draws the 2D scatter sample (diffuse hemisphere) for a bounce.
     * @param[in] depth (unsigned) Current ray depth
     * @return (std::array<double,2>) (r1u, r2) in [0, 1)
     */
    inline std::array<double, 2> bounce_scatter(unsigned depth) const {
        const unsigned d = sobol_bounce_dim(depth) + SOBOL_BOUNCE_OFF_SCATTER;
        return { get(d), get(d + 1) };
    }
};

// ================================================================================
// Per-pixel scramble derivation
// ================================================================================

/**
 * @brief Derives a deterministic per-pixel 52-bit scramble value.
 *
 * Uses a 64-bit integer hash (SplitMix64 finaliser) of the pixel coordinates so
 * each pixel realizes a different scrambled Sobol sequence, while the result is
 * reproducible for the same (x, y). Masked to Matrices::size bits because
 * the sampler only uses the least-significant `size` bits of the scramble.
 *
 * @param[in] px (uint32_t) Pixel x coordinate (column)
 * @param[in] py (uint32_t) Pixel y coordinate (row)
 * @return (unsigned long long) Scramble value in the valid range
 */
inline unsigned long long sobol_pixel_scramble(uint32_t px, uint32_t py){
    // Pack the two coordinates into one 64-bit key
    uint64_t z = (static_cast<uint64_t>(py) << 32) ^ static_cast<uint64_t>(px);
    // SplitMix64 finaliser
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    // Keep only the bits the sampler actually consumes (Matrices::size LSBs)
    const unsigned long long mask = (1ULL << sobol::Matrices::size) - 1ULL;
    return z & mask;
}








#endif // RTSOBOLSAMPLER_H