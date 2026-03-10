// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// InterpolatorFactory.h
#pragma once


#include <memory>
#include <stdexcept>

#include "dicinterpBspline.hpp"
#include "dicinterpHermite.hpp"

/**
 * @brief Factory function for creating interpolator instances.
 *
 * @param routine  Interpolation method to use. Supported values:
 *                 - `"BSPLINE"` — cubic B-spline with prefiltering
 *                 - `"HERMITE"` — piecewise cubic Hermite
 * @param img      Pointer to the image data in row-major order.
 * @param px_hori  Number of horizontal pixels (image width).
 * @param px_vert  Number of vertical pixels (image height).
 *
 * @return A `unique_ptr` to the constructed @ref Interpolator.
 *
 * @throws std::invalid_argument if @p routine is not a recognised method.
 */
inline std::unique_ptr<Interpolator> make_interp(
    const std::string& routine, double* img, int px_hori, int px_vert)
{
    if (routine == "BSPLINE") return std::make_unique<Bspline>(img, px_hori, px_vert);
    if (routine == "HERMITE") return std::make_unique<Hermite>(img, px_hori, px_vert);
    throw std::invalid_argument("Unknown interpolation routine: " + routine);
}
