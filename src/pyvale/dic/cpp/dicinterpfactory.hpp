// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// InterpolatorFactory.h
#pragma once


#include <memory>
#include <stdexcept>

// DIC header files
#include "dicinterpBspline.hpp"
#include "dicinterpHermite.hpp"
#include "dicutil.hpp"

// commoncpp header files
#include "../../commoncpp/util.hpp"
#include "../../commoncpp/img_read.hpp"

/**
 * @brief Factory function for creating interpolator instances.
 *
 * @param routine  Interpolation method to use.
 * @param img      reference to the image data in row-major order.
 *
 * @return A `unique_ptr` to the constructed @ref Interpolator.
 *
 * @throws std::invalid_argument if @p routine is not a recognised method.
 */
inline std::unique_ptr<Interpolator> make_interp(util::InterpRoutine routine, const Image &img) {

    switch (routine) {
        case util::InterpRoutine::BSPLINE:
            return std::make_unique<Bspline>(img);
        case util::InterpRoutine::HERMITE:
            return std::make_unique<Hermite>(img);
    }
    throw std::invalid_argument("Unknown interpolation routine");
}

inline std::unique_ptr<Interpolator> make_interp(util::InterpRoutine routine, const std::string &image_str) {

    Image img = read_img(image_str);
    return make_interp(routine, img);
}
