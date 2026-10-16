// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICFILTER_H
#define DICFILTER_H

// STD library Header files

// Program Header files

// commoncpp header files
#include "../../commoncpp/util.hpp"



/**
 * @brief Apply a Gaussian filter to image.
 *
 * Converts integer images to float32. Float32 and float64 images retain their
 * input precision. The filter is applied using separable horizontal and
 * vertical Gaussian convolutions.
 *
 * @param[in,out] img Image to filter.
 * @param[in] kernel_size Odd Gaussian kernel size (e.g. 3, 5, 7).
 * @param[in] sigma Gaussian standard deviation (> 0).
 *
 * @throws std::runtime_error If @p kernel_size is even or
 *                            @p sigma is not positive.
 */
void apply_filter(Image& img, int kernel_size, double sigma);




#endif // DICFILTER_H
