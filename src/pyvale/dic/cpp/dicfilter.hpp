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
 * Converts the image to float if required and stores the filtered
 * result in @c data32f. The filter is applied using separable
 * horizontal and vertical Gaussian convolutions.
 *
 * @param[in,out] img Image to filter.
 * @param[in] kernel_size Odd Gaussian kernel size (e.g. 3, 5, 7).
 * @param[in] sigma Gaussian standard deviation (> 0).
 *
 * @throws std::runtime_error If @p kernel_size is even or
 *                            @p sigma is not positive.
 */
void apply_filter(Image& img, int kernel_size, float sigma);




#endif // DICFILTER_H
