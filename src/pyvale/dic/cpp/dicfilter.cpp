// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <cmath>
#include <algorithm>

// commoncpp header files
#include "../../commoncpp/util.hpp"
#include "./dicfilter.hpp"

// DIC Header files




/**
 * @brief Apply a separable Gaussian filter to floating-point image data.
 *
 * The input vector is filtered in place. The template preserves the scalar
 * type of the supplied image data, allowing the same implementation to be
 * used for float32 and float64 images.
 *
 * @tparam T Floating-point image scalar type.
 * @param[in,out] data Image pixels in row-major order.
 * @param[in] width Image width in pixels.
 * @param[in] height Image height in pixels.
 * @param[in] kernel_size Odd Gaussian kernel size.
 * @param[in] sigma Gaussian standard deviation.
 */
template <typename T>
void filter_float_image(std::vector<T>& data,
                        int width,
                        int height,
                        int kernel_size,
                        double sigma) {

    const int radius = kernel_size / 2;
    const T denom = static_cast<T>(2.0 * sigma * sigma);
    T sum = static_cast<T>(0);
    std::vector<T> kernel(kernel_size);

    for (int i = -radius; i <= radius; ++i) {
        T v = std::exp(static_cast<T>(-(i * i)) / denom);
        kernel[i + radius] = v;
        sum += v;
    }

    for (T& v : kernel)
        v /= sum;

    #pragma omp parallel
    {
        std::vector<T> line(static_cast<size_t>(std::max(width, height)));

        #pragma omp for schedule(static)
        for (int y = 0; y < height; ++y) {
            const size_t row = static_cast<size_t>(y) * width;
            std::copy_n(data.begin() + row, width, line.begin());

            for (int x = 0; x < width; ++x) {
                T accum = 0;
                for (int k = -radius; k <= radius; ++k) {
                    const int xx = std::clamp(x + k, 0, width - 1);
                    accum += line[xx] * kernel[k + radius];
                }
                data[row + x] = accum;
            }
        }

        #pragma omp for schedule(static)
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y)
                line[y] = data[static_cast<size_t>(y) * width + x];

            for (int y = 0; y < height; ++y) {
                T accum = 0;
                for (int k = -radius; k <= radius; ++k) {
                    const int yy = std::clamp(y + k, 0, height - 1);
                    accum += line[yy] * kernel[k + radius];
                }
                data[static_cast<size_t>(y) * width + x] = accum;
            }
        }
    }
}

void apply_filter(Image& img, int kernel_size, double sigma) {

    if (kernel_size < 0 || kernel_size % 2 == 0)
        throw std::runtime_error("image prefilter kernel size must be an odd number");

    if (kernel_size <= 1)
        return;

    if (sigma <= 0.0)
        throw std::runtime_error("sigma must be positive");

    const int width  = static_cast<int>(img.width);
    const int height = static_cast<int>(img.height);

    if (width == 0 || height == 0)
        return;

    switch (img.type) {
    case PixelType::UINT8: {
        img.data32f.resize(width * height);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < img.data8.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data8[i]);
        std::vector<uint8_t>().swap(img.data8);
        break;
    }
    case PixelType::UINT16: {
        img.data32f.resize(width * height);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < img.data16.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data16[i]);
        std::vector<uint16_t>().swap(img.data16);
        break;
    }

    case PixelType::UINT32: {
        img.data32f.resize(width * height);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < img.data32.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data32[i]);
        std::vector<uint32_t>().swap(img.data32);
        break;
    }
    case PixelType::UINT32F:
        filter_float_image(img.data32f, width, height, kernel_size, sigma);
        return;
    case PixelType::FLOAT64:
        filter_float_image(img.data64f, width, height, kernel_size, sigma);
        return;
    }

    img.type = PixelType::UINT32F;
    filter_float_image(img.data32f, width, height, kernel_size, sigma);
}
