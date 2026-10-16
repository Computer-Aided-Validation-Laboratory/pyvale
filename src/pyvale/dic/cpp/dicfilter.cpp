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




void apply_filter(Image& img, int kernel_size, float sigma) {

    if (kernel_size < 0 || kernel_size % 2 == 0)
        throw std::runtime_error("image prefilter kernel size must be an odd number");

    if (kernel_size <= 1)
        return;

    if (sigma <= 0.0f)
        throw std::runtime_error("sigma must be positive");

    const int width  = static_cast<int>(img.width);
    const int height = static_cast<int>(img.height);

    if (width == 0 || height == 0)
        return;

    switch (img.type) {
    case PixelType::UINT8: {
        img.data32f.resize(width * height);
        for (size_t i = 0; i < img.data8.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data8[i]);
        img.data8.clear();
        break;
    }
    case PixelType::UINT16: {
        img.data32f.resize(width * height);
        for (size_t i = 0; i < img.data16.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data16[i]);
        img.data16.clear();
        break;
    }

    case PixelType::UINT32: {
        img.data32f.resize(width * height);
        for (size_t i = 0; i < img.data32.size(); ++i)
            img.data32f[i] = static_cast<float>(img.data32[i]);
        img.data32.clear();
        break;
    }
    case PixelType::UINT32F:
        break;
    }

    // change image type
    img.type = PixelType::UINT32F;

    const int radius = kernel_size / 2;
    const float denom = 2.0f * sigma * sigma;

    float sum = 0.0f;

    std::vector<float> kernel(kernel_size);

    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-(i * i) / denom);
        kernel[i + radius] = v;
        sum += v;
    }

    for (float& v : kernel)
        v /= sum;

    // temporary image buffer
    std::vector<float> tmp(width * height);

    // horizontal pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            float accum = 0.0f;

            for (int k = -radius; k <= radius; ++k) {
                int xx = std::clamp(x + k, 0, width - 1);
                accum += img.data32f[y * width + xx] * kernel[k + radius];
            }

            tmp[y * width + x] = accum;
        }
    }

    // vertical pass
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            float accum = 0.0f;

            for (int k = -radius; k <= radius; ++k) {
                int yy = std::clamp(y + k, 0, height - 1);
                accum += tmp[yy * width + x] * kernel[k + radius];
            }

            img.data32f[y * width + x] = accum;
        }
    }
}
