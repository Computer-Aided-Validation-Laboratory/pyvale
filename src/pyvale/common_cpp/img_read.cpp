// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include <tiffio.h>
#include <vector>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <filesystem>

#include "./util.hpp"
#include "./img_read.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../common_cpp/stb_image.h"

Image read_img(const std::string& fullpath) {


    common_util::Timer time("to read " +std::filesystem::path(fullpath).filename().string() + ":", 2);

    // Find extension
    auto dotPos = fullpath.find_last_of('.');
    if (dotPos == std::string::npos) {
        throw std::runtime_error("File has no extension: " + fullpath);
    }

    std::string ext = fullpath.substr(dotPos);

    // Convert extension to lowercase
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (ext == ".tif" || ext == ".tiff") {
        return read_tiff(fullpath);
    }
    else if (ext == ".bmp") {
        return read_bmp(fullpath);
    }
    else {
        throw std::runtime_error("Unsupported image format: " + ext);
    }
}

Image read_tiff(const std::string &fullpath) {


    TIFF* tif = TIFFOpen(fullpath.c_str(), "r");
    if (!tif) throw std::runtime_error("Failed to open: " + fullpath);
    uint32_t width = 0, height = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    uint16_t bps = 8, spp = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);

    // check whether tiff is int, uint, or f32
    uint16_t format;
    int found = TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &format);

    Image img;
    img.filename = std::filesystem::path(fullpath).filename().string();
    img.width = width;
    img.height = height;
    std::vector<uint8_t> scanline(TIFFScanlineSize(tif));
    if (format == SAMPLEFORMAT_UINT && bps == 8) {
        img.type = PixelType::UINT8;
        img.data8.resize(width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            if (spp == 1) {
                std::memcpy(img.data8.data() + static_cast<size_t>(row) * width,
                            scanline.data(),
                            width * sizeof(uint8_t));
            }
            else {
                for (uint32_t x = 0; x < width; ++x)
                    img.data8[row * width + x] = scanline[x * spp];
            }
        }
    } 
    else if (format == SAMPLEFORMAT_UINT && bps == 16) {
        img.type = PixelType::UINT16;
        img.data16.resize((size_t)width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            auto *p = reinterpret_cast<uint16_t*>(scanline.data());
            if (spp == 1) {
                std::memcpy(img.data16.data() + static_cast<size_t>(row) * width,
                            scanline.data(),
                            static_cast<size_t>(width) * sizeof(uint16_t));
            }
            else {
                for (uint32_t x = 0; x < width; ++x)
                    img.data16[row * width + x] = p[x * spp];
            }
        }
    }
    else if (format == SAMPLEFORMAT_UINT && bps == 32) {
        img.type = PixelType::UINT32;
        img.data32.resize((size_t)width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            auto *p = reinterpret_cast<uint32_t*>(scanline.data());
            if (spp == 1) {
                std::memcpy(img.data32.data() + static_cast<size_t>(row) * width,
                            scanline.data(),
                            static_cast<size_t>(width) * sizeof(uint32_t));
            }
            else {
                for (uint32_t x = 0; x < width; ++x)
                    img.data32[row * width + x] = p[x * spp];
            }
        }
    }
    else if (format == SAMPLEFORMAT_IEEEFP && bps == 32) {
        img.type = PixelType::UINT32F;
        img.data32f.resize((size_t)width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            auto *p = reinterpret_cast<uint32_t*>(scanline.data());
            if (spp == 1) {
                std::memcpy(img.data32f.data() + static_cast<size_t>(row) * width,
                            scanline.data(),
                            static_cast<size_t>(width) * sizeof(uint32_t));
            }
            else {
                for (uint32_t x = 0; x < width; ++x)
                    img.data32f[row * width + x] = p[x * spp];
            }
        }
    }
    else {
        const char* format_name = "unknown";
        switch (format) {
            case SAMPLEFORMAT_UINT:   format_name = "unsigned integer"; break;
            case SAMPLEFORMAT_INT:    format_name = "signed integer"; break;
            case SAMPLEFORMAT_IEEEFP: format_name = "IEEE float"; break;
            case SAMPLEFORMAT_VOID:   format_name = "undefined"; break;
        }

        TIFFClose(tif);
        throw std::runtime_error(
            "Unsupported TIFF pixel type: " +
            std::to_string(bps) + "-bit " + format_name +
            " (SampleFormat=" + std::to_string(format) +
            ", SamplesPerPixel=" + std::to_string(spp) + ").");
    }

    TIFFClose(tif);
    return img;
}

Image read_bmp(const std::string& fullpath) {
    int width = 0, height = 0, channels = 0;

    // First, check if image is 16-bit
    if (stbi_is_16_bit(fullpath.c_str())) {
        uint16_t* raw = stbi_load_16(
            fullpath.c_str(),
            &width,
            &height,
            &channels,
            0
        );

        if (!raw) {
            throw std::runtime_error(
                "Failed to open BMP: " + fullpath + " (" + stbi_failure_reason() + ")"
            );
        }

        Image img;
        img.filename = std::filesystem::path(fullpath).filename().string();
        img.width = static_cast<uint32_t>(width);
        img.height = static_cast<uint32_t>(height);
        img.type = PixelType::UINT16;
        const size_t pixel_count = static_cast<size_t>(width) * height;
        img.data16.resize(pixel_count);

        if (channels == 1) {
            std::memcpy(img.data16.data(), raw, pixel_count * sizeof(uint16_t));
        }
        else {
            // Use first channel only
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    img.data16[y * width + x] =
                        raw[(y * width + x) * channels];
                }
            }
        }

        stbi_image_free(raw);
        return img;
    }
    else {
        // Standard 8-bit BMP
        uint8_t* raw = stbi_load(
            fullpath.c_str(),
            &width,
            &height,
            &channels,
            0
        );

        if (!raw) {
            throw std::runtime_error(
                "Failed to open BMP: " + fullpath + " (" + stbi_failure_reason() + ")"
            );
        }

        Image img;
        img.filename = std::filesystem::path(fullpath).filename().string();
        img.width = static_cast<uint32_t>(width);
        img.height = static_cast<uint32_t>(height);
        img.type = PixelType::UINT8;
        const size_t pixel_count = static_cast<size_t>(width) * height;
        img.data8.resize(pixel_count);

        if (channels == 1) {
            std::memcpy(img.data8.data(), raw, pixel_count * sizeof(uint8_t));
        }
        else {
            // Use first channel only
            #pragma omp parallel for schedule(static)
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    img.data8[y * width + x] =
                        raw[(y * width + x) * channels];
                }
            }
        }

        stbi_image_free(raw);
        return img;
    }
}
