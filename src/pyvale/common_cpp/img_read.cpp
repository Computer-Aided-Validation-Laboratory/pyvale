// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include <tiffio.h>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "./util.hpp"


Image read_tiff(const std::string &fullpath) {

    TIFF* tif = TIFFOpen(fullpath.c_str(), "r");
    if (!tif) throw std::runtime_error("Failed to open: " + fullpath);
    uint32_t width = 0, height = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    uint16_t bps = 8, spp = 1;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
    Image img;
    img.width = width;
    img.height = height;
    std::vector<uint8_t> scanline(TIFFScanlineSize(tif));
    if (bps == 8) {
        img.type = PixelType::UINT8;
        img.data8.resize(width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            for (uint32_t x = 0; x < width; ++x)
                img.data8[row * width + x] = scanline[x * spp];
        }
    } 
    else if (bps == 16) {
        img.type = PixelType::UINT16;
        img.data16.resize((size_t)width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            auto *p = reinterpret_cast<uint16_t*>(scanline.data());
            for (uint32_t x = 0; x < width; ++x)
                img.data16[row * width + x] = p[x * spp];
        }
    }
    else if (bps == 32) {
        img.type = PixelType::UINT32;
        img.data32.resize((size_t)width * height);
        for (uint32_t row = 0; row < height; ++row) {
            if (TIFFReadScanline(tif, scanline.data(), row) < 0)
                throw std::runtime_error("Failed to read row " + std::to_string(row));
            auto *p = reinterpret_cast<uint32_t*>(scanline.data());
            for (uint32_t x = 0; x < width; ++x)
                img.data32[row * width + x] = p[x * spp];
        }
    }
    else {
        TIFFClose(tif);
        throw std::runtime_error("Unsupported bit depth: " + std::to_string(bps));
    }
    TIFFClose(tif);
    return img;
}
