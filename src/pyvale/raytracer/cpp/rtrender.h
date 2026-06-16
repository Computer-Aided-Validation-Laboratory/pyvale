// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTRENDER_H
#define RTRENDER_H

// STD header files 
#include <array>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <csignal>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtbvh.h"
#include "rtmathutils.h"
#include "rtsignal.h"

// commmon header files
#include "../../common_cpp/progressbar.hpp"
//#include "../../common_cpp/dicsignalhandler.hpp" in the future

// ================================================================================
// Enums for output render configuration - must match Python API
// ================================================================================

enum class RenderColor{
    COLOR = 0,
    GRAYSCALE = 1
};

enum class OutputFormat{
    PPM = 0,
    TIFF_8BIT = 1,
    TIFF_16BIT = 2,
    BMP_8BIT = 3,
    BMP_24BIT = 4
    //NP_BUFFER = 5 // not available yet
};

enum class BitDepth{
    BIT_8 = 8,
    BIT_10 = 10,
    BIT_12 = 12,
    BIT_16 = 16,
};

enum class ChannelCount{
    MONO = 1,
    RGB = 3
};

/// @brief Specifies buffer type for render_image based on the desired bit depth.
enum class BufferType{
    UINT_8 = 0,
    UINT_16 = 1
};

// ================================================================================
// Outputwriter - writing the image pixel buffer to the chosen format
// ================================================================================

// TO DO: add NumPy array buffer

namespace outputwriter{

    // Header function aliases
    using SaveImage8bitFn = void (*)(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);
    
    // This covers anything that will fit in 16-bit uint, so 10, 12, etc.
    using SaveImage16bitFn = void (*)(const std::vector<uint16_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    // Currently selected output backend
    extern SaveImage8bitFn save_image_8bit;
    extern SaveImage16bitFn save_image_16bit;
    extern BitDepth bit_depth;

    /**
     * @brief  Helper that writes 16-bit integers in Little-Endian byte order
     */
    static inline void write_16bit(std::ofstream& image_file,
        uint16_t value);
    
    /**
     * @brief  Helper that writes 16-bit integers in Little-Endian byte order
     */
    static inline void write_32bit(std::ofstream& image_file,
        uint32_t value);
    
    /**
     * @brief Helper that writes a TIFF tag (12-byte IFD tag)
     */
    static inline void write_tag(std::ofstream& image_file,
        uint16_t tag,
        uint16_t type,
        uint32_t count,
        uint32_t value);

    void set_depth(BitDepth depth);

    // ================================================================================
    // BMP and PPM
    // ================================================================================

    /**
     * @brief Saves the stored pixel buffer in BMP (24-bit) format.
     * 
     * This should be fine for both RGB and grayscale renders without any losses.
     * 
     * 24-bit format => 8 bits per channel (R,G,B) => 256 shades per channel.
     * 3 channels, 8-bit depth
     * 
     * Adds ".bmp" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    void saveBMP_24bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);


    /**
     * @brief Saves the stored pixel buffer in BMP (8-bit) format.
     * 
     * This is intended for grayscale renders only.
     * 
     * 24-bit format => 1 byte per pixel to represent colours between [0, 255]
     * => Needs to include a colour table to display colours
     * => Each 1-byte pixel is an index poinitng to a specific 24-bit RGB colour stored in this table.
     * 1 channel, 8-bit depth, grayscale only
     * 
     * Adds ".bmp" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    void saveBMP_8bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    /**
     * @brief Saves the stored pixel buffer in PPM format.
     * 
     * Adds ".ppm" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * PPM = 3 channels, 8-bit depth
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    void savePPM(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    // ================================================================================
    // 8-bit TIFF
    // ================================================================================

   /**
     * @brief Saves the stored pixel buffer in an 8-bit TIFF format, and either 1 or 3 channels.
     * 
     * Adds ".tiff" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * @tparam channel_count (ChannelCount) Number of channels in the image.
     * 
     * @param[in] pixel_buffer (std::vector<uint8_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    template <ChannelCount channel_count>
    void saveTIFF_8bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {
        
        // Set and calculate offsets/values that change depending on the number of channels
        uint16_t samples_per_pixel, photometric_interpretation;
        uint32_t bits_per_sample_count, strip_byte_counts;
        uint32_t bits_per_sample_offset, xresolution_offset, yresolution_offset, strip_offsets;


        constexpr uint16_t ifd_entry_count = 12;
        constexpr uint32_t tiff_header_size = 8;
        constexpr uint32_t ifd_count_size = 2;
        constexpr uint32_t ifd_entry_size = 12;
        constexpr uint32_t next_ifd_offset_size = 4;


        constexpr uint32_t ifd_size = ifd_count_size + ifd_entry_count * ifd_entry_size + next_ifd_offset_size;
        constexpr uint32_t base_extra_data_offset = tiff_header_size + ifd_size;


        if constexpr (channel_count == ChannelCount::MONO) {
            samples_per_pixel = 1; // 1 channel
            photometric_interpretation = 1; // BlackIsZero (0 = black, 255 = white)
            bits_per_sample_count = 1;
            strip_byte_counts = static_cast<uint32_t>(image_width) * image_height;


            // For MONO, BitsPerSample = 8 fits directly in the tag value field,
            // so we do NOT store it out-of-line.
            bits_per_sample_offset = 8; // Stored inline in tag, not used as an offset
            xresolution_offset = base_extra_data_offset; // immediately after IFD
            yresolution_offset = xresolution_offset + 8; // after XResolution RATIONAL
            strip_offsets = yresolution_offset + 8; // after YResolution RATIONAL
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            samples_per_pixel = 3; // 3 channels
            photometric_interpretation = 2; // RGB
            bits_per_sample_count = 3;
            strip_byte_counts = static_cast<uint32_t>(image_width) * image_height * 3;


            bits_per_sample_offset = base_extra_data_offset; // 6 bytes
            xresolution_offset = bits_per_sample_offset + 6; // after 3 SHORTs
            yresolution_offset = xresolution_offset + 8; // after XResolution RATIONAL
            strip_offsets = yresolution_offset + 8; // after YResolution RATIONAL
        }

        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".tiff"); // Concat will do "whatever_path_we_have.tiff", which is what we want as we already pass the name of the image file, we just need to add the extension
        //std::cout << "Output filepath:" << output_filepath << std::endl; // For checking if path is generated correctly


        // std::ios::binary is important for TIFF so Windows doesn't corrupt the file by changing \n to \r\n
        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // Write the TIFF Header (8 bytes)
        // "II" (little-endian), 42 (magic number), 8 (offset to first IFD)
        image_file.write("II\x2A\x00\x08\x00\x00\x00", 8);


        // Write the IFD (Image File Directory)
        write_16bit(image_file, 12); // Number of directory entries


        // TIFF tags must be written in strictly ascending order of the Tag ID
        write_tag(image_file, 0x0100, 4, 1, image_width); // ImageWidth (LONG)
        write_tag(image_file, 0x0101, 4, 1, image_height); // ImageLength (LONG)

        if constexpr (channel_count == ChannelCount::MONO) {
            // BitsPerSample = 8 stored directly in the tag value field
            write_tag(image_file, 0x0102, 3, 1, 8);
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            // BitsPerSample (SHORTx3 -> Pointer offset)
            write_tag(image_file, 0x0102, 3, bits_per_sample_count, bits_per_sample_offset);
        }

        write_tag(image_file, 0x0103, 3, 1, 1); // Compression (SHORT -> 1 = None)
        write_tag(image_file, 0x0106, 3, 1, photometric_interpretation); // PhotometricInterpretation
        write_tag(image_file, 0x0111, 4, 1, strip_offsets); // StripOffsets
        write_tag(image_file, 0x0115, 3, 1, samples_per_pixel); // SamplesPerPixel
        write_tag(image_file, 0x0116, 4, 1, image_height); // RowsPerStrip (LONG)
        write_tag(image_file, 0x0117, 4, 1, strip_byte_counts); // StripByteCounts
        write_tag(image_file, 0x011A, 5, 1, xresolution_offset); // XResolution
        write_tag(image_file, 0x011B, 5, 1, yresolution_offset); // YResolution
        write_tag(image_file, 0x0128, 3, 1, 2); // ResolutionUnit (2 = Inches)

        write_32bit(image_file, 0); // Offset to next IFD (0 indicates end of IFDs)

        // Write data that exceeds the 4-byte limit in the IFD value fields
        if constexpr (channel_count == ChannelCount::MONO) {
            // No out-of-line BitsPerSample block is needed for MONO
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            // BitsPerSample
            write_16bit(image_file, 8);
            write_16bit(image_file, 8);
            write_16bit(image_file, 8);
        }

        // XResolution (Numerator / Denominator = 72 / 1)
        write_32bit(image_file, 72); write_32bit(image_file, 1);            
        // YResolution (Numerator / Denominator = 72 / 1)
        write_32bit(image_file, 72); write_32bit(image_file, 1);
        
        // Write pixel data
        if constexpr (channel_count == ChannelCount::MONO) {
            // Assumes pixel_buffer is still RGB-shaped with identical grayscale values (it is) and we write only the first channel
            for (int y = 0; y < image_height; ++y) {
                for (int x = 0; x < image_width; ++x) {
                    const int i = (y * image_width + x) * 3;
                    image_file.put(static_cast<char>(pixel_buffer[i]));
                }
            }
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            image_file.write(reinterpret_cast<const char*>(pixel_buffer.data()), pixel_buffer.size());
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    // ================================================================================
    // 16-bit TIFF
    // ================================================================================

    inline uint16_t encode_to_16bit(uint16_t logical_code, BitDepth bit_depth) {
        switch (bit_depth) {
            case BitDepth::BIT_8:
                // 8-bit code in 0..255, shift to the top 8 bits
                return static_cast<uint16_t>(logical_code << 8);
            case BitDepth::BIT_10:
                // 10-bit code in 0..1023, shift to the top 10 bits
                return static_cast<uint16_t>(logical_code << 6);
            case BitDepth::BIT_12:
                // 12-bit code in 0..4095, shift to the top 12 bits
                return static_cast<uint16_t>(logical_code << 4);
            case BitDepth::BIT_16:
            default:
                // Already full 16-bit code
                return logical_code;
        }
    }
    /**
     * @brief Saves the stored pixel buffer in anj 16-bit TIFF format, and either 1 or 3 channels.
     * 
     * Currently, the only function capable of saving images with 8/10/12/16-bit depths.
     * It DOES NOT scale the values to match the 16-bit depth.
     * 
     * Adds ".tiff" extension to the passed filepath, opens the file, writes appropriate tags,
     * converts the buffer and writes it to the file.
     * 
     * NOTE: Currently does NOT support RBG (3-channel) in anything other than 16-bits
     * 
     * @tparam channel_count (ChannelCount) Number of channels in the image.
     * 
     * @param[in] pixel_buffer (std::vector<uint16_t>) Buffer storing pixel colour values ready to write, either in RGB or grayscale.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    template <ChannelCount channel_count>
    void saveTIFF_16bit(const std::vector<uint16_t>& pixel_buffer,
    const int image_height,
    const int image_width,
    std::filesystem::path& output_filepath) {

        BitDepth selected_bit_depth = outputwriter::bit_depth; // Use the bit depth from namespace
        const uint16_t bits_per_sample = static_cast<uint16_t>(selected_bit_depth);

        // Set parameters that depend on whether this is MONO or RGB channel
        uint16_t samples_per_pixel, photometric_interpretation;
        uint32_t bits_per_sample_count, strip_byte_counts;
        uint32_t bits_per_sample_offset, xresolution_offset, yresolution_offset, strip_offsets;

        constexpr uint16_t ifd_entry_count = 13;
        constexpr uint32_t tiff_header_size = 8;
        constexpr uint32_t ifd_count_size = 2;
        constexpr uint32_t ifd_entry_size = 12;
        constexpr uint32_t next_ifd_offset_size = 4;
        constexpr uint32_t ifd_size = ifd_count_size + ifd_entry_count * ifd_entry_size + next_ifd_offset_size;
        constexpr uint32_t base_extra_data_offset = tiff_header_size + ifd_size;

        if constexpr (channel_count == ChannelCount::MONO) {
            samples_per_pixel = 1; // 1 channel
            photometric_interpretation = 1; // BlackIsZero
            bits_per_sample_count = 1;

            const uint32_t pixel_count =
                static_cast<uint32_t>(image_width) *
                static_cast<uint32_t>(image_height);
            const uint32_t total_bits = pixel_count * static_cast<uint32_t>(bits_per_sample);
            strip_byte_counts = (total_bits + 7u) / 8u; // ceil(total_bits / 8)

            // For MONO, BitsPerSample fits directly in the tag value field, so we do NOT store it out-of-line
            bits_per_sample_offset = bits_per_sample; // Stored inline in tag, not used as an offset
            xresolution_offset = base_extra_data_offset; // immediately after IFD
            yresolution_offset = xresolution_offset + 8; // after XResolution RATIONAL
            strip_offsets = yresolution_offset + 8; // after YResolution RATIONAL
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            samples_per_pixel = 3;
            photometric_interpretation = 2; // RGB
            bits_per_sample_count = 3;

            const uint32_t pixel_count =
                static_cast<uint32_t>(image_width) *
                static_cast<uint32_t>(image_height);
            const uint32_t total_bits =
                pixel_count * static_cast<uint32_t>(samples_per_pixel) * static_cast<uint32_t>(bits_per_sample);
            strip_byte_counts = (total_bits + 7u) / 8u; // ceil(total_bits / 8)

            // RGB BitsPerSample has 3 SHORT values written out-of-line
            bits_per_sample_offset = base_extra_data_offset; // 6 bytes
            xresolution_offset = bits_per_sample_offset + 6; // After 3 SHORTs
            yresolution_offset = xresolution_offset + 8; // After XResolution RATIONAL
            strip_offsets = yresolution_offset + 8; // After YResolution RATIONAL
        }

        output_filepath.concat(".tiff");

        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // TIFF header: II, 42, first IFD at offset 8
        image_file.write("II\x2A\x00\x08\x00\x00\x00", 8);

        // IFD entry count
        write_16bit(image_file, ifd_entry_count);

        // Tags in ascending order
        write_tag(image_file, 0x0100, 4, 1, image_width); // ImageWidth
        write_tag(image_file, 0x0101, 4, 1, image_height); // ImageLength
        if constexpr (channel_count == ChannelCount::MONO) {
            write_tag(image_file, 0x0102, 3, 1, bits_per_sample); // BitsPerSample inline
        } else {
            write_tag(image_file, 0x0102, 3, 3, bits_per_sample_offset); // BitsPerSample offset
        }
        write_tag(image_file, 0x0103, 3, 1, 1); // Compression = None
        write_tag(image_file, 0x0106, 3, 1, photometric_interpretation); // PhotometricInterpretation
        write_tag(image_file, 0x0111, 4, 1, strip_offsets); // StripOffsets
        write_tag(image_file, 0x0115, 3, 1, samples_per_pixel); // SamplesPerPixel
        write_tag(image_file, 0x0116, 4, 1, image_height); // RowsPerStrip
        write_tag(image_file, 0x0117, 4, 1, strip_byte_counts); // StripByteCounts
        write_tag(image_file, 0x011A, 5, 1, xresolution_offset); // XResolution
        write_tag(image_file, 0x011B, 5, 1, yresolution_offset); // YResolution
        write_tag(image_file, 0x011C, 3, 1, 1); // PlanarConfiguration = Chunky
        write_tag(image_file, 0x0128, 3, 1, 2); // ResolutionUnit = Inches

        // Next IFD = none
        write_32bit(image_file, 0);

        // Extra data area
        // No out-of-line BitsPerSample block is needed for MONO
        if constexpr (channel_count == ChannelCount::RGB) {
            // 3 channels => 3 writes
            write_16bit(image_file, bits_per_sample);
            write_16bit(image_file, bits_per_sample);
            write_16bit(image_file, bits_per_sample);
        }

        // XResolution = 72/1
        write_32bit(image_file, 72);
        write_32bit(image_file, 1);

        // YResolution = 72/1
        write_32bit(image_file, 72);
        write_32bit(image_file, 1);

        // Pixel data
        if constexpr (channel_count == ChannelCount::MONO) {
            // Assumes pixel_buffer is still RGB-shaped with identical grayscale values (it is) and we write only the first channel
            // IMPORTANT: For BIT_8 / BIT_10 / BIT_12, TIFF expects the samples to be packed according to BitsPerSample
            // Writing raw uint16_t values here would produce a formally inconsistent TIFF and many viewers will show it as black
            const uint32_t pixel_count =
                static_cast<uint32_t>(image_width) *
                static_cast<uint32_t>(image_height);
            const uint32_t total_bits = pixel_count * static_cast<uint32_t>(bits_per_sample);
            const uint32_t packed_byte_count = (total_bits + 7u) / 8u;

            std::vector<uint8_t> packed_data(packed_byte_count, 0u);

            uint32_t bit_position = 0u;

            for (int y = 0; y < image_height; ++y) {
                for (int x = 0; x < image_width; ++x) {
                    const int i = (y * image_width + x) * 3;

                    // Pixel buffer stores logical sample codes already in the selected range:
                    // BIT_8  => [0, 255]
                    // BIT_10 => [0, 1023]
                    // BIT_12 => [0, 4095]
                    // BIT_16 => [0, 65535]
                    uint16_t sample = pixel_buffer[i];

                    // Mask off any accidental higher bits
                    if (bits_per_sample < 16) {
                        sample &= static_cast<uint16_t>((1u << bits_per_sample) - 1u);
                    }

                    // Pack the sample MSB-first into the output byte stream
                    for (uint16_t b = 0; b < bits_per_sample; ++b) {
                        const uint32_t out_byte = bit_position / 8u;
                        const uint32_t out_bit = bit_position % 8u;

                        const uint16_t bit = (sample >> (bits_per_sample - 1u - b)) & 1u;
                        packed_data[out_byte] |= static_cast<uint8_t>(bit << (7u - out_bit));

                        ++bit_position;
                    }
                }
            }

            image_file.write(
                reinterpret_cast<const char*>(packed_data.data()),
                static_cast<std::streamsize>(packed_data.size()));
        }
        else if constexpr (channel_count == ChannelCount::RGB) {
            // RGB is stored as chunky/interleaved samples: [R G B R G B R G B...]
            // Each sample is packed using the selected BitsPerSample with no scaling
            const uint32_t pixel_count =
                static_cast<uint32_t>(image_width) *
                static_cast<uint32_t>(image_height);
            const uint32_t total_bits =
                pixel_count * 3u * static_cast<uint32_t>(bits_per_sample);
            const uint32_t packed_byte_count = (total_bits + 7u) / 8u;

            std::vector<uint8_t> packed_data(packed_byte_count, 0u);

            uint32_t bit_position = 0u;

            for (int y = 0; y < image_height; ++y) {
                for (int x = 0; x < image_width; ++x) {
                    const int i = (y * image_width + x) * 3;

                    uint16_t r = pixel_buffer[i];
                    uint16_t g = pixel_buffer[i + 1];
                    uint16_t b = pixel_buffer[i + 2];

                    // Mask off any accidental higher bits
                    if (bits_per_sample < 16) {
                        const uint16_t mask = static_cast<uint16_t>((1u << bits_per_sample) - 1u);
                        r &= mask;
                        g &= mask;
                        b &= mask;
                    }

                    const uint16_t samples[3] = {r, g, b};

                    // Pack R, then G, then B for each pixel (Chunky / interleaved layout) 
                    for (int c = 0; c < 3; ++c) {
                        for (uint16_t bit_idx = 0; bit_idx < bits_per_sample; ++bit_idx) {
                            const uint32_t out_byte = bit_position / 8u;
                            const uint32_t out_bit = bit_position % 8u;

                            const uint16_t bit = (samples[c] >> (bits_per_sample - 1u - bit_idx)) & 1u;
                            packed_data[out_byte] |= static_cast<uint8_t>(bit << (7u - out_bit));
                            ++bit_position;
                        }
                    }
                }
            }
            image_file.write(reinterpret_cast<const char*>(packed_data.data()),
                static_cast<std::streamsize>(packed_data.size()));
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    /**
     * @brief Setter for the appropriate output writing function based on the passed configuration.
     */
    void set(OutputFormat output_format, ChannelCount channel_count);

    // Overloaded save wrappers for the pointers. Why:
    // 1. We keep runtime selection based on the Python API
    // 2. Correct overload gets chosen at compile time (based on the pixel buffer)
    // (https://www.ibm.com/docs/en/i/7.4.0?topic=only-resolving-addresses-overloaded-functions-c)
    // 3. While keeping nicer syntax

    /// @brief Saves the pixel array in the chosen output format, which supports 8-bit depth.
    void save_image(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);
    
    /// @brief Saves the pixel array in the chosen output format, supporting up to 16-bit depth.
    void save_image(const std::vector<uint16_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

}

// ================================================================================
// Return ray color 
// ================================================================================
namespace renderer{
    /// @brief Maximum depth for the secondary rays.
    extern int MAX_DEPTH;
    /// @brief Background colour for the scene.
    extern EiVector3d background_color;
    /// @brief Maximum integer range used to multiply pixel colour values to get the desired bit depth.
    extern uint32_t max_code_range;
    // Alias for the renderer function pointer
    using RenderingFunction = void (*) (const EiVector3d& camera_center,
        const EiVector3d& pixel_00_center,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
        const TLAS& TLAS,
        const int image_height,
        const int image_width,
        const int number_of_samples,
        const double scene_ri,
        std::filesystem::path& output_filepath);

    extern RenderingFunction render_image;

    /**
     * @brief Returns a procedural sky color for a ray direction.
     * 
     * Produces a simple vertical white-to-blue gradient based on the y-component
     * of the ray direction.
     * 
     * Note: This has been used for a very long time as the default background (stemming
     * from Ray Tracing in One Weekend), but realistically, we probably do not need that
     * for a virtual laboratory.
     * Replaced with a solid background colour; feel free to remove it, or let users 
     * pick between this and a solid background, etc.
     * 
     * @param[in] ray (const Ray&) Input ray
     * 
     * @return (EiVector3d) RGB sky color corresponding to the ray direction.
     */
    inline EiVector3d ray_blue_sky(const Ray& ray){
        double a = 0.5 * (ray.direction(1) + 1.0);
        static EiVector3d white, blue;
        white << 1.0, 1.0, 1.0;
        blue << 0.5, 0.7, 1.0;
        return (1.0 - a) * white + a * blue;
    }

    /**
     * @brief Processes a single primary primary ray to find its corresponding pixel colour via iterative tracking of secondary rays.
     * 
     * Creates a thread-safe stack of RayState objects, then dispatches to intersect_TLAS to find the nearest intersection.
     * Adds a blue sky colour if there is no intersection. Otherwise, it checks and applies material absorption,
     * and verifies the InteriorList for nested dielectrics where applicable, to evaluate if the hit is true or false.
     * Finally, dispatches to the appropriate material colour and adds the output iteratively to the stack.
     * The stack is traversed until empty, terminated early due to Russian rulette, or the MAX_DEPTH is reached.
     * 
     * @param[in] primary_ray (const Ray&) The primary Ray with direction and origin determined by its corresponing pixel in render_image.
     * @param[in] scene_ri (const double) Refractive index of the scene (ambient medium) which is used as a fallback value in shading.
     * @param[in] TLAS (const &TLAS) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
     * 
     * @return (EiVector3d) A 3D, row-major Eigen vector storing the final colour of the pixel in the (r,g,b) format.
     * 
     */
    EiVector3d return_ray_color_stack(const Ray& primary_ray,
        const double scene_ri,
        const TLAS& TLAS);

    /**
     * @brief Iterates over each pixel in the viewport to shoot rays and retrieve their colours.
     * 
     * This is template-based to avoid having to branch etc. based on whether the output image is in grayscale or not.
     * Creates a buffer of pixels, then goes over the image height and width to determine the ray direction and origin
     * for each pixel. Dispatches the ray to return_ray_color_stack and retrieves the final colour value, which is then
     * averaged over n samples if anti-aliasing is on.
     * Finally, it clamps the colour between [0,1] and either converts it to grayscale or RGB, then stores it in the buffer, which
     * is dispatched to an appropriate output writer.
     * 
     * @param[in] camera_center (const EiVector3d&) Row-major 3D Eigen vector with the [x,y,z] coordinates of the chosen camera.
     * @param[in] pixel_00_center (const EiVector3d&) Row-major 3D Eigen vector with the [x,y,z] coordinates of the (0,0) (upper left) pixel of the viewport corresponding to the passed camera.
     * @param[in] matrix_pixel_spacing (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the vectors defining the horizontal and vertical spacing of the pixels in the viewport.
     *          They are defined towards the right, and downward.
     * @param[in] matrix_defocus_disc (const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>&) Matrix storing the horizontal and vertical defocus dics basis vectors for the thin lens approximation;
     *          it will be full of zeros if not DoF is not used.
     * @param[in] TLAS (const TLAS&) Top level acceleration structure (BVH) storing smaller BVHs for each mesh in its nodes.
     * @param[in] image_height (const int) Output image height
     * @param[in] image_width (const int) Output image width
     * @param[in] number_of_samples (const int) Number of samples used for anti-aliasing
     * @param[in] scene_ri (const double) Refractive index of the scene (ambient medium) which is used as a fallback value in shading.
     * @param[in] output_filepath (std::filesystem::path&) Filepath for the output image that will be written into, already storing the name
     *            of the image, but without extension. For example, /home/user1/pyvale-output/rtimage_1_cam0
     */
    template <RenderColor color, BufferType buffer_type>
    void render_img(const EiVector3d& camera_center,
        const EiVector3d& pixel_00_center,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
        const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
        const TLAS& TLAS,
        const int image_height,
        const int image_width,
        const int number_of_samples,
        const double scene_ri,
        std::filesystem::path& output_filepath) {

        // DEV NOTES:
        // Forgive me father for I have sinned, but that was the only neat solution I could find for this that is
        // a) Compile time
        // b) Not requiring very many metatemplates (example here: https://www.reddit.com/r/cpp_questions/comments/1dtr4vo/how_do_create_a_conditional_type_based_on/)
        // c) Or rewriting this function twice separately for 8- and 16-bits
        // Note if we ever add more buffer types, however, we'd either have to stack these or do something else
        using BufferDType = std::conditional_t<buffer_type == BufferType::UINT_8, uint8_t, uint16_t>;

        std::vector<BufferDType> buffer;
        buffer.resize(image_width * image_height * 3); // Preallocate memory for the image buffer

        // Pull invariants out of the loops to avoid re-computing these
        const EiVector3d pixel_row_0 = matrix_pixel_spacing.row(0);
        const EiVector3d pixel_row_1 = matrix_pixel_spacing.row(1);
        const EiVector3d defocus_row_0 = matrix_defocus_disc.row(0);
        const EiVector3d defocus_row_1 = matrix_defocus_disc.row(1);
        static const double color_scaling = 1.0 /number_of_samples; // Multiplication is faster than division, so we pre-divide it before looping

        // Progress bar - useful for higher anti-aliasing and/or refractive scenes
        std::string bar_title = "Processing scanlines:";
        ProgressBar pbar(bar_title, image_height);
        std::atomic<int> current_progress = 0;

        #pragma omp parallel for shared(stop_request) schedule(dynamic) 
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                EiVector3d pixel_color = EiVector3d::Zero();
                for (int k = 0; k < number_of_samples; k++) {
                    // Exit the main loop in rtmain when CTRL+C is pressed
                    if (stop_request) continue;

                    double offset[2] = { random_double() - 0.5, random_double() - 0.5 };
                    EiVector3d pixel_sample = pixel_00_center +
                        (i + offset[0]) * pixel_row_0 + (j + offset[1]) * pixel_row_1;
                        // Below is true for pinhole camera
                        //EiVector3d ray_origin = camera_center;
                        //EiVector3d ray_direction = pixel_sample - camera_center;

                        // Thin lens approximation camera
                        std::array<double, 2> defocus_disc_offset = point_in_unit_disk();
                        EiVector3d defocus_disc_sample = defocus_disc_offset[0] * defocus_row_0 + defocus_disc_offset[1] * defocus_row_1;
                        EiVector3d ray_origin = camera_center + defocus_disc_sample; // ray direction in thin lens approx
                        EiVector3d ray_direction = pixel_sample - ray_origin; // ray direction in thin lens approx
                        Ray current_ray{ ray_origin, ray_direction.stableNormalized() }; 

                        //Clamp fireflies - optional, makes images less bright
                        //EiVector3d sample = return_ray_color_stack(current_ray, scene_ri, TLAS);
                        //double lum = 0.2126*sample.x() + 0.7152*sample.y() + 0.0722*sample.z();
                        //static constexpr double MAX_LUM = 10.0; // Tune per scene; hoist this out of the loop if using 
                        //if (lum > MAX_LUM) sample *= MAX_LUM / lum;
                        //pixel_color += sample;
                        pixel_color += renderer::return_ray_color_stack(current_ray, scene_ri, TLAS);
                
            }
                int px_idx = (i + j * image_width) * 3;
                // Divide by the number of samples to get the mean colour
                pixel_color = pixel_color * color_scaling;
                // Clamp each channel to [0,1]
                pixel_color = pixel_color.cwiseMax(0.0).cwiseMin(1.0); 
                if constexpr (color == RenderColor::GRAYSCALE) {
                    // Convert to a single-channel grayscale
                    const double gray = 0.2126 * pixel_color[0] + 0.7152 * pixel_color[1] + 0.0722 * pixel_color[2];
                    // Scale to bytes
                    BufferDType gray_byte; // uint8_t or uint16_t
                    if constexpr (buffer_type == BufferType::UINT_8){
                        gray_byte = static_cast<BufferDType>(gray * 255.999);
                    }
                    else{
                        // Anything in the 8-16 bit range - we have pre-set the max_code_range to scale accordingly
                        gray_byte = static_cast<BufferDType>(gray * max_code_range + 0.5); 
                    }
                    buffer[px_idx] = gray_byte;
                    buffer[px_idx + 1] = gray_byte;
                    buffer[px_idx + 2] = gray_byte;
                }
                else if constexpr (color == RenderColor::COLOR) {
                    // Scale to bytes
                    if constexpr(buffer_type == BufferType::UINT_8){
                        pixel_color *= 255.999;
                        buffer[px_idx] = static_cast<uint8_t>(pixel_color.x());
                        buffer[px_idx + 1] = static_cast<uint8_t>(pixel_color.y());
                        buffer[px_idx + 2] = static_cast<uint8_t>(pixel_color.z());
                    }
                    else{  
                        pixel_color = pixel_color.cwiseMax(0.0).cwiseMin(1.0);
                        buffer[px_idx] = static_cast<uint16_t>(pixel_color.x() * max_code_range + 0.5);
                        buffer[px_idx + 1] = static_cast<uint16_t>(pixel_color.y() * max_code_range + 0.5);
                        buffer[px_idx + 2] = static_cast<uint16_t>(pixel_color.z() * max_code_range + 0.5);
                    }  
                }
            }
            
            // Update progress bar
            int progress = current_progress.fetch_add(1);
            if (omp_get_thread_num()==0) pbar.update(progress);
        }
        // Finish progress bar
        int progress = current_progress;
        pbar.finish();
        // Write the buffer in whatever output format we want
        outputwriter::save_image(buffer, image_height, image_width, output_filepath);
    };

    /**
     * @brief Setter for the MAX_DEPTH of the secondary ray bounces.
     * 
     * @param[in] max_depth (const int). Desired maximum depth. Higher is needed for refractive materials.
     */
    void set_depth(int max_depth);

     /**
     * @brief Setter for the background colour of the scene.
     * 
     * @param[in] color (const EiVector3d&) Desired background colour as an RGB triplet in the [0,1] range.
     */
    void set_background(const EiVector3d& color);

    void set_max_code_range(const BitDepth bit_depth);

    void set_rendering_function(const bool grayscale,
        const BitDepth bit_depth,
        const OutputFormat output_format);
}

// ================================================================================
// render_image template for colour and grayscale
// ================================================================================


// ================================================================================
// Mock ray shooter for debug
// ================================================================================

/**
 * @brief Quick debug function that shoots a single ray into TLAS so it can be tracked, either to
 * compare with analytical solution or troubleshoot certain bugs without having to go through the entire image.
 * Note: The ray needs to be hard-coded into this function.
 */
void mock_ray_shoot(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath);

#endif // RTRENDER_H