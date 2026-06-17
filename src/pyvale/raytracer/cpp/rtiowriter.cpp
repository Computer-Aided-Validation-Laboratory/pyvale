// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include "rtiowriter.h"

// ================================================================================
// Writing the output
// ================================================================================
namespace outputwriter{

    SaveImage8bitFn save_image_8bit = nullptr;
    SaveImage16bitFn save_image_16bit = nullptr;
    BitDepth bit_depth = BitDepth::BIT_8;

    // Helper that writes 16-bit integers in Little-Endian byte order
    static inline void write_16bit(std::ofstream& image_file,
        uint16_t value){

        uint8_t bytes[2] = {static_cast<uint8_t>(value & 0xFF), static_cast<uint8_t>(value >> 8) };
        image_file.write(reinterpret_cast<const char*>(bytes), 2);
    }

    // Helper that writes 32-bit integers in Little-Endian byte order
    static inline void write_32bit(std::ofstream& image_file,
        uint32_t value){

        uint8_t bytes[4] = { static_cast<uint8_t>(value & 0xFF),
        static_cast<uint8_t>((value >> 8) & 0xFF),
        static_cast<uint8_t>((value >> 16) & 0xFF),
        static_cast<uint8_t>((value >> 24) & 0xFF) };
        image_file.write(reinterpret_cast<const char*>(bytes), 4);
    }

    // Helper that writes a TIFF tag (12-byte IFD tag)
    static inline void write_tag(std::ofstream& image_file,
        uint16_t tag,
        uint16_t type,
        uint32_t count,
        uint32_t value) {

        write_16bit(image_file, tag);
        write_16bit(image_file, type);
        write_32bit(image_file, count);
        write_32bit(image_file, value);
    }

    void set_depth(BitDepth depth){
        bit_depth = depth;
    }

    void saveBMP_24bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {
            
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".bmp");

        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // BMP rows must be a multiple of 4 bytes. Calculate necessary padding
        int row_bytes = image_width * 3; // 24-bit mode => 3 bytes/pixel
        int padding = (4 - (row_bytes % 4)) % 4;
        uint32_t pixel_data_size = (row_bytes + padding) * image_height;
        uint32_t file_size = 54 + pixel_data_size; // 14 (file header) + 40 (info header) + pixels

        // Write the BMP file header (14 bytes)
        image_file.write("BM", 2); // Signature
        write_32bit(image_file, file_size); // File Size
        write_32bit(image_file, 0); // Reserved
        write_32bit(image_file, 54); // Offset to pixel data (14 + 40)

        // Write the DIB Info Header (40 bytes)
        write_32bit(image_file, 40); // Info Header size
        write_32bit(image_file, image_width); // Image width
        write_32bit(image_file, image_height); // Image height (positive = bottom-up)
        write_16bit(image_file, 1); // Color planes (must be 1)
        write_16bit(image_file, 24); // Bits per pixel (24 for RGB)
        write_32bit(image_file, 0); // Compression (0 = None / BI_RGB)
        write_32bit(image_file, pixel_data_size); // Image size (including padding)
        write_32bit(image_file, 2835); // X pixels per meter (approx 72 DPI)
        write_32bit(image_file, 2835); // Y Pixels per meter (approx 72 DPI)
        write_32bit(image_file, 0); // Total colors (0 = default)
        write_32bit(image_file, 0); // Important colors (0 = default)

        // Write pixel data
        uint8_t pad_bytes[3] = {0, 0, 0};

        // Iterate backwards through rows for bottom-up writing
        for (int y = image_height - 1; y >= 0; --y) {
            for (int x = 0; x < image_width; ++x) {
                int i = (y * image_width + x) * 3;
                
                // Extract RGB from buffer and write as BGR
                char b = static_cast<char>(pixel_buffer[i + 2]);
                char g = static_cast<char>(pixel_buffer[i + 1]);
                char r = static_cast<char>(pixel_buffer[i]);

                image_file.put(b);
                image_file.put(g);
                image_file.put(r);
            }
            // Add required 4-byte alignment padding at the end of the row
            if (padding > 0) {
                image_file.write(reinterpret_cast<const char*>(pad_bytes), padding);
            }
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    void saveBMP_8bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {
            
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".bmp");

        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // 8-bit BMP rows must be a multiple of 4 bytes. Calculate padding.
        int row_bytes = image_width; // 1 byte per pixel in 8-bit mode
        int padding = (4 - (row_bytes % 4)) % 4;
        uint32_t pixel_data_size = (row_bytes + padding) * image_height;
        
        // File sizes
        uint32_t header_size = 14;
        uint32_t info_header_size = 40;
        uint32_t palette_size = 256 * 4; // 256 colors * 4 bytes (B, G, R, Reserved)
        uint32_t offset_to_pixels = header_size + info_header_size + palette_size;
        uint32_t file_size = offset_to_pixels + pixel_data_size; 

        // Write the BMP file header (14 bytes)
        image_file.write("BM", 2); // Signature
        write_32bit(image_file, file_size); // File size
        write_32bit(image_file, 0); // Reserved
        write_32bit(image_file, offset_to_pixels); // Offset to pixel data (1078)

        // Write the DIB info header (40 bytes)
        write_32bit(image_file, info_header_size); // Info header size
        write_32bit(image_file, image_width); // Image width
        write_32bit(image_file, image_height); // Image jeight (Positive = bottom-up)
        write_16bit(image_file, 1); // Color planes (must be 1)
        write_16bit(image_file, 8); // Bits per pixel (8 for indexed)
        write_32bit(image_file, 0); // Compression (0 = None)
        write_32bit(image_file, pixel_data_size); // Image size (including padding)
        write_32bit(image_file, 2835); // X pixels per meter (approx 72 DPI)
        write_32bit(image_file, 2835); // Y pixels per meter (approx 72 DPI)
        write_32bit(image_file, 256); // Total colors in palette (256)
        write_32bit(image_file, 256); // Important colors (256)

        // Write the 256-colour grayscale palette (1024 bytes)
        // This is needed so the viewer can map 8-bit index to a screen colour
        // Palette format: Blue, Green, Red, Reserved (0)
        for (int i = 0; i < 256; ++i) {
            char gray_val = static_cast<char>(i);
            image_file.put(gray_val); // Blue
            image_file.put(gray_val); // Green
            image_file.put(gray_val); // Red
            image_file.put(0); // Reserved byte
        }

        // Write pixel data
        uint8_t pad_bytes[3] = {0, 0, 0};

        // Iterate backwards through rows for bottom-up writing
        for (int y = image_height - 1; y >= 0; --y) {
            for (int x = 0; x < image_width; ++x) {
                // Buffer is formatted as 3 bytes per pixel (R, G, B)
                int i = (y * image_width + x) * 3;
                
                // In grayscale mode, R=G=B, so we just grab the first byte (Red channel)
                // This acts as the 8-bit index into our grayscale palette
                char pixel_index = static_cast<char>(pixel_buffer[i]);
                image_file.put(pixel_index);
            }
            // Add required 4-byte alignment padding at the end of the row
            if (padding > 0) {
                image_file.write(reinterpret_cast<const char*>(pad_bytes), padding);
            }
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    void savePPM(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath){
        
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".ppm");
        //std::cout << "Output filepath:" << output_filepath << std::endl; // For checking if path is generated correctly

        std::ofstream image_file;

        image_file.open(output_filepath);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
        image_file.write(reinterpret_cast<const char*>(pixel_buffer.data()), pixel_buffer.size());

        image_file.close();
        std::cout << "\r Done. \n";
    }

    // Setter
    void set(OutputFormat output_format, ChannelCount channel_count){

        // Reset both so only the valid one is active
        save_image_8bit = nullptr;
        save_image_16bit = nullptr;

        switch (output_format){
            case OutputFormat::PPM:
                save_image_8bit = &savePPM;
                break;
            case OutputFormat::TIFF_8BIT:
                switch(channel_count){
                    case(ChannelCount::MONO):
                        save_image_8bit = &saveTIFF_8bit<ChannelCount::MONO>; break;
                    case (ChannelCount::RGB):
                        save_image_8bit = &saveTIFF_8bit<ChannelCount::RGB>; break;
                }
                break;
            case OutputFormat::TIFF_16BIT:
                switch(channel_count){
                    case(ChannelCount::MONO):
                        save_image_16bit = &saveTIFF_16bit<ChannelCount::MONO>; break;
                    case (ChannelCount::RGB):
                        save_image_16bit = &saveTIFF_16bit<ChannelCount::RGB>; break;
                }
                break;
            case OutputFormat::BMP_24BIT:
                save_image_8bit = &saveBMP_24bit;
                break;
            case OutputFormat::BMP_8BIT:
                save_image_8bit = &saveBMP_8bit;
                break;
            //case OutputFormat::NP_BUFFER:
                //save_image_? = &saveNPBuffer;
                //break;
            default:
                save_image_8bit = &saveBMP_8bit;
                break;
        }
    }

    // Overload for 8-bit pixel buffers
    void save_image(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {

        if (save_image_8bit == nullptr) {
            throw std::runtime_error(
                "No 8-bit output writer has been configured for the selected output format.");
        }

        save_image_8bit(pixel_buffer, image_height, image_width, output_filepath);
    }


    // Overload for 16-bit pixel buffers
    void save_image(const std::vector<uint16_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {

        if (save_image_16bit == nullptr) {
            throw std::runtime_error(
                "No 16-bit output writer has been configured for the selected output format.");
        }
        save_image_16bit(pixel_buffer, image_height, image_width, output_filepath);
    }

}