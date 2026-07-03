// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef UTIL_H
#define UTIL_H

// STD library Header files
#include <filesystem>
#include <string>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <cmath>
#include <vector>

// common_cpp header files
#include "./defines.hpp"

enum class PixelType { UINT8, UINT16, UINT32};

struct Image {
    std::string filename;
    PixelType type;
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t>  data8;
    std::vector<uint16_t> data16;
    std::vector<uint32_t> data32;
};

namespace common_util {

    struct SaveConfig {
        std::string basepath;
        std::string prefix;
        std::string delimiter;
        bool binary;
        bool output_below_threshold;
        bool shape_params;
    };

    std::string current_datetime_ms();


    inline std::string format_duration_ms(const std::chrono::milliseconds& duration)
    {
        const auto total_seconds = duration.count() / 1000;
        const auto milliseconds = duration.count() % 1000;

        const int hours = static_cast<int>(total_seconds / 3600);
        const int minutes = static_cast<int>((total_seconds % 3600) / 60);
        const int seconds = static_cast<int>(total_seconds % 60);

        std::ostringstream oss;
        oss << std::setfill('0');

        if (hours > 0)
        {
            oss << std::setw(2) << hours << ":";
        }

        oss << std::setw(2) << minutes << ":"
            << std::setw(2) << seconds << "."
            << std::setw(3) << milliseconds;

        return oss.str();
    }


    template<typename A, typename B>
    void info_out(const A& a, const B& b)
    {
        constexpr std::size_t total_width = 80;

        std::ostringstream lhs_stream;
        lhs_stream << "[" << current_datetime_ms() << "] " << a;

        std::ostringstream rhs_stream;
        rhs_stream << b;

        const std::string lhs = lhs_stream.str();
        const std::string rhs = rhs_stream.str();

        std::size_t padding = 1;

        if (lhs.size() + rhs.size() < total_width)
        {
            padding = total_width - lhs.size() - rhs.size();
        }

        std::cout << lhs
                << std::string(padding, ' ')
                << rhs
                << '\n';
    }


    class Timer
    {
    public:
        Timer(const std::string& label, int level = 1)
            : label_(label),
            enabled_(g_debug_level >= level),
            start_(std::chrono::high_resolution_clock::now())
        {}

        ~Timer()
        {
            if (!enabled_)
                return;

            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start_);

            info_out("Time " + label_, format_duration_ms(elapsed));
        }

    private:
        std::string label_;
        bool enabled_;
        std::chrono::high_resolution_clock::time_point start_;
    };

    inline void title(const std::string& text)
    {
        constexpr std::size_t width = 80;

        std::cout << std::string(width, '-') << '\n';

        const std::size_t content_width = text.size() + 2; // spaces around title

        if (content_width >= width)
        {
            std::cout << text << '\n';
        }
        else
        {
            const std::size_t total_pad = width - content_width;
            const std::size_t left_pad = total_pad / 2;
            const std::size_t right_pad = total_pad - left_pad;

            std::cout << std::string(left_pad, '-')
                    << ' '
                    << text
                    << ' '
                    << std::string(right_pad, '-')
                    << '\n';
        }

        std::cout << std::string(width, '-') << '\n';
    }

    inline void write_int(std::ofstream& out, int val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(int));
    }

    inline void write_uint8t(std::ofstream& out, int val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(uint8_t));
    }

    inline void write_dbl(std::ofstream& out, double val) {
        out.write(reinterpret_cast<const char*>(&val), sizeof(double));
    }

    /**
    * @brief Sets the number of threads to be used by the DIC engine.
    *
    * @param n The number of threads to set for the DIC engine.
    */
    void set_num_threads(int n);
}

#endif //UTIL_H
