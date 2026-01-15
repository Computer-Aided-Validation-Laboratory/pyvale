// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef UTIL_H
#define UTIL_H

// STD library Header files
#include <string>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>

// common_cpp header files
#include "./defines.hpp"

namespace common_util {

    struct SaveConfig {
        std::string basepath;
        std::string prefix;
        std::string delimiter;
        bool binary;
        bool at_end;
        bool output_below_threshold;
        bool shape_params;
    };

    class Timer {
        public:
            Timer(const std::string& label)
                : label_(label), start_(std::chrono::high_resolution_clock::now()) {}

            ~Timer() {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start_;
                INFO_OUT("Time taken for " + label_, elapsed.count() << " [s]");
            }

        private:
            std::string label_;
            std::chrono::high_resolution_clock::time_point start_;
    };

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
