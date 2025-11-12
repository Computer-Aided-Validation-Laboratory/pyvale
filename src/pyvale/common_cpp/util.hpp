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
#include <iostream>
#include <iomanip>

// common_cpp header files
#include "./defines.hpp"


struct SaveConfig {
    std::string basepath;
    std::string prefix;
    std::string delimiter;
    bool binary;
    bool at_end;
    bool output_unconverged;
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


#endif //UTIL_H
