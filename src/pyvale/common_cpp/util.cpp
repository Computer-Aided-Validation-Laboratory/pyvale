// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>

// common_cpp header files
#include "./util.hpp"

namespace common_util {

    std::string current_datetime_ms() {
        const auto now = std::chrono::system_clock::now();
        const auto now_time = std::chrono::system_clock::to_time_t(now);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::tm local_time{};
        
        #if defined(_WIN32)
            localtime_s(&local_time, &now_time);
        #else
            localtime_r(&now_time, &local_time);
        #endif

        std::ostringstream oss;
        oss << std::put_time(&local_time, "%Y-%m-%d %H:%M:%S")
            << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }

    void set_num_threads(int n){
        omp_set_num_threads(n);
    }

    int get_num_threads(){
        return omp_get_max_threads();
    }

}
