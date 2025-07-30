// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// defines.hpp
#ifndef DEFINES_H
#define DEFINES_H

#define TITLE(a) std::cout << std::string(80, '-') << std::endl; std::cout << std::string(38 - std::strlen(a)/2, '-') << " " << a << " " << std::string(38 - std::strlen(a)/2, '-') << std::endl; std::cout << std::string(80, '-') << std::endl;
#define INFO_OUT(a,b) std::cout << "  - " << std::left << std::setw(50) << a << b << std::endl;
#define DEBUGGER std::cout << __FILE__ << " " << __LINE__ << std::endl;

#ifdef CUDA
        #define CUDA_CALL(x) do { if((x) != cudaSuccess) {\
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);}} while(0)

        #define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
        printf("Error at %s:%d\n",__FILE__,__LINE__);\
        exit(EXIT_FAILURE);}} while(0)
#endif

// Use static function approach instead of inline variable
inline int& get_debug_level() {
    static int level = 0;
    return level;
}

inline int& g_debug_level = get_debug_level();
#define DEBUG_PROGRESS_BAR(level) \
    if (g_debug_level >= level)
#define DEBUG(level) \
    if (g_debug_level >= level)

#endif // DEFINES_HPP


