// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICBUILDINFO_H
#define DICBUILDINFO_H

// STD Library Header Files
#include <string>

namespace dic {

    const char* get_cpu_comp();
    const char* get_git_info();
    const char* get_git_dirty();
    const char* get_hostname();
    const char* get_build_time();

}

#endif //DICBUILDINFO_H