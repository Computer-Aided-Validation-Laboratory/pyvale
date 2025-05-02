// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD Library Header Files
#include <string>

namespace dic {

    const char* get_cpu_comp()   { return CPUCOMP; }
    const char* get_git_info()   { return GITINFO; }
    const char* get_git_dirty()  { return GITDIRTY; }
    const char* get_hostname()   { return HOSTNAME; }
    const char* get_build_time() { return BUILDTIME; }

}