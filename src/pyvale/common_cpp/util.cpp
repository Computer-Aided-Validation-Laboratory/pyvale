// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <omp.h>

// common_cpp header files

namespace common_util {

    void set_num_threads(int n){
        omp_set_num_threads(n);
    }

}
