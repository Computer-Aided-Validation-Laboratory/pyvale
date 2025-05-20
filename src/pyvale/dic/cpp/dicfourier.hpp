// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICFOURIER_H
#define DICFOURIER_H

// STD library Header files
#include <fftw3.h>

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"

namespace fourier {

    void init(bool *img_roi, util::Config conf, int *windows, int n_windows);

    void mgwd(double *img_def, double *img_ref, 
              int *windows, int n_windows, 
              util::Config conf);

    void cleanup();
}

#endif // DICFOURIER_H
