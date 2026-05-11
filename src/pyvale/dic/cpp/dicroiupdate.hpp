// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICROIUPDATE_H
#define DICROIUPDATE_H

#include "dicresults.hpp"
#include "dicutil.hpp"
#include <opencv2/opencv.hpp>
#include <cmath>




bool* propagate_roi(
    const bool* img_roi,
    const ResultArrays results_def,
    const util::Config conf,
    const subset::Grid ss_grid);

#endif // DICROIUPDATE_H
