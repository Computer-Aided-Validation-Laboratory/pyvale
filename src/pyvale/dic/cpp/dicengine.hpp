// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICENGINE_H
#define DICENGINE_H

#include "./dicmultiwindow_util.hpp"
#include "./dicresults.hpp"

/**
 * @brief Builds the initial subset grid for the selected scan method.
 */
SubsetGrid build_initial_grid(const bool *img_roi,
                              const util::Config &conf,
                              const MultiwindowConfig &mwconf,
                              const common_util::SaveConfig &saveconf,
                              MultiwindowPyramid &pyramid);


bool should_update_ref(const int img_num_def_l,
                       const ResultArrays &results,
                       const util::Config &conf);

#endif // DICENGINE_H
