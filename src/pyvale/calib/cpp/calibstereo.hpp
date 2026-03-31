// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef CALIBSTEREO_H
#define CALIBSTEREO_H


// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>

// master func that gets called from python
void stereo_calibration(const std::vector<double> &init_params,
                        const std::vector<double> &dots_cam0,
                        const std::vector<double> &dots_cam1,
                        const std::vector<double> &grid,
                        const std::vector<int> &lengths,
                        const int px_hori, const int px_vert, const int num_img_pairs);


#endif
