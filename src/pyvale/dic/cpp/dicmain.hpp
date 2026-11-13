// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#ifndef DICMAIN_H
#define DICMAIN_H


// STD Library header files
#include <iostream>
#include <vector>

// Pybind11 Header Files
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// commoncpp header files
#include "../../commoncpp/util.hpp"

// calib header files
#include "../../calib/cpp/calibstereo.hpp"

// DIC Header Files
#include "./dicutil.hpp"
#include "./dicmultiwindow_util.hpp"
#include "./dicresults.hpp"

namespace py = pybind11;

/**
 * @brief Run DIC on the image files selected by the configuration.
 *
 * @param img_roi_arr C-contiguous NumPy bool array of shape (height, width).
 * @param calib Camera calibration used for stereo reconstruction.
 * @param conf Correlation settings, image paths, dimensions and seed positions.
 * @param multiwindowconf FFT window sizes and search settings.
 * @param saveconf Output paths, filenames and file-format settings.
 *
 * @details The ROI buffer is borrowed for the synchronous call and is not
 * retained. The engine owns the image buffers it loads and its working arrays.
 * Subset coordinates and displacement in the 2D correlation are in pixels;
 * calibrated stereo coordinates use the calibration's length unit.
 * Results are written to files according to saveconf; no output array is
 * returned through this function.
 */
void engine(const py::array_t<bool>& img_roi_arr,
            const Calib &calib,
            const util::Config& conf,
            const MultiwindowConfig &multiwindowconf,
            const common_util::SaveConfig& saveconf);

/**
 * @brief Run DIC on image buffers already loaded into native Image objects.
 *
 * @details Each Image owns its row-major pixel buffer. Supported pixel types
 * are uint8, uint16, uint32 and float32. This function borrows the Image vector
 * and the C-contiguous bool ROI for the duration of the call; no caller buffer
 * pointer is retained. The Python binding copies supplied image arrays into
 * owned Image storage before calling this entry point. Configuration, units
 * and file output follow engine().
 */
void engine_images(const std::vector<Image>& images,
                   const py::array_t<bool>& img_roi_arr,
                   const Calib &calib,
                   const util::Config& conf,
                   const MultiwindowConfig &multiwindowconf,
                   const common_util::SaveConfig& saveconf);

void build_info();



bool should_update_ref(const int img_num_def_l, const ResultArrays& results, const util::Config& conf);



#endif // DICMAIN_H

