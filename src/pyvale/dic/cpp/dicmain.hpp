// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#ifndef DICMAIN_H
#define DICMAIN_H


// STD Library header files
#include <iostream>
#include <functional>
#include <memory>
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
 * @brief Runs DIC using images loaded from the configured file paths.
 *
 * @param img_roi_arr Boolean ROI mask.
 * @param calib Stereo calibration data.
 * @param conf DIC and image configuration.
 * @param multiwindowconf Multiwindow pyramid configuration.
 * @param saveconf Output configuration.
 */
void engine(const py::array_t<bool>& img_roi_arr,
            const Calib &calib,
            const util::Config& conf,
            const MultiwindowConfig &multiwindowconf,
            const common_util::SaveConfig& saveconf);

/**
 * @brief Runs DIC using images supplied in memory.
 *
 * @param images Reference and deformed images.
 * @param img_roi_arr Boolean ROI mask.
 * @param calib Stereo calibration data.
 * @param conf DIC and image configuration.
 * @param multiwindowconf Multiwindow pyramid configuration.
 * @param saveconf Output configuration.
 */
void engine_images(const std::vector<Image>& images,
                   const py::array_t<bool>& img_roi_arr,
                   const Calib &calib,
                   const util::Config& conf,
                   const MultiwindowConfig &multiwindowconf,
                   const common_util::SaveConfig& saveconf);

/**
 * @brief Shared implementation used by both public engine entry points.
 *
 * @param img_roi_arr Boolean ROI mask.
 * @param calib Stereo calibration data.
 * @param conf DIC and image configuration.
 * @param mwconf Multiwindow pyramid configuration.
 * @param saveconf Output configuration.
 * @param interp_factory Creates an interpolator for a requested image index.
 */
void engine_impl(const py::array_t<bool>& img_roi_arr,
                 const Calib &calib,
                 const util::Config &conf,
                 const MultiwindowConfig &mwconf,
                 const common_util::SaveConfig &saveconf,
                 const std::function<std::unique_ptr<Interpolator>(int)> &interp_factory);



#endif // DICMAIN_H

