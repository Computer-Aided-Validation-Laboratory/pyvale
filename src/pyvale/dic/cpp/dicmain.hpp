// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
#ifndef DICMAIN_H
#define DICMAIN_H

// Pybind11 Header Files
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// common_cpp header files
#include "../../common_cpp/util.hpp"

// DIC Header Files
#include "./dicutil.hpp"

namespace py = pybind11;

/**
 * @brief Runs the 2D Digital Image Correlation (DIC) engine on input image data.
 *
 * This function performs 2D DIC using a specified correlation criterion,
 * shape function, interpolation scheme, and scan method.
 * It computes displacement fields by comparing a reference image with a stack of
 * deformed images over a defined region of interest (ROI).
 *
 * @param img_ref_arr A 2D NumPy array (pybind11) representing the reference image.
 * @param img_def_stack_arr A 3D NumPy array containing a stack of deformed images.
 *                          Each image must have the same dimensions as the reference image.
 * @param img_roi_arr A 2D boolean NumPy array specifying the region of interest (ROI).
 *                    True values indicate active ROI pixels.
 * @param conf Configuration parameters for the DIC engine (subset size, correlation method, etc.).
 *             This is an instance of `util::Config`.
 * @param saveconf Output configuration for saving DIC results to disk.
 *                 This is an instance of `SaveConfig`.
 *
 * @details
 * The function supports multiple scan methods including raster scan, brute force,
 * reliability-guided, and Fourier-based approaches. It computes correlation
 * between the reference and deformed images over subsets defined by the ROI.
 * 
 * Subset data is initialized and processed in parallel using OpenMP.
 * Results are optionally saved after each image or at the end of processing,
 * depending on `saveconf.at_end`.
 *
 * @note This function is intended to be called via the Python interface using pybind11.
 *       Image arrays are expected to be contiguous and C-style (row-major) in memory.
 */
void DICengine(const py::array_t<double>& img_stack_arr,
               const py::array_t<bool>& img_roi_arr,
               util::Config& conf,
               common_util::SaveConfig& saveconf);

void build_info();



#endif // DICMAIN_H

