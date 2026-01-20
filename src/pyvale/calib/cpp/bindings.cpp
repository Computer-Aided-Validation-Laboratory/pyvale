// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// Strain Header files
#include "./calibstereo.hpp"

namespace py = pybind11;

PYBIND11_MODULE(calibcpp, m) {

    m.def("stereo_calibration", &stereo_calibration, "stereo_calibration");
}

