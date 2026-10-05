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
#include "./strain.hpp"

namespace py = pybind11;

PYBIND11_MODULE(strain_cpp, m) {

    py::add_ostream_redirect(m, "ostream_redirect");
    
    // Bind the engine functions
    m.def("strain_engine", &strain::engine, "Run 2D strain analysis on input images with config");
    m.def("strain_engine_2d", &strain::engine_2d, "Run 2D strain analysis using subset-grid coordinates");
    m.def("strain_engine_3d", &strain::engine_3d, "Run 3D surface strain analysis using physical stereo coordinates");
}

