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
    
    // Bind the engine function
    m.def("strain_engine", &strain::engine, "Run DIC analysis on input images with config");
}

