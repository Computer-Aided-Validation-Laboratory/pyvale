// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// raytracer header files
#include "rtmain.h"


namespace py = pybind11;

// Bind the engine function
PYBIND11_MODULE(rtmaincpp, a) {
	py::add_ostream_redirect(a, "ostream_redirect");
    a.def("cpp_render_scene", &render_scene, "Render scene using ray tracing.");
}