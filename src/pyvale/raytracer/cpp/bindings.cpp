// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>

// raytracer header files
#include "rtmain.h"

namespace nb = nanobind;

NB_MODULE(rtmaincpp, a) {
    a.def("cpp_render_scene", &render_scene, "Render scene using ray tracing.");
}