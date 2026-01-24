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

// common_cpp Header Files
#include "./util.hpp"

namespace py = pybind11;

PYBIND11_MODULE(common_cpp, m) {

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<common_util::SaveConfig>(m, "SaveConfig")
        .def(py::init<>())
        .def_readwrite("basepath", &common_util::SaveConfig::basepath)
        .def_readwrite("binary", &common_util::SaveConfig::binary)
        .def_readwrite("prefix", &common_util::SaveConfig::prefix)
        .def_readwrite("delimiter", &common_util::SaveConfig::delimiter)
        .def_readwrite("at_end", &common_util::SaveConfig::at_end)
        .def_readwrite("output_below_threshold", &common_util::SaveConfig::output_below_threshold)
        .def_readwrite("shape_params", &common_util::SaveConfig::shape_params);

    m.def("set_num_threads", &common_util::set_num_threads, "Set number of OMP threads");
}
