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
#include "../../common_cpp/util.hpp"

// DIC Header files
#include "./dicutil.hpp"
#include "./dicmain.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dic2dcpp, m) {

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<util::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("ss_step", &util::Config::ss_step)
        .def_readwrite("ss_size", &util::Config::ss_size)
        .def_readwrite("max_iter", &util::Config::max_iter)
        .def_readwrite("precision", &util::Config::precision)
        .def_readwrite("threshold", &util::Config::threshold)
        .def_readwrite("bf_threshold", &util::Config::bf_threshold)
        .def_readwrite("max_disp", &util::Config::max_disp)
        .def_readwrite("corr_crit", &util::Config::corr_crit)
        .def_readwrite("shape_func", &util::Config::shape_func)
        .def_readwrite("interp_routine", &util::Config::interp_routine)
        .def_readwrite("scan_method", &util::Config::scan_method)
        .def_readwrite("px_hori", &util::Config::px_hori)
        .def_readwrite("px_vert", &util::Config::px_vert)
        .def_readwrite("num_def_img", &util::Config::num_def_img)
        .def_readwrite("rg_seed", &util::Config::rg_seed)
        .def_readwrite("num_params", &util::Config::num_params)
        .def_readwrite("fft_mad", &util::Config::fft_mad)
        .def_readwrite("fft_mad_scale", &util::Config::fft_mad_scale)
        .def_readwrite("filenames", &util::Config::filenames)
        .def_readwrite("debug_level", &util::Config::debug_level);
    
    // Bind the engine function
    m.def("dic_engine", &DICengine, "Run DIC analysis on input images with config");
}

