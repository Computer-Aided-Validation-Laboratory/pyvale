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

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<CamIntrinsics>(m, "CamIntrinsics")
        .def(py::init<>())
        .def_readwrite("fx", &CamIntrinsics::fx)
        .def_readwrite("fy", &CamIntrinsics::fy)
        .def_readwrite("fs", &CamIntrinsics::fs)
        .def_readwrite("cx", &CamIntrinsics::cx)
        .def_readwrite("cy", &CamIntrinsics::cy)
        .def_readwrite("distortion", &CamIntrinsics::distortion);


    py::class_<Calib>(m, "Calib")
        .def(py::init<>())
        .def_readwrite("cam0", &Calib::cam0)
        .def_readwrite("cam1", &Calib::cam1)
        .def_readwrite("rotation", &Calib::rotation)
        .def_readwrite("translation", &Calib::translation);

    m.def("calibrate_stereo", &calibrate_stereo, "calibrate_stereo");
}

