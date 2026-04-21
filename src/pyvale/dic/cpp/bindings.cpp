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
#include "./dicutil.hpp"
#include "./dicmain.hpp"
#include "./dicinterp.hpp"
#include "./dicinterpBspline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(diccpp, m) {

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
        .def_readwrite("basenames", &util::Config::basenames)
        .def_readwrite("fullpaths", &util::Config::fullpaths)
        .def_readwrite("debug_level", &util::Config::debug_level)
        .def_readwrite("stereo", &util::Config::stereo);

    // Bind the engine function
    m.def("engine", &engine, "Run DIC analysis on input images with config");

    // interpolator bindings
    py::class_<InterpVals>(m, "InterpVals")
        .def_readonly("f", &InterpVals::f)
        .def_readonly("dfdx", &InterpVals::dfdx)
        .def_readonly("dfdy", &InterpVals::dfdy);

    // ABC
    py::class_<Interpolator>(m, "Interpolator")
        .def("eval", &Interpolator::eval)
        .def("eval_dx", &Interpolator::eval_dx)
        .def("eval_dy", &Interpolator::eval_dy)
        .def("eval_and_derivs", &Interpolator::eval_and_derivs);

    // 2d b-spline interpolator
    py::class_<Bspline, Interpolator>(m, "Bspline")
    .def(py::init([](py::array arr) {

        if (arr.ndim() != 2)
            throw std::runtime_error("img must be a 2D numpy array");

        int px_vert = arr.shape(0);
        int px_hori = arr.shape(1);

        Image img;
        img.width  = px_hori;
        img.height = px_vert;

        // CHecking the type of the python array
        if (py::isinstance<py::array_t<uint8_t>>(arr)) {

            img.type = PixelType::UINT8;
            auto buf = arr.cast<py::array_t<uint8_t, py::array::c_style>>();

            img.data8.assign(buf.data(), buf.data() + px_hori * px_vert);

        } else if (py::isinstance<py::array_t<uint16_t>>(arr)) {

            img.type = PixelType::UINT16;
            auto buf = arr.cast<py::array_t<uint16_t, py::array::c_style>>();

            img.data16.assign(buf.data(), buf.data() + px_hori * px_vert);

        } else if (py::isinstance<py::array_t<uint32_t>>(arr)) {

            img.type = PixelType::UINT32;
            auto buf = arr.cast<py::array_t<uint32_t, py::array::c_style>>();

            img.data32.assign(buf.data(), buf.data() + px_hori * px_vert);

        } else {
            throw std::runtime_error("Unsupported numpy dtype (expected uint8/uint16/uint32)");
        }

        return std::make_unique<Bspline>(img);
    }))
    .def("eval", &Bspline::eval)
    .def("eval_dx", &Bspline::eval_dx)
    .def("eval_dy", &Bspline::eval_dy)
    .def("eval_and_derivs", &Bspline::eval_and_derivs);

}

