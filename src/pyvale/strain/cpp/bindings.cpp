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
#include <memory>

namespace py = pybind11;

namespace {
template <typename T>
py::array_t<T> result_view(const std::vector<T> &data,
                           std::size_t frames,
                           std::size_t points,
                           const py::object &owner,
                           std::size_t components = 1) {
    if (components == 1) {
        return py::array_t<T>(
            {static_cast<py::ssize_t>(frames), static_cast<py::ssize_t>(points)},
            {static_cast<py::ssize_t>(points * sizeof(T)), static_cast<py::ssize_t>(sizeof(T))},
            data.data(), owner);
    }
    return py::array_t<T>(
        {static_cast<py::ssize_t>(frames), static_cast<py::ssize_t>(points), static_cast<py::ssize_t>(components)},
        {static_cast<py::ssize_t>(points * components * sizeof(T)),
         static_cast<py::ssize_t>(components * sizeof(T)),
         static_cast<py::ssize_t>(sizeof(T))},
        data.data(), owner);
}

template <typename T>
py::array_t<T> point_view(const std::vector<T> &data, const py::object &owner) {
    return py::array_t<T>({static_cast<py::ssize_t>(data.size())},
                           {static_cast<py::ssize_t>(sizeof(T))},
                           data.data(), owner);
}
}

PYBIND11_MODULE(strain_cpp, m) {

    py::add_ostream_redirect(m, "ostream_redirect");

    py::class_<strain::InMemoryResults, std::shared_ptr<strain::InMemoryResults>>(
        m, "InMemoryResults")
        .def_property_readonly("num_frames", [](const strain::InMemoryResults &r) { return r.nframes; })
        .def_property_readonly("num_windows", [](const strain::InMemoryResults &r) { return r.nwindows; })
        .def_property_readonly("num_ss_x", [](const strain::InMemoryResults &r) { return r.nss_x; })
        .def_property_readonly("num_ss_y", [](const strain::InMemoryResults &r) { return r.nss_y; })
        .def("x", [](const std::shared_ptr<strain::InMemoryResults> &r) { return point_view(r->x, py::cast(r)); })
        .def("y", [](const std::shared_ptr<strain::InMemoryResults> &r) { return point_view(r->y, py::cast(r)); })
        .def("x_mm", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->x_mm, r->nframes, r->nwindows, py::cast(r)); })
        .def("y_mm", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->y_mm, r->nframes, r->nwindows, py::cast(r)); })
        .def("z_mm", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->z_mm, r->nframes, r->nwindows, py::cast(r)); })
        .def("F", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->F, r->nframes, r->nwindows, py::cast(r), 6); })
        .def("strain", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->strain, r->nframes, r->nwindows, py::cast(r), 4); })
        .def("principal_strain", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->principal_strain, r->nframes, r->nwindows, py::cast(r), 2); })
        .def("valid_window", [](const std::shared_ptr<strain::InMemoryResults> &r) { return result_view(r->valid_window, r->nframes, r->nwindows, py::cast(r)); });
    
    // Bind the engine functions
    m.def("strain_engine", &strain::engine, "Run 2D strain analysis on input images with config");
    m.def("strain_engine_2d", &strain::engine_2d, "Run 2D strain analysis using subset-grid coordinates");
    m.def("strain_engine_3d", &strain::engine_3d, "Run 3D surface strain analysis using physical stereo coordinates");
    m.def("strain_engine_2d_results", &strain::engine_2d_results, "Run 2D strain analysis and retain results in memory");
    m.def("strain_engine_3d_results", &strain::engine_3d_results, "Run 3D strain analysis and retain results in memory");
    m.def("strain_engine_2d_column", &strain::engine_2d_column, "Run 2D strain analysis from compact column data");
    m.def("strain_engine_3d_column", &strain::engine_3d_column, "Run 3D strain analysis from compact column data");
    m.def("strain_engine_2d_column_results", &strain::engine_2d_column_results, "Run 2D strain analysis from compact data and retain results");
    m.def("strain_engine_3d_column_results", &strain::engine_3d_column_results, "Run 3D strain analysis from compact data and retain results");
}

