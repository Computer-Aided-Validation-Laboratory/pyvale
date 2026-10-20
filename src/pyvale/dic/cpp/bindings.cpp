// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// pybind11 header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// Standard Library
#include <vector>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <memory>

// commoncpp Header Files
#include "../../commoncpp/util.hpp"

// DIC Header files
#include "./dicutil.hpp"
#include "./dicmain.hpp"
#include "./dicinterp.hpp"
#include "./dicmultiwindow_util.hpp"
#include "./dicinterpBspline.hpp"

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> result_view(const std::vector<T> &data,
                           const std::size_t frames,
                           const std::size_t points,
                           const py::object &owner) {
    return py::array_t<T>(
        {static_cast<ssize_t>(frames), static_cast<ssize_t>(points)},
        {static_cast<ssize_t>(points * sizeof(T)), static_cast<ssize_t>(sizeof(T))},
        data.data(), owner);
}

template <typename T>
py::array_t<T> point_view(const std::vector<T> &data, const py::object &owner) {
    return py::array_t<T>(
        {static_cast<ssize_t>(data.size())},
        {static_cast<ssize_t>(sizeof(T))}, data.data(), owner);
}

template <typename T>
bool copy_array_to_image(py::handle h, Image &img, PixelType type) {
    using Array = py::array_t<T, py::array::c_style>;
    auto arr = Array::ensure(h);
    if (!arr || arr.ndim() != 2) {
        return false;
    }

    img.width = static_cast<uint32_t>(arr.shape(1));
    img.height = static_cast<uint32_t>(arr.shape(0));
    img.type = type;

    const size_t num_pixels = static_cast<size_t>(img.width) * img.height;
    if constexpr (std::is_same_v<T, uint8_t>) {
        img.data8.assign(arr.data(), arr.data() + num_pixels);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        img.data16.assign(arr.data(), arr.data() + num_pixels);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        img.data32.assign(arr.data(), arr.data() + num_pixels);
    } else if constexpr (std::is_same_v<T, float>) {
        img.data32f.assign(arr.data(), arr.data() + num_pixels);
    } else if constexpr (std::is_same_v<T, double>) {
        img.data64f.assign(arr.data(), arr.data() + num_pixels);
    }
    return true;
}

Image image_from_array(py::handle h, const std::string &filename) {
    Image img;
    img.filename = filename;

    if (copy_array_to_image<uint8_t>(h, img, PixelType::UINT8) ||
        copy_array_to_image<uint16_t>(h, img, PixelType::UINT16) ||
        copy_array_to_image<uint32_t>(h, img, PixelType::UINT32) ||
        copy_array_to_image<float>(h, img, PixelType::UINT32F) ||
        copy_array_to_image<double>(h, img, PixelType::FLOAT64)) {
        return img;
    }

    throw std::runtime_error(
        "Unsupported array: expected 2D C-contiguous NumPy array of dtype uint8/uint16/uint32/float32/float64"
    );
}

std::vector<Image> images_from_sequence(py::sequence image_arrays, const util::Config &conf) {
    std::vector<Image> images;
    images.reserve(py::len(image_arrays));

    for (size_t i = 0; i < py::len(image_arrays); ++i) {
        std::string filename = i < conf.basenames.size()
            ? conf.basenames[i]
            : "array_img_" + std::to_string(i);
        images.push_back(image_from_array(image_arrays[i], filename));
    }

    return images;
}

} // namespace

PYBIND11_MODULE(diccpp, m) {

    py::add_ostream_redirect(m, "ostream_redirect");

    py::enum_<util::ECorrCrit>(m, "CorrCrit")
        .value("SSD", util::ECorrCrit::SSD)
        .value("NSSD", util::ECorrCrit::NSSD)
        .value("ZNSSD", util::ECorrCrit::ZNSSD);

    py::enum_<util::EOptimizer>(m, "Optimizer")
        .value("ICGN", util::EOptimizer::ICGN)
        .value("FALM", util::EOptimizer::FALM);

    py::enum_<util::EShapeFunc>(m, "ShapeFunc")
        .value("RIGID", util::EShapeFunc::RIGID)
        .value("AFFINE", util::EShapeFunc::AFFINE)
        .value("QUAD", util::EShapeFunc::QUAD);

    py::enum_<util::EInterpRoutine>(m, "InterpRoutine")
        .value("BSPLINE", util::EInterpRoutine::BSPLINE)
        .value("HERMITE", util::EInterpRoutine::HERMITE);

    py::enum_<util::EScanMethod>(m, "ScanMethod")
        .value("MULTIWINDOW_RG", util::EScanMethod::MULTIWINDOW_RG)
        .value("SINGLEWINDOW_RG", util::EScanMethod::SINGLEWINDOW_RG)
        .value("MULTIWINDOW", util::EScanMethod::MULTIWINDOW)
        .value("RASTER", util::EScanMethod::RASTER);

    py::enum_<util::EIncrementalCond>(m, "IncrementalCond")
        .value("IMAGE", util::EIncrementalCond::IMAGE)
        .value("ITER", util::EIncrementalCond::ITER)
        .value("COST", util::EIncrementalCond::COST);

    py::enum_<util::EFFTPrecision>(m, "FFTPrecision")
        .value("FLOAT32", util::EFFTPrecision::FLOAT32)
        .value("FLOAT64", util::EFFTPrecision::FLOAT64);

    py::class_<util::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("ss_step", &util::Config::ss_step)
        .def_readwrite("ss_size", &util::Config::ss_size)
        .def_readwrite("partial_subset", &util::Config::partial_subset)
        .def_readwrite("partial_subset_multiwindow", &util::Config::partial_subset_multiwindow)
        .def_readwrite("max_iter", &util::Config::max_iter)
        .def_readwrite("precision", &util::Config::precision)
        .def_readwrite("threshold", &util::Config::threshold)
        .def_readwrite("max_disp", &util::Config::max_disp)
        .def_readwrite("epi_search_distance", &util::Config::epi_search_distance)
        .def_readwrite("epi_distance_threshold", &util::Config::epi_distance_threshold)
        .def_readwrite("corr_crit", &util::Config::corr_crit)
        .def_readwrite("optimizer", &util::Config::optimizer)
        .def_readwrite("shape_func", &util::Config::shape_func)
        .def_readwrite("interp_routine", &util::Config::interp_routine)
        .def_readwrite("scan_method", &util::Config::scan_method)
        .def_readwrite("px_hori", &util::Config::px_hori)
        .def_readwrite("px_vert", &util::Config::px_vert)
        .def_readwrite("num_def_img", &util::Config::num_def_img)
        .def_readwrite("rg_seeds", &util::Config::rg_seeds)
        .def_readwrite("num_params", &util::Config::num_params)
        .def_readwrite("fft_precision", &util::Config::fft_precision)
        .def_readwrite("basenames", &util::Config::basenames)
        .def_readwrite("fullpaths", &util::Config::fullpaths)
        .def_readwrite("debug_level", &util::Config::debug_level)
        .def_readwrite("stereo", &util::Config::stereo)
        .def_readwrite("incremental", &util::Config::incremental)
        .def_readwrite("incremental_update_cond", &util::Config::incremental_update_cond)
        .def_readwrite("image_filter_kernel", &util::Config::image_filter_kernel)
        .def_readwrite("image_filter_sigma", &util::Config::image_filter_sigma)
        .def_readwrite("incremental_update_val",  &util::Config::incremental_update_val);

    py::class_<MultiwindowConfig>(m, "MultiwindowConfig")
        .def(py::init<>())
        .def_readwrite("overlap", &MultiwindowConfig::overlap)
        .def_readwrite("subset_size", &MultiwindowConfig::subset_size)
        .def_readwrite("search_area", &MultiwindowConfig::search_area)
        .def_readwrite("fft_filter", &MultiwindowConfig::fft_filter)
        .def_readwrite("fft_filter_threshold", &MultiwindowConfig::fft_filter_threshold)
        .def_readwrite("fft_filter_radius", &MultiwindowConfig::fft_filter_radius)
        .def_readwrite("fft_filter_corr_power", &MultiwindowConfig::fft_filter_corr_power)
        .def_readwrite("fft_save", &MultiwindowConfig::fft_save);

    py::class_<InMemoryResults, std::shared_ptr<InMemoryResults>>(m, "InMemoryResults")
        .def_property_readonly("num_frames", [](const InMemoryResults &r) { return r.num_frames; })
        .def_property_readonly("num_points", [](const InMemoryResults &r) { return r.num_ss; })
        .def_property_readonly("num_ss_x", [](const InMemoryResults &r) { return r.num_ss_x; })
        .def_property_readonly("num_ss_y", [](const InMemoryResults &r) { return r.num_ss_y; })
        .def_property_readonly("stereo", [](const InMemoryResults &r) { return r.stereo; })
        .def("ss_x", [](const std::shared_ptr<InMemoryResults> &r) { return point_view(r->ss_x, py::cast(r)); })
        .def("ss_y", [](const std::shared_ptr<InMemoryResults> &r) { return point_view(r->ss_y, py::cast(r)); })
        .def("u", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->u, r->num_frames, r->num_ss, py::cast(r)); })
        .def("v", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->v, r->num_frames, r->num_ss, py::cast(r)); })
        .def("mag", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->mag, r->num_frames, r->num_ss, py::cast(r)); })
        .def("converged", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->converged, r->num_frames, r->num_ss, py::cast(r)); })
        .def("cost", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->cost, r->num_frames, r->num_ss, py::cast(r)); })
        .def("ftol", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->ftol, r->num_frames, r->num_ss, py::cast(r)); })
        .def("xtol", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->xtol, r->num_frames, r->num_ss, py::cast(r)); })
        .def("niter", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->niter, r->num_frames, r->num_ss, py::cast(r)); })
        .def("disparity_u", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->disparity_u, r->num_frames, r->num_ss, py::cast(r)); })
        .def("disparity_v", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->disparity_v, r->num_frames, r->num_ss, py::cast(r)); })
        .def("disparity_mag", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->disparity_mag, r->num_frames, r->num_ss, py::cast(r)); })
        .def("x_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->x_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("y_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->y_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("z_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->z_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("u_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->u_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("v_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->v_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("w_mm", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->w_mm, r->num_frames, r->num_ss, py::cast(r)); })
        .def("stereo_converged", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->stereo_converged, r->num_frames, r->num_ss, py::cast(r)); })
        .def("stereo_cost", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->stereo_cost, r->num_frames, r->num_ss, py::cast(r)); })
        .def("stereo_ftol", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->stereo_ftol, r->num_frames, r->num_ss, py::cast(r)); })
        .def("stereo_xtol", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->stereo_xtol, r->num_frames, r->num_ss, py::cast(r)); })
        .def("stereo_niter", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->stereo_niter, r->num_frames, r->num_ss, py::cast(r)); })
        .def("epi_dist_px", [](const std::shared_ptr<InMemoryResults> &r) { return result_view(r->epi_dist_px, r->num_frames, r->num_ss, py::cast(r)); });

    // Bind the engine functions
    m.def("engine", &engine, "Run DIC analysis on image files with config");
    m.def("engine_results", &engine_results, "Run DIC analysis and retain results in memory");
    m.def("engine_images_results", [](py::sequence image_arrays,
                                       const py::array_t<bool>& img_roi_arr,
                                       const Calib &calib,
                                       const util::Config& conf,
                                       const MultiwindowConfig &multiwindowconf,
                                       const common_util::SaveConfig& saveconf) {
        return engine_images_results(images_from_sequence(image_arrays, conf),
                                     img_roi_arr, calib, conf,
                                     multiwindowconf, saveconf);
    }, "Run in-memory-image DIC and retain results in memory");
    m.def("engine_images", [](py::sequence image_arrays,
                              const py::array_t<bool>& img_roi_arr,
                              const Calib &calib,
                              const util::Config& conf,
                              const MultiwindowConfig &multiwindowconf,
                              const common_util::SaveConfig& saveconf) {
        engine_images(images_from_sequence(image_arrays, conf), img_roi_arr, calib, conf, multiwindowconf, saveconf);
    }, "Run DIC analysis on in-memory image arrays with config");

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
    using U8Array  = py::array_t<uint8_t,  py::array::c_style>;
    using U16Array = py::array_t<uint16_t, py::array::c_style>;
    using U32Array = py::array_t<uint32_t, py::array::c_style>;
    using F32Array = py::array_t<float, py::array::c_style>;
    using F64Array = py::array_t<double, py::array::c_style>;

    py::class_<Bspline, Interpolator>(m, "Bspline")
        .def(py::init([](py::handle h) {
            Image img;

            if (auto arr = U8Array::ensure(h)) {
                if (arr.ndim() != 2)
                    throw std::runtime_error("Unsupported array: expected 2D array");

                img.width = arr.shape(1);
                img.height = arr.shape(0);
                img.type = PixelType::UINT8;
                img.data8.assign(arr.data(), arr.data() + img.width * img.height);
            } else if (auto arr = U16Array::ensure(h)) {
                if (arr.ndim() != 2)
                    throw std::runtime_error("Unsupported array: expected 2D array");

                img.width = arr.shape(1);
                img.height = arr.shape(0);
                img.type = PixelType::UINT16;
                img.data16.assign(arr.data(), arr.data() + img.width * img.height);
            } else if (auto arr = U32Array::ensure(h)) {
                if (arr.ndim() != 2)
                    throw std::runtime_error("Unsupported array: expected 2D array");

                img.width = arr.shape(1);
                img.height = arr.shape(0);
                img.type = PixelType::UINT32;
                img.data32.assign(arr.data(), arr.data() + img.width * img.height);
            } else if (auto arr = F32Array::ensure(h)) {
                if (arr.ndim() != 2)
                    throw std::runtime_error("Unsupported array: expected 2D array");

                img.width = arr.shape(1);
                img.height = arr.shape(0);
                img.type = PixelType::UINT32F;
                img.data32f.assign(arr.data(), arr.data() + img.width * img.height);
            } else if (auto arr = F64Array::ensure(h)) {
                if (arr.ndim() != 2)
                    throw std::runtime_error("Unsupported array: expected 2D array");

                img.width = arr.shape(1);
                img.height = arr.shape(0);
                img.type = PixelType::FLOAT64;
                img.data64f.assign(arr.data(), arr.data() + img.width * img.height);
            } else {
                throw std::runtime_error(
                    "Unsupported array: expected 2D C-contiguous NumPy array of dtype uint8/uint16/uint32/float32/float64"
                );
            }

            return Bspline(img);
        }))
        .def("eval", &Bspline::eval)
        .def("eval_dx", &Bspline::eval_dx)
        .def("eval_dy", &Bspline::eval_dy)
        .def("eval_and_derivs", &Bspline::eval_and_derivs);

}
