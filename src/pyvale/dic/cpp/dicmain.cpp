// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>
#include <signal.h>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicbruteforce.hpp"
#include "./dicoptimizer.hpp"
#include "./dicscanmethod.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicstrain.hpp"
#include "./dicfourier.hpp"

// cuda Header files
#include "../cuda/malloc.hpp"

namespace py = pybind11;


void DICengine(const py::array_t<double>& img_ref_arr,
               const py::array_t<double>& img_def_stack_arr,
               const py::array_t<bool>&   img_roi_arr, 
               util::Config &conf,
               util::SaveConfig &saveconf){

    // ------------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------------
    TITLE("CONFIGURATION");
    INFO_OUT("Width of Images: ", conf.px_hori << " [px]");
    INFO_OUT("Height of Images: ", conf.px_vert << " [px]");
    INFO_OUT("Number of Deformed Images: ", conf.num_def_img);
    INFO_OUT("Max number of solver iterations: ", conf.max_iter);
    INFO_OUT("Correlation Criterion: ", conf.corr_crit);
    INFO_OUT("Shape Function: ", conf.shape_func);
    INFO_OUT("Interpolation Routine: ", conf.interp_routine);
    INFO_OUT("Image Scan Method: ", conf.scan_method);
    INFO_OUT("Number of OMP threads:", omp_get_max_threads());

    // Register signal handler for Ctrl+C
    signal(SIGINT, scanmethod::signalHandler);

       // get raw pointers
    bool* img_roi = static_cast<bool*>(img_roi_arr.request().ptr);
    double* img_ref = static_cast<double*>(img_ref_arr.request().ptr);
    double* img_def_stack = static_cast<double*>(img_def_stack_arr.request().ptr);

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<util::SubsetData> ssdata;
    if (conf.scan_method == "FFT"){
        fourier::init(ssdata, img_roi, conf);
    }
    else {
        ssdata.push_back(util::gen_ss_list(img_roi, conf.ss_step[0],
                                           conf.ss_size[0], conf.px_hori, 
                                           conf.px_vert));
    }

    // resize the results based on subset information
    util::resize_results(conf.num_def_img, ssdata.back().num,
                         conf.num_params, saveconf.at_end);

   
    // define our interpolator for the reference image
    Interpolator interp_ref(img_ref, conf.px_hori, conf.px_vert);

    // initialise the LM optimizer with shape func and corr crit
    optimizer::init(conf.corr_crit, conf.shape_func);

    // initialise the brute force scan
    std::string brute_method = "SPIRAL";
    brute::init(conf.corr_crit, brute_method);



    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    util::Timer timer("DIC Engine:");

    for (int img_num = 0; img_num < conf.num_def_img; img_num++){


        // pointer to starting location of deformed image in memory
        int num_px_in_image = conf.px_hori * conf.px_vert;
        double *img_def = img_def_stack + img_num*num_px_in_image;

        // raster scan
        if (conf.scan_method=="IMAGE_SCAN") 
            scanmethod::image(interp_ref, img_def, ssdata, conf, img_num);

        // raster with brute force
        else if (conf.scan_method=="IMAGE_SCAN_WITH_BF") 
            scanmethod::image_with_bf(interp_ref, img_ref, img_def, ssdata, conf, img_num);

        // reliability Guided
        else if (conf.scan_method=="RG")
            scanmethod::reliability_guided(interp_ref, img_ref, img_def, ssdata, conf, img_num);

        // multi window fft
        else if (conf.scan_method=="FFT")
            scanmethod::multi_window_fourier(interp_ref, img_ref, img_def, ssdata, conf, img_num);

        if (!saveconf.at_end)
            util::save_to_disk(img_num, saveconf, ssdata.back(), conf.num_def_img, conf.num_params);

    }

    if (saveconf.at_end)
        for (int img_num = 0; img_num < conf.num_def_img; img_num++)
            util::save_to_disk(img_num, saveconf, ssdata.back(), conf.num_def_img, conf.num_params);
}


void build_info(){
        //std::cout << "Buld Information:" << std::endl;
        //INFO_OUT("- g++ version:", CPUCOMP);
        //INFO_OUT("- Compiler directory:", COMPPATH);
        //INFO_OUT("- Git SHA:", GITINFO);
        //INFO_OUT("- Number of dirty files:", GITDIRTY);
        //INFO_OUT("- Compiled on Machine:", HOSTNAME);
        //INFO_OUT("- Compiled on OS:", OSNAME);
        //INFO_OUT("- Compiled at:", BUILDTIME);
        //std::cout << std::endl;
}



PYBIND11_MODULE(dic2dcpp, m) {
    py::class_<util::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("ss_step", &util::Config::ss_step)
        .def_readwrite("ss_size", &util::Config::ss_size)
        .def_readwrite("max_iter", &util::Config::max_iter)
        .def_readwrite("precision", &util::Config::precision)
        .def_readwrite("threshold_lm", &util::Config::threshold_lm)
        .def_readwrite("threshold_bf", &util::Config::threshold_bf)
        .def_readwrite("range_bf", &util::Config::range_bf)
        .def_readwrite("corr_crit", &util::Config::corr_crit)
        .def_readwrite("shape_func", &util::Config::shape_func)
        .def_readwrite("interp_routine", &util::Config::interp_routine)
        .def_readwrite("scan_method", &util::Config::scan_method)
        .def_readwrite("px_hori", &util::Config::px_hori)
        .def_readwrite("px_vert", &util::Config::px_vert)
        .def_readwrite("num_def_img", &util::Config::num_def_img)
        .def_readwrite("num_params", &util::Config::num_params);

    py::class_<util::SaveConfig>(m, "SaveConfig")
        .def(py::init<>())
        .def_readwrite("basepath", &util::SaveConfig::basepath)
        .def_readwrite("binary", &util::SaveConfig::binary)
        .def_readwrite("prefix", &util::SaveConfig::prefix)
        .def_readwrite("delimiter", &util::SaveConfig::delimiter)
        .def_readwrite("at_end", &util::SaveConfig::at_end);

    // Bind the engine function
    m.def("build_info", &build_info, "build information");
    m.def("dic_engine", &DICengine, "Run 2D analysis on input images with config");
    m.def("strain_engine", &strain::engine, "Strain C++ calculations");
}




