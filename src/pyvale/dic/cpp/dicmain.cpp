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
    TITLE("Config");
    INFO_OUT("Width of Images: ", conf.px_hori << " [px]");
    INFO_OUT("Height of Images: ", conf.px_vert << " [px]");
    INFO_OUT("Number of Deformed Images: ", conf.num_def_img);
    INFO_OUT("Max number of solver iterations: ", conf.max_iter);
    INFO_OUT("Correlation Criterion: ", conf.corr_crit);
    INFO_OUT("Shape Function: ", conf.shape_func);
    INFO_OUT("Interpolation Routine: ", conf.interp_routine);
    INFO_OUT("FFT MAD outlier removal enabled: ", conf.fft_mad);
    INFO_OUT("FFT MAD scale: ", conf.fft_mad_scale);
    INFO_OUT("Image Scan Method: ", conf.scan_method);
    INFO_OUT("Optimization Precision:", conf.precision);
    INFO_OUT("Optimization Threshold:", conf.opt_threshold);
    INFO_OUT("Estimate for Max Displacement:", conf.max_disp << " [px]");
    INFO_OUT("Subset Size:", conf.ss_size << " [px]");
    INFO_OUT("Subset Step:", conf.ss_step << " [px]" );
    INFO_OUT("Number of OMP threads:", omp_get_max_threads());
    if (conf.scan_method=="RG") INFO_OUT("Reliability Guided Seed central px location: ", "(" 
                                         << conf.rg_seed.first+conf.ss_size/2 << ", " << conf.rg_seed.second+conf.ss_size/2 << ") [px] " )

    // Register signal handler for Ctrl+C
    signal(SIGINT, scanmethod::signalHandler);

       // get raw pointers
    bool* img_roi = static_cast<bool*>(img_roi_arr.request().ptr);
    double* img_ref = static_cast<double*>(img_ref_arr.request().ptr);
    double* img_def_stack = static_cast<double*>(img_def_stack_arr.request().ptr);

    // debugging
    //for (int y = 0; y < conf.px_vert; y++){
    //    for (int x = 0; x < conf.px_hori; x++){
    //        std::cout << x << " " << y << " ";
    //        std::cout << img_ref[y*conf.px_hori + x] << " ";
    //        std::cout << img_def_stack[y*conf.px_hori + x] << " ";
    //        std::cout << img_roi[y*conf.px_hori+x] << std::endl;
    //    }
    //}
    //exit(0);

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<util::SubsetData> ssdata;
    std::vector<int> ss_sizes, ss_steps;
    if ((conf.scan_method == "FFT") || (conf.scan_method == "RG")){
        util::Timer timer("subset list initialisation");
        util::gen_size_and_step_vector(ss_sizes, ss_steps, conf.ss_size, conf.ss_step, conf.max_disp);
        fourier::init(ssdata, ss_sizes, ss_steps, img_roi, conf);
    }
    else {
        util::Timer timer("subset list initialisation");
        ssdata.push_back(util::gen_ss_list(img_roi, conf.ss_step,
                                           conf.ss_size, conf.px_hori, 
                                           conf.px_vert));
    }



    // resize the results based on subset information
    util::resize_results(conf.num_def_img, ssdata.back().num,
                         conf.num_params, saveconf.at_end);

    // initialise the LM optimizer with shape func and corr crit
    optimizer::init(conf.corr_crit, conf.shape_func);

    // initialise the brute force scan
    std::string brute_method = "EXPANDING_WAVEFRONT";
    brute::init(conf.corr_crit, brute_method);



    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    std::cout << std::endl;
    TITLE("Starting Correlation")
    util::Timer timer("DIC Engine:");
    for (int img_num = 0; img_num < conf.num_def_img; img_num++){

        // pointer to starting location of deformed image in memory
        int num_px_in_image = conf.px_hori * conf.px_vert;
        double *img_def = img_def_stack + img_num*num_px_in_image;

        // define our interpolator for the reference image
        Interpolator interp_def(img_def, conf.px_hori, conf.px_vert);

        // raster scan
        if (conf.scan_method=="IMAGE_SCAN") 
            scanmethod::image(img_ref, interp_def, ssdata[0], conf, img_num);

        // raster with brute force
        else if (conf.scan_method=="IMAGE_SCAN_WITH_BF") 
            scanmethod::image_with_bf(img_ref, img_def, interp_def, ssdata[0], conf, img_num);

        // reliability Guided
        else if (conf.scan_method=="RG")
            scanmethod::reliability_guided(img_ref, img_def, interp_def, ssdata, conf, img_num, saveconf.at_end);

        // multi window fft
        else if (conf.scan_method=="FFT")
            scanmethod::multi_window_fourier(img_ref, img_def, interp_def, ssdata, conf, img_num);

        // single window fft
        else if (conf.scan_method=="FFT_test")
            scanmethod::single_window_fourier(img_ref, img_def, interp_def, ssdata[0], conf, img_num);

        if (!saveconf.at_end)
            util::save_to_disk(img_num, saveconf, ssdata.back(), conf.num_def_img, conf.num_params, conf.filenames);

    }

    if (saveconf.at_end)
        for (int img_num = 0; img_num < conf.num_def_img; img_num++)
            util::save_to_disk(img_num, saveconf, ssdata.back(), conf.num_def_img, conf.num_params, conf.filenames);
}


void build_info(){
        //std::cout << "Buld Information:" << std::endl;
        //INFO_OUT("- g++ version:", CPUCOMP);
        //INFO_OUT("- Co
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
        .def_readwrite("opt_threshold", &util::Config::opt_threshold)
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
        .def_readwrite("filenames", &util::Config::filenames);

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




