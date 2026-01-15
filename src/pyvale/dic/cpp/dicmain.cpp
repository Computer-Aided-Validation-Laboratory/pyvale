// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <atomic>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>
#include <signal.h>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// common_cpp header files
#include "../../common_cpp/dicsignalhandler.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/util.hpp"

// DIC Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicscanmethod.hpp"
#include "./dicutil.hpp"
#include "./dicfourier.hpp"
#include "./dicshapefunc.hpp"
#include "./dicresults.hpp"
#include "dicsubset.hpp"

// cuda Header files
//#include "../cuda/malloc.hpp"

namespace py = pybind11;


void DICengine(const py::array_t<double>& img_stack_arr,
               const py::array_t<bool>&   img_roi_arr, 
               util::Config &conf,
               common_util::SaveConfig &saveconf){

    // Register signal handler for Ctrl+C and set debug_level
    signal(SIGINT, signalHandler);
    g_debug_level = conf.debug_level;

    // ------------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------------
    if (g_debug_level>0){
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
    INFO_OUT("Correlation Cutoff Threshold:", conf.threshold);
    INFO_OUT("Estimate for Max Displacement:", conf.max_disp << " [px]");
    INFO_OUT("Subset Size:", conf.ss_size << " [px]");
    INFO_OUT("Subset Step:", conf.ss_step << " [px]" );
    INFO_OUT("Number of OMP threads:", omp_get_max_threads());
    INFO_OUT("Debug level: ", conf.debug_level);
    if (conf.scan_method.find("RG") != std::string::npos)INFO_OUT("Reliability Guided Seed central px location: ", "(" 
                                         << conf.rg_seed.first+conf.ss_size/2 << ", " << conf.rg_seed.second+conf.ss_size/2 << ") [px] " )
    }

    // get raw pointers
    bool* img_roi = static_cast<bool*>(img_roi_arr.request().ptr);
    double* img_stack = static_cast<double*>(img_stack_arr.request().ptr);

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<subset::Grid> ss_grids;
    std::vector<int> ss_sizes, ss_steps;
    util::gen_size_and_step_vector(ss_sizes, ss_steps, conf.ss_size, conf.ss_step, conf.max_disp);
    fourier::init(ss_grids, ss_sizes, ss_steps, img_roi, conf);


    // resize the results based on subset information
    OptResultArrays result_arrays(conf.num_def_img, ss_grids.back().num,
                               conf.num_params, saveconf.at_end);


    // set relevent shape function
    shapefunc::set(conf.shape_func);

    // set cost function to use in optimization
    optimizer::set_cost_function(conf.corr_crit);



    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    if (g_debug_level>0){
        std::cout << std::endl;
        TITLE("Starting Correlation")
    }
    common_util::Timer timer("DIC Engine:");

    // pointer to reference image at start of stack
    double *img_ref = img_stack;
    
    // pointer to hold the reference interpolator (will be created once)
    Interpolator* interp_ref = nullptr;

    // loop over deformed images. They start at index 1 in the stack
    for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){

        // pointer to starting location of deformed image in memory
        int num_px_in_image = conf.px_hori * conf.px_vert;
        double *img_def = img_stack + img_num*num_px_in_image;

        // define our interpolator for the reference image
        Interpolator interp_def(img_def, conf.px_hori, conf.px_vert);

        // -------------------------------------------------------------------------------------------------------------------------------------------
        // raster scan
        // -------------------------------------------------------------------------------------------------------------------------------------------
        if (conf.scan_method=="IMAGE_SCAN") 
            scanmethod::image(img_ref, interp_def, ss_grids[0], conf, img_num, result_arrays);




        // -------------------------------------------------------------------------------------------------------------------------------------------
        // multiwindow FFTCC + reliability Guided
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="MULTIWINDOW_RG")
            scanmethod::multiwindow_reliability_guided(img_ref, img_def, interp_def, ss_grids, conf, img_num, result_arrays);




        // -------------------------------------------------------------------------------------------------------------------------------------------
        // singlewindow FFTCC + reliability Guided
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="SINGLEWINDOW_RG"){
            if (!interp_ref) interp_ref = new Interpolator(img_ref, conf.px_hori, conf.px_vert);
            scanmethod::singlewindow_incremental_reliability_guided(img_ref, img_def, *interp_ref, interp_def, ss_grids, conf, 0, img_num, result_arrays);
        }



        // -------------------------------------------------------------------------------------------------------------------------------------------
        // multi window FFTCC ONLY
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="MULTIWINDOW")
            scanmethod::multiwindow(img_ref, img_def, interp_def, ss_grids, conf, img_num, result_arrays);




        // -------------------------------------------------------------------------------------------------------------------------------------------
        // singlewindow FFTCC + reliability Guided + Incremental Updating
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="SINGLEWINDOW_RG_INCREMENTAL"){
            double *img_prev = nullptr;
            int img_num_prev = img_num-1;
            img_prev = img_stack + img_num_prev*num_px_in_image;
            Interpolator interp_ref_inc(img_prev, conf.px_hori, conf.px_vert);
            scanmethod::singlewindow_incremental_reliability_guided(img_prev, img_def, interp_ref_inc, interp_def, ss_grids, conf, img_num_prev, img_num, result_arrays);
        }





        if (!saveconf.at_end)
            result_arrays.write_to_disk(img_num, saveconf, ss_grids.back(), conf.num_def_img, conf.filenames);

        if (stop_request) break;
    }

    if (saveconf.at_end)
        for (int img_num = 1; img_num < conf.num_def_img+1; img_num++)
            result_arrays.write_to_disk(img_num, saveconf, ss_grids.back(), conf.num_def_img, conf.filenames);


    if (interp_ref) delete interp_ref;

    // TODO: don't have shifts as a global var. Should probably make fourier
    // stuff a class at some point in the future
    fourier::shifts.clear();
    fourier::shifts.shrink_to_fit();
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



