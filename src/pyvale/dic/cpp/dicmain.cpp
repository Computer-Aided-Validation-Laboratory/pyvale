// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iomanip>
#include <iostream>
#include <cstring>
#include <vector>
#include <signal.h>
#include <memory>
#include <algorithm>

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
#include "./dicinterpfactory.hpp"
#include "./dicscanmethod.hpp"
#include "./dicutil.hpp"
#include "./dicresults.hpp"
#include "./dicsubset.hpp"
#include "./dicmultiwindow.hpp"

// stereo header files
#include "./stereomatching.hpp"
#include "./stereoutil.hpp"

// cuda Header files
//#include "../cuda/malloc.hpp"

namespace py = pybind11;


void engine(const py::array_t<double>& img_stack_arr,
               const py::array_t<bool>&   img_roi_arr, 
               const Calib &calib,
               const util::Config &conf,
               const common_util::SaveConfig &saveconf){

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


    int num_px_in_image = conf.px_hori * conf.px_vert;

    // get raw pointers
    bool* img_roi = static_cast<bool*>(img_roi_arr.request().ptr);
    double* img_stack = static_cast<double*>(img_stack_arr.request().ptr);

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<WindowLevel> multiwindow_l;
    subset::Grid ss_grid_l;   // actual object, not reference

    if (conf.scan_method == "MULTIWINDOW_RG" || conf.scan_method == "MULTIWINDOW") {

        multiwindow_init(multiwindow_l, img_roi, conf);
        ss_grid_l = multiwindow_l.back().layout;

    }
    else if (conf.scan_method == "SINGLEWINDOW_RG" ||
             conf.scan_method == "SINGLEWINDOW_RG_INCREMENTAL" ||
             conf.scan_method == "IMAGE_SCAN") {

        ss_grid_l = subset::create_grid(img_roi, conf.ss_step,
                                        conf.ss_size, conf.ss_size,
                                        conf.px_hori, conf.px_vert, false);
    }
    else {
        std::cout << "ERROR " << std::endl;
        exit(0);
    }

    // resize the results based on subset information
    ResultArrays result_arrays_l(ss_grid_l.num, conf.num_params, false);

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    if (g_debug_level>0){
        std::cout << std::endl;
        TITLE("Starting Correlation")
    }
    common_util::Timer timer("DIC Engine:");

    // pointer to reference images at start of stack
    double *img_ref_l = img_stack;
    double *img_ref_r = nullptr;

    // pointer to hold the reference interpolators (will be created once)
    std::unique_ptr<Interpolator> interp_ref_l;
    std::unique_ptr<Interpolator> interp_ref_r;
    std::unique_ptr<Interpolator> interp_def_l;
    std::unique_ptr<Interpolator> interp_def_r;
    std::unique_ptr<Interpolator> interp_ref_l_inc;
    std::unique_ptr<Interpolator> interp_ref_r_inc;


    // objects only needed for stereo
    std::vector<WindowLevel> multiwindow_r;
    subset::Grid *ss_grid_r = nullptr;
    ResultArrays stereo_matches;
    ResultArrays result_arrays_r;
    stereo::Geometry stereo_geom;
    common_util::SaveConfig saveconf_stereo;

    // split out filenames into two vectors
    auto [filenames_l, filenames_r] = stereo::split_filenames(conf);

    // main bulk of initialisation for stereo matching
    int match_strat=3;

    if (conf.stereo){

        // image series goes l0,r0,l1,r1,l2,r2
        img_ref_r = img_stack + (conf.num_def_img+1)*num_px_in_image; // r0 image

        // interpolators
        interp_ref_l = make_interp(conf.interp_routine, img_ref_l, conf.px_hori, conf.px_vert);
        interp_ref_r = make_interp(conf.interp_routine, img_ref_r, conf.px_hori, conf.px_vert);

        // resize the results based on subset information
        result_arrays_r = ResultArrays(ss_grid_l.num, conf.num_params, true);


        stereo_geom = stereo::compute_stereo_geometry(calib);

        // if (match_strat==1){
        //     stereo_matches = ResultArrays(ss_grid_l.num, conf.num_params, true);
        //     // updated ROI for the right image
        //     bool *img_roi_r = stereo::compute_roi_r(ss_grid_l, stereo_matches,
        //                                         conf.px_hori, conf.px_vert,
        //                                         conf.ss_size, conf.ss_size);
        //
        //     // multiwindow setup for the right image;
        //     multiwindow_init(multiwindow_r, img_roi_r, conf);
        //
        //     multiwindow_r.back() = multiwindow_l.back();
        //
        //     // subset information in the right image
        //     ss_grid_r = &multiwindow_r.back().layout;
        //
        //     for (int ss = 0; ss < ss_grid_r->num; ss++){
        //     ss_grid_r->coords[2*ss]   += stereo_matches.u[ss];
        //     ss_grid_r->coords[2*ss+1] += stereo_matches.v[ss];
        //     }
        //
        //     // need to redo the neighlist
        //     multiwindow_r.back().gen_neighlist(multiwindow_r[multiwindow_r.size()-2].layout);
        //
        //     // subset information in the right image
        //     ss_grid_r = &multiwindow_r.back().layout;
        //
        //     stereo::remove_unmatched_subsets(ss_grid_l, *ss_grid_r, stereo_matches);
        // }
    }




    ResultArrays result_arrays_l_prev(ss_grid_l.num, conf.num_params, false);


    // loop over deformed images. They start at index 1 in the stack
    for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){

        // Pointer to starting location of deformed image in memory.
        // Layout is l0,l1,l2,...,ln,r0,r1,r2,r3,...,rn,
        const int img_num_l = img_num;
        const int img_num_r = conf.num_def_img+1+img_num;
        double *img_def_l = img_stack + img_num_l*num_px_in_image;
        double *img_def_r = img_stack + img_num_r*num_px_in_image;



        // interpolator for the L image
        interp_def_l = make_interp(conf.interp_routine, img_def_l, conf.px_hori, conf.px_vert);

        // interpolator for the R image
        if (conf.stereo) {
            interp_def_r = make_interp(conf.interp_routine, img_def_r, conf.px_hori, conf.px_vert);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        // raster scan
        // -------------------------------------------------------------------------------------------------------------------------------------------
        if (conf.scan_method=="IMAGE_SCAN") {
            if (conf.stereo) {std::cerr << "ERROR: Scan method=\"IMAGE_SCAN\" does not support stereo" << std::endl; return;}
            scanmethod::image(img_ref_l, *interp_def_l, ss_grid_l, conf, 0, img_num, result_arrays_l);
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        // multiwindow FFTCC + reliability Guided
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="MULTIWINDOW_RG"){
            scanmethod::multiwindow_reliability_guided(img_ref_l, img_def_l, *interp_def_l,
                                                       multiwindow_l, conf, 0, img_num_l,
                                                       result_arrays_l);

            if (conf.stereo) {
                if (match_strat==3) {
                    stereo::matching(img_def_l, img_def_r, *interp_def_l, *interp_def_r,
                                        ss_grid_l, conf, img_num_l, img_num_r,
                                        stereo_geom.F, result_arrays_l, result_arrays_r);

                    stereo::pixel_to_world(ss_grid_l, calib,
                                           result_arrays_l, result_arrays_r,
                                           stereo_geom.K0, stereo_geom.K1,
                                           stereo_geom.R, conf.ss_size);
                }
                else {
                    std::cerr << "UNKNOWN MATCH_STRAT!!" << std::endl; exit(0);
                }
            }
        }

        // -------------------------------------------------------------------------------------------------------------------------------------------
        // singlewindow FFTCC + reliability Guided
        // -------------------------------------------------------------------------------------------------------------------------------------------
        else if (conf.scan_method=="SINGLEWINDOW_RG"){
            if (!interp_ref_l) interp_ref_l = make_interp(conf.interp_routine,
                                                          img_ref_l,
                                                          conf.px_hori,
                                                          conf.px_vert);

            scanmethod::singlewindow_incremental_reliability_guided(img_ref_l, img_def_l,
                                                                    *interp_ref_l, *interp_def_l,
                                                                    ss_grid_l, conf, 0, img_num,
                                                                    result_arrays_l_prev,
                                                                    result_arrays_l);
        }


        // ----------------------------------------------------------------------------------------
        // multi window FFTCC ONLY
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method=="MULTIWINDOW")
            scanmethod::multiwindow_only(img_ref_l,
                                         img_def_l,
                                         *interp_def_l,
                                         multiwindow_l,
                                         conf,
                                         0, img_num,
                                         result_arrays_l);


        // ----------------------------------------------------------------------------------------
        // singlewindow FFTCC + reliability Guided + Incremental Updating
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method=="SINGLEWINDOW_RG_INCREMENTAL"){


            double *img_prev = img_stack + (img_num-1)*num_px_in_image;
            interp_ref_l_inc = make_interp(conf.interp_routine, img_prev, conf.px_hori, conf.px_vert);
            std::swap(result_arrays_l_prev, result_arrays_l);

            scanmethod::singlewindow_incremental_reliability_guided(img_prev, img_def_l, 
                                                                    *interp_ref_l_inc, 
                                                                    *interp_def_l, 
                                                                    ss_grid_l, conf, 
                                                                    img_num-1, img_num, 
                                                                    result_arrays_l_prev,
                                                                    result_arrays_l);


            // REFERENCE IMAGE UPDATING CONDITIONS
            // double avg_iter = 0.0;
            // for (int j = 0; j < result_arrays_l.niter.size(); j++){
            //     avg_iter += result_arrays_l.niter[j];
            // }
            // avg_iter /= result_arrays_l.niter.size();
            // std::cout << "avg_iter: " << avg_iter << std::endl;
        }

        if (!saveconf.at_end){
            if (!conf.stereo){
                write_to_disk_2d(result_arrays_l,saveconf, ss_grid_l, filenames_l[img_num]);
            }
            else {
                write_to_disk_stereo(result_arrays_l,result_arrays_r, saveconf, ss_grid_l, filenames_l[img_num]);
            }
        }

        if (stop_request) break;
    }

    // if (saveconf.at_end)
    //     for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){
    //         result_arrays_l.write_to_disk(img_num, saveconf, ss_grid_l, conf.num_def_img, filenames_l);
    //         if (conf.stereo) result_arrays_r.write_to_disk(img_num, saveconf, *ss_grid_r, conf.num_def_img, filenames_r);
    //     }
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



