// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iomanip>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <signal.h>
#include <memory>
#include <algorithm>
#include <numeric>

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
#include "./dicutil.hpp"
#include "./dicresults.hpp"
#include "./dicsubset.hpp"
#include "./dicroiupdate.hpp"
#include "./dicmain.hpp"
#include "./dicmultiwindow_rg.hpp"
#include "./dicmultiwindow_only.hpp"
#include "./dicsinglewindow_rg.hpp"
#include "./dicrasterscan.hpp"

// stereo header files
#include "./stereoutil.hpp"

// cuda Header files
//#include "../cuda/malloc.hpp"

namespace py = pybind11;


void engine(const py::array_t<bool>& img_roi_arr, 
            const Calib &calib,
            const util::Config &conf,
            const MultiwindowConfig &mwconf,
            const common_util::SaveConfig &saveconf){

    // Register signal handler for Ctrl+C and set debug_level
    signal(SIGINT, signalHandler);
    g_debug_level = conf.debug_level;

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    if (g_debug_level>0){
        common_util::title("Starting Correlation");
    }


    int num_px_in_image = conf.px_hori * conf.px_vert;

    // get raw pointers
    const bool *img_roi = img_roi_arr.data();

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<WindowLevel> multiwindow_l;
    subset::Grid ss_grid_l;
    subset::Grid ss_grid_l_inc;

    if (conf.scan_method == util::ScanMethod::MULTIWINDOW_RG || conf.scan_method == util::ScanMethod::MULTIWINDOW) {
        multiwindow_init(multiwindow_l, img_roi, conf, mwconf, saveconf);
        ss_grid_l = multiwindow_l.back().layout;
        ss_grid_l_inc = ss_grid_l;
    }
    else if (conf.scan_method == util::ScanMethod::SINGLEWINDOW_RG ||
             conf.scan_method == util::ScanMethod::RASTER) {

        common_util::Timer timer("to create subset grid:", 2);
        ss_grid_l = subset::create_grid(img_roi, conf.ss_step,
                                        conf.ss_size, conf.ss_size,
                                        conf.px_hori, conf.px_vert, false);
    }
    else {
        throw std::invalid_argument("Unsupported scan method");
    }


    // resize the results based on subset information
    ResultArrays results_def_l(ss_grid_l.num, conf.num_params, false);


    // pointer to hold the reference interpolators (will be created once)
    std::unique_ptr<Interpolator> interp_ref_l;
    std::unique_ptr<Interpolator> interp_ref_r;
    std::unique_ptr<Interpolator> interp_def_l;
    std::unique_ptr<Interpolator> interp_def_r;
    interp_ref_l = make_interp(conf.interp_routine, conf.fullpaths[0]);


    // objects only needed for stereo
    std::vector<WindowLevel> multiwindow_r;
    subset::Grid *ss_grid_r = nullptr;
    ResultArrays stereo_matches;
    ResultArrays results_ref_r;
    ResultArrays results_def_r;
    stereo::Geometry stereo_geom;
    common_util::SaveConfig saveconf_stereo;

    // split out filenames into two vectors
    auto [basenames_l, basenames_r] = stereo::split_basenames(conf);

    // main bulk of initialisation for stereo matching
    int match_strat=3;

    ResultArrays results_ref_l(ss_grid_l.num, conf.num_params, false);

    if (conf.stereo){

        // resize the results based on subset information
        results_ref_r = ResultArrays(ss_grid_l.num, conf.num_params, true);
        results_def_r = ResultArrays(ss_grid_l.num, conf.num_params, true);

        // sort out intrinsic and extrinsic matrices into struct
        stereo_geom = stereo::compute_stereo_geometry(calib);

        std::unique_ptr<Interpolator> interp_l = make_interp(conf.interp_routine, conf.fullpaths[0]);
        std::unique_ptr<Interpolator> interp_r = make_interp(conf.interp_routine, conf.fullpaths[conf.num_def_img+1]);
    }


    int img_num_ref_l = 0;

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){

        int img_num_def_l = img_num;
        int img_num_def_r = conf.num_def_img+1+img_num;

        // interpolator for the L image
        interp_def_l = make_interp(conf.interp_routine, conf.fullpaths[img_num_def_l]);

        // interpolator for the R image
        if (conf.stereo) {
            interp_def_r = make_interp(conf.interp_routine, conf.fullpaths[img_num_def_r]);
        }

        // clear the image buffers. Dont need them anymore

        // ----------------------------------------------------------------------------------------
        // raster scan
        // ----------------------------------------------------------------------------------------
        if (conf.scan_method == util::ScanMethod::RASTER) {
            if (conf.stereo) 
                throw std::invalid_argument("Unsupported scan method");

            results_def_l.reset();
            results_def_r.reset();

            raster(*interp_ref_l,
                   *interp_def_l,
                   ss_grid_l,
                   conf, 0,
                   img_num,
                   results_def_l);
        }


        // ----------------------------------------------------------------------------------------
        // multiwindow FFTCC
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == util::ScanMethod::MULTIWINDOW) {

            bool update_ref = conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf);
            if (update_ref) {
                img_num_ref_l = img_num_def_l - 1;
                results_ref_l = results_def_l;
                interp_ref_l = make_interp(conf.interp_routine, conf.fullpaths[img_num_ref_l]);

                bool* roi_updated = propagate_roi(img_roi, results_def_l, conf, ss_grid_l);
                multiwindow_l.clear();
                multiwindow_init_partial(multiwindow_l, roi_updated, conf, mwconf, saveconf,
                                        mwconf.overlap.size() - 1);

                WindowLevel last_level;
                last_level.u.assign(ss_grid_l.num, 0.0);
                last_level.v.assign(ss_grid_l.num, 0.0);
                last_level.cost.assign(ss_grid_l.num, 0.0);
                last_level.max_val.assign(ss_grid_l.num, 0.0);
                last_level.level         = mwconf.overlap.size() - 1;
                last_level.fft_filter             = conf.fft_filter;
                last_level.fft_filter_threshold   = conf.fft_filter_threshold;
                last_level.fft_filter_radius      = conf.fft_filter_radius;
                last_level.fft_filter_corr_power  = conf.fft_filter_corr_power;
                last_level.fft_save      = conf.fft_save;
                last_level.saveconf      = saveconf;
                last_level.step          = mwconf.overlap.back();
                last_level.template_size = mwconf.subset_size.back();
                last_level.search_area   = mwconf.search_area.back();


                // Step 1: update active flags on ss_grid_l
                if (img_num_def_l > 1){
                    for (int i = 0; i < ss_grid_l.num; i++) {
                        if (!results_ref_l.above_thresh[i]) {
                            ss_grid_l.active_ss[i] = false;
                        }
                    }
                    ss_grid_l.active_total = std::count(ss_grid_l.active_ss.begin(),
                                                        ss_grid_l.active_ss.end(), true);
                }
                last_level.layout = ss_grid_l;
                for (int i = 0; i < ss_grid_l.num; i++) {
                    if (ss_grid_l.active_ss[i]) {
                        last_level.layout.coords[2*i]   += results_ref_l.u[i];
                        last_level.layout.coords[2*i+1] += results_ref_l.v[i];
                    }
                }

                multiwindow_l.push_back(std::move(last_level));
                if (multiwindow_l.size() > 1) {
                    multiwindow_l.back().gen_neighlist(multiwindow_l[multiwindow_l.size()-2].layout);
                }

            }


            results_def_l.reset();
            results_def_r.reset();

            multiwindow_only(*interp_ref_l, 
                             *interp_def_l,
                             multiwindow_l, 
                             conf,
                             img_num_ref_l, 
                             img_num_def_l, 
                             results_ref_l, 
                             results_def_l);



        }


        // ----------------------------------------------------------------------------------------
        // singlewindow FFTCC + RG
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == util::ScanMethod::SINGLEWINDOW_RG) {
            if (conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf)) {

                // update left image vars
                img_num_ref_l = img_num_def_l - 1;
                results_ref_l = results_def_l;
                interp_ref_l = make_interp(conf.interp_routine, conf.fullpaths[img_num_ref_l]);

                // update right image vars
                if (conf.stereo)
                    results_ref_r = results_def_r;

                // Step 1: update active flags on ss_grid_l
                if (img_num_def_l > 1){
                    for (int i = 0; i < ss_grid_l.num; i++) {
                        if (!results_ref_l.above_thresh[i]) {
                            ss_grid_l.active_ss[i] = false;
                        }
                    }
                    ss_grid_l.active_total = std::count(ss_grid_l.active_ss.begin(),
                                                        ss_grid_l.active_ss.end(), true);
                }
            }

            results_def_l.reset();
            results_def_r.reset();

            singlewindow_rg(*interp_ref_l, 
                            *interp_def_l,
                            ss_grid_l, 
                            conf, 
                            img_num_ref_l, 
                            img_num_def_l,
                            results_ref_l, 
                            results_def_l);

            if (conf.stereo) {

                if (match_strat != 3) {
                    std::cerr << "UNKNOWN MATCH_STRAT\n";
                    exit(0);
                }

                singlewindow_rg(*interp_ref_l,
                                *interp_def_r,
                                ss_grid_l,
                                conf,
                                img_num_ref_l,
                                img_num_def_r,
                                results_ref_l,
                                results_def_r,
                                "stereo",
                                stereo_geom.F);

                stereo::pixel_to_world(ss_grid_l,
                                    calib,
                                    results_def_l,
                                    results_ref_r,
                                    results_def_r,
                                    stereo_geom.K0,
                                    stereo_geom.K1,
                                    stereo_geom.R,
                                    conf.ss_size,
                                    (img_num_def_l==1));
            }
        }


        // ----------------------------------------------------------------------------------------
        // multiwindow FFTCC + reliability Guided
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == util::ScanMethod::MULTIWINDOW_RG) {

            bool update_ref = conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf);
            if (update_ref) {

                // update left image vars
                img_num_ref_l = img_num_def_l - 1;
                results_ref_l = results_def_l;
                interp_ref_l = make_interp(conf.interp_routine, conf.fullpaths[img_num_ref_l]);

                // update right image vars
                if (conf.stereo)
                    results_ref_r = results_def_r;


                bool* roi_updated = propagate_roi(img_roi, results_def_l, conf, ss_grid_l);
                multiwindow_l.clear();
                multiwindow_init_partial(multiwindow_l, roi_updated, conf, mwconf, saveconf,
                                        mwconf.overlap.size() - 1);

                WindowLevel last_level;
                last_level.u.assign(ss_grid_l.num, 0.0);
                last_level.v.assign(ss_grid_l.num, 0.0);
                last_level.cost.assign(ss_grid_l.num, 0.0);
                last_level.max_val.assign(ss_grid_l.num, 0.0);
                last_level.level         = mwconf.overlap.size() - 1;
                last_level.fft_filter             = conf.fft_filter;
                last_level.fft_filter_threshold   = conf.fft_filter_threshold;
                last_level.fft_filter_radius      = conf.fft_filter_radius;
                last_level.fft_filter_corr_power  = conf.fft_filter_corr_power;
                last_level.fft_save      = conf.fft_save;
                last_level.saveconf      = saveconf;
                last_level.step          = mwconf.overlap.back();
                last_level.template_size = mwconf.subset_size.back();
                last_level.search_area   = mwconf.search_area.back();


                // Step 1: update active flags on ss_grid_l
                if (img_num_def_l > 1){
                    for (int i = 0; i < ss_grid_l.num; i++) {
                        if (!results_ref_l.above_thresh[i]) {
                            ss_grid_l.active_ss[i] = false;
                        }
                    }
                    ss_grid_l.active_total = std::count(ss_grid_l.active_ss.begin(),
                                                        ss_grid_l.active_ss.end(), true);
                }
                last_level.layout = ss_grid_l;
                for (int i = 0; i < ss_grid_l.num; i++) {
                    if (ss_grid_l.active_ss[i]) {
                        last_level.layout.coords[2*i]   += results_ref_l.u[i];
                        last_level.layout.coords[2*i+1] += results_ref_l.v[i];
                    }
                }

                multiwindow_l.push_back(std::move(last_level));
                if (multiwindow_l.size() > 1) {
                    multiwindow_l.back().gen_neighlist(multiwindow_l[multiwindow_l.size()-2].layout);
                }

            }

            results_def_l.reset();
            results_def_r.reset();

            multiwindow_rg(*interp_ref_l, 
                           *interp_def_l,
                           multiwindow_l, 
                           conf, 
                           img_num_ref_l, 
                           img_num_def_l,
                           results_ref_l, 
                           results_def_l);

            // multiwindow_rg_stereo(*interp_ref_l, 
            //                       *interp_def_l,
            //                       *interp_def_r,
            //                       multiwindow_l, 
            //                       conf, 
            //                       img_num_ref_l, 
            //                       img_num_def_l,
            //                       results_ref_l, 
            //                       results_ref_r, 
            //                       results_def_l,
            //                       results_def_r,
            //                       stereo_geom.F);


            // ------------------------------------------------------------
            // stereo
            // ------------------------------------------------------------
            if (conf.stereo) {

                if (match_strat != 3) {
                    std::cerr << "UNKNOWN MATCH_STRAT\n";
                    exit(0);
                }

                singlewindow_rg(*interp_ref_l,
                                *interp_def_r,
                                multiwindow_l.back().layout,
                                conf,
                                img_num_ref_l,
                                img_num_def_r,
                                results_ref_l,
                                results_def_r,
                                "stereo",
                                stereo_geom.F,
                                results_def_l);

                stereo::pixel_to_world(multiwindow_l.back().layout,
                                    calib,
                                    results_def_l,
                                    results_ref_r,
                                    results_def_r,
                                    stereo_geom.K0,
                                    stereo_geom.K1,
                                    stereo_geom.R,
                                    conf.ss_size,
                                    (img_num_def_l==1));
            }

        }
        else {
            throw std::invalid_argument("Unsupported scan method");
        }

        if (!conf.stereo){
            results_def_l.write_to_disk_2d(saveconf, ss_grid_l, basenames_l[img_num]);
        }
        else {
            results_def_l.write_to_disk_stereo(results_def_r, saveconf, ss_grid_l, basenames_l[img_num]);
        }

        if (stop_request) break;
    }
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

bool should_update_ref(const int img_num_def_l, const ResultArrays& results, const util::Config& conf) {

    // never update on the first deformed image
    if (img_num_def_l == 1) {
        return false;
    }

    switch (conf.incremental_update_cond) {

        case util::IncrementalCond::IMAGE: {
            int interval = static_cast<int>(conf.incremental_update_val);
            return (img_num_def_l - 1) % interval == 0;
        }

        case util::IncrementalCond::ITER: {
            if (results.niter.empty()) return false;

            double avg = std::accumulate(results.niter.begin(), results.niter.end(), 0.0) / results.niter.size();
            return avg > conf.incremental_update_val;
        }

        case util::IncrementalCond::COST: {
            if (results.cost.empty()) return false;

            double avg = std::accumulate(results.cost.begin(), results.cost.end(), 0.0) / results.cost.size();
            return avg > conf.incremental_update_val;
        }

        default:
            throw std::runtime_error(
                "Unknown incremental update condition: " +
                std::to_string(static_cast<int>(conf.incremental_update_cond))
            );
    }
}
