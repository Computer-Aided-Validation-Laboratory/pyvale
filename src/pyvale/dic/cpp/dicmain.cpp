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

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// common_cpp header files
#include "../../common_cpp/dicsignalhandler.hpp"
#include "../../common_cpp/img_read.hpp"
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
#include "./stereomatching.hpp"
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
    // if (conf.scan_method.find("RG") != std::string::npos)INFO_OUT("Reliability Guided Seed central px location: ", "(" 
    //                                      << conf.rg_seed.first+conf.ss_size/2 << ", " << conf.rg_seed.second+conf.ss_size/2 << ") [px] " )
    }


    int num_px_in_image = conf.px_hori * conf.px_vert;

    // get raw pointers
    bool* img_roi = static_cast<bool*>(img_roi_arr.request().ptr);

    // ------------------------------------------------------------------------
    // get a list of ss coordinates within RIO;
    // ------------------------------------------------------------------------
    std::vector<WindowLevel> multiwindow_l;
    subset::Grid ss_grid_l;
    subset::Grid ss_grid_l_inc;

    if (conf.scan_method == "MULTIWINDOW_RG" || conf.scan_method == "MULTIWINDOW") {
        multiwindow_init(multiwindow_l, img_roi, conf, mwconf, saveconf);
        ss_grid_l = multiwindow_l.back().layout;
        ss_grid_l_inc = ss_grid_l;
    }
    else if (conf.scan_method == "SINGLEWINDOW_RG" ||
             conf.scan_method == "SINGLEWINDOW_RG_INCREMENTAL" ||
             conf.scan_method == "RASTER") {
        ss_grid_l = subset::create_grid(img_roi, conf.ss_step,
                                        conf.ss_size, conf.ss_size,
                                        conf.px_hori, conf.px_vert, false);
    }
    else {
        throw std::invalid_argument("Unsupported scan method: " + conf.scan_method);
    }


    // resize the results based on subset information
    ResultArrays results_def_l(ss_grid_l.num, conf.num_params, false);

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    if (g_debug_level>0){
        std::cout << std::endl;
        TITLE("Starting Correlation")
    }
    common_util::Timer timer("DIC Engine:");

    // pointer to reference images at start of stack
    //double *img_ref_l = img_stack;
    Image img_ref_l = read_img(conf.fullpaths[0]);
    Image img_ref_r;

    // pointer to hold the reference interpolators (will be created once)
    std::unique_ptr<Interpolator> interp_ref_l;
    std::unique_ptr<Interpolator> interp_ref_r;
    std::unique_ptr<Interpolator> interp_def_l;
    std::unique_ptr<Interpolator> interp_def_r;
    interp_ref_l = make_interp(conf.interp_routine, img_ref_l);


    // objects only needed for stereo
    std::vector<WindowLevel> multiwindow_r;
    subset::Grid *ss_grid_r = nullptr;
    ResultArrays stereo_matches;
    ResultArrays results_def_r;
    stereo::Geometry stereo_geom;
    common_util::SaveConfig saveconf_stereo;

    // split out filenames into two vectors
    auto [basenames_l, basenames_r] = stereo::split_basenames(conf);

    // main bulk of initialisation for stereo matching
    int match_strat=3;

    if (conf.stereo){

        // image series goes l0,r0,l1,r1,l2,r2
        //img_ref_r = img_stack + (conf.num_def_img+1)*num_px_in_image; // r0 image
        img_ref_r = read_img(conf.fullpaths[conf.num_def_img+1]);

        // interpolators
        interp_ref_l = make_interp(conf.interp_routine, img_ref_l);
        interp_ref_r = make_interp(conf.interp_routine, img_ref_r);

        // resize the results based on subset information
        results_def_r = ResultArrays(ss_grid_l.num, conf.num_params, true);

        // sort out intrinsic and extrinsic matrices into struct
        stereo_geom = stereo::compute_stereo_geometry(calib);
    }

    ResultArrays results_ref_l(ss_grid_l.num, conf.num_params, false);

    // loop over deformed images. They start at index 1 in the stack
    for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){

        int img_num_ref_l = 0;
        int img_num_def_l = img_num;
        int img_num_def_r = conf.num_def_img+1+img_num;

        Image img_def_l, img_def_r;

        // interpolator for the L image
        img_def_l = read_img(conf.fullpaths[img_num_def_l]);
        interp_def_l = make_interp(conf.interp_routine, img_def_l);

        // interpolator for the R image
        if (conf.stereo) {
            img_def_r = read_img(conf.fullpaths[img_num_def_r]);
            interp_def_r = make_interp(conf.interp_routine, img_def_r);
        }
        // ----------------------------------------------------------------------------------------
        // raster scan
        // ----------------------------------------------------------------------------------------
        if (conf.scan_method == "RASTER") {
            if (conf.stereo) 
                throw std::invalid_argument("Unsupported scan method: " + conf.scan_method);

            raster(img_ref_l, *interp_def_l, ss_grid_l, conf, 0, img_num, results_def_l);
        }


        // ----------------------------------------------------------------------------------------
        // multiwindow FFTCC
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == "MULTIWINDOW") {

            bool update_ref = conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf);
            if (update_ref) {
                img_num_ref_l = img_num_def_l - 1;
                img_ref_l = read_img(conf.fullpaths[img_num_ref_l]);
                interp_ref_l = make_interp(conf.interp_routine, img_ref_l);

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
                last_level.mad_filter    = conf.fft_mad;
                last_level.mad_scale     = conf.fft_mad_scale;
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
                }
                last_level.layout = ss_grid_l;
                for (int i = 0; i < ss_grid_l.num; i++) {
                    if (ss_grid_l.active_ss[i]) {
                        last_level.layout.coords[2*i]   += results_ref_l.u[i];
                        last_level.layout.coords[2*i+1] += results_ref_l.v[i];
                    }
                }

                multiwindow_l.push_back(std::move(last_level));
                multiwindow_l.back().gen_neighlist(multiwindow_l[multiwindow_l.size()-2].layout);

            }

            multiwindow_only(img_ref_l, img_def_l, *interp_ref_l, *interp_def_l,
                            multiwindow_l, conf,
                            img_num_ref_l, img_num_def_l, results_ref_l, results_def_l);


            if (update_ref) results_ref_l = results_def_l;
        }


        // ----------------------------------------------------------------------------------------
        // singlewindow FFTCC + RG
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == "SINGLEWINDOW_RG") {
            if (conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf)) {
                img_num_ref_l = img_num_def_l - 1;
                img_ref_l = read_img(conf.fullpaths[img_num_ref_l]);
                interp_ref_l = make_interp(conf.interp_routine, img_ref_l);
                std::swap(results_ref_l, results_def_l);
                //results_ref_l.get_latest_matches(results_def_l, img_num_def_l);
            }
            singlewindow_rg(img_ref_l, img_def_l, *interp_ref_l, *interp_def_l,
                            ss_grid_l, conf, img_num_ref_l, img_num_def_l,
                            results_ref_l, results_def_l);
        }


        // ----------------------------------------------------------------------------------------
        // multiwindow FFTCC + reliability Guided
        // ----------------------------------------------------------------------------------------
        else if (conf.scan_method == "MULTIWINDOW_RG") {

            bool update_ref = conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf);
            if (update_ref) {
                img_num_ref_l = img_num_def_l - 1;
                img_ref_l = read_img(conf.fullpaths[img_num_ref_l]);
                interp_ref_l = make_interp(conf.interp_routine, img_ref_l);

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
                last_level.mad_filter    = conf.fft_mad;
                last_level.mad_scale     = conf.fft_mad_scale;
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
                }
                last_level.layout = ss_grid_l;
                for (int i = 0; i < ss_grid_l.num; i++) {
                    if (ss_grid_l.active_ss[i]) {
                        last_level.layout.coords[2*i]   += results_ref_l.u[i];
                        last_level.layout.coords[2*i+1] += results_ref_l.v[i];
                    }
                }

                multiwindow_l.push_back(std::move(last_level));
                multiwindow_l.back().gen_neighlist(multiwindow_l[multiwindow_l.size()-2].layout);

            }

            multiwindow_rg(img_ref_l, img_def_l, *interp_ref_l, *interp_def_l,
                        multiwindow_l, conf, img_num_ref_l, img_num_def_l,
                        results_ref_l, results_def_l);


            // ------------------------------------------------------------
            // stereo
            // ------------------------------------------------------------
            if (conf.stereo) {

                if (match_strat != 3) {
                    std::cerr << "UNKNOWN MATCH_STRAT\n";
                    exit(0);
                }

                stereo::matching(img_def_l,
                                img_def_r,
                                *interp_def_l,
                                *interp_def_r,
                                multiwindow_l.back().layout,
                                conf,
                                img_num_def_l,
                                img_num_def_r,
                                stereo_geom.F,
                                results_def_l,
                                results_def_r);

                stereo::pixel_to_world(multiwindow_l.back().layout,
                                    calib,
                                    results_def_l,
                                    results_def_r,
                                    stereo_geom.K0,
                                    stereo_geom.K1,
                                    stereo_geom.R,
                                    conf.ss_size);
            }
            if (update_ref) results_ref_l = results_def_l;

        }
        else {
            throw std::invalid_argument("Unsupported scan method: " + conf.scan_method);
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

    if (conf.incremental_update_cond == "IMAGE") {
        int interval = static_cast<int>(conf.incremental_update_val);
        return (img_num_def_l - 1) % interval == 0;
    }
    if (conf.incremental_update_cond == "ITER") {
        double avg = std::accumulate(results.niter.begin(), results.niter.end(), 0.0) / results.niter.size();
        return avg > conf.incremental_update_val;
    }
    if (conf.incremental_update_cond == "COST") {
        double avg = std::accumulate(results.cost.begin(), results.cost.end(), 0.0) / results.cost.size();
        return avg > conf.incremental_update_val;
    }
    throw std::runtime_error("Unknown incremental update condition: " + conf.incremental_update_cond);
}
