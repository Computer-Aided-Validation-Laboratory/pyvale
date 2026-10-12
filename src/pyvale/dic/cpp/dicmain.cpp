// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <signal.h>
#include <memory>
#include <algorithm>
#include <numeric>
#include <functional>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// commoncpp header files
#include "../../commoncpp/dicsignalhandler.hpp"
#include "../../commoncpp/defines.hpp"
#include "../../commoncpp/util.hpp"

// DIC Header files
#include "./dicinterpfactory.hpp"
#include "./dicutil.hpp"
#include "./dicresults.hpp"
#include "./dicsubset.hpp"
#include "./dicroiupdate.hpp"
#include "./dicmain.hpp"
#include "./dicengine.hpp"
#include "./dicmultiwindow_rg.hpp"
#include "./dicmultiwindow_only.hpp"
#include "./dicsinglewindow_rg.hpp"
#include "./dicrasterscan.hpp"
#include "./dicfilter.hpp"

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

    engine_impl(img_roi_arr, calib, conf, mwconf, saveconf, [&](int image_index) {

        Image image = read_img(conf.fullpaths.at(image_index));
        apply_filter(image, conf.image_filter_kernel, static_cast<float>(conf.image_filter_sigma));
        return make_interp(conf.interp_routine, image);

    });
}

void engine_images(const std::vector<Image>& images,
                   const py::array_t<bool>& img_roi_arr,
                   const Calib &calib,
                   const util::Config &conf,
                   const MultiwindowConfig &mwconf,
                   const common_util::SaveConfig &saveconf){

    const size_t expected_images = conf.stereo
        ? static_cast<size_t>(2 * (conf.num_def_img + 1))
        : static_cast<size_t>(conf.num_def_img + 1);

    if (images.size() != expected_images) {
        throw std::runtime_error("Image array count does not match DIC config.");
    }

    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].width != static_cast<uint32_t>(conf.px_hori) ||
            images[i].height != static_cast<uint32_t>(conf.px_vert)) {
            throw std::runtime_error("Image array dimensions do not match DIC config.");
        }
    }

    engine_impl(img_roi_arr, calib, conf, mwconf, saveconf, [&](int image_index) {

        Image image = images.at(image_index);
        apply_filter(image, conf.image_filter_kernel, static_cast<float>(conf.image_filter_sigma));
        return make_interp(conf.interp_routine, image);

    });
}

void engine_impl(const py::array_t<bool>& img_roi_arr,
                 const Calib &calib,
                 const util::Config &conf,
                 const MultiwindowConfig &mwconf,
                 const common_util::SaveConfig &saveconf,
                 const std::function<std::unique_ptr<Interpolator>(int)> &interp_factory){


    signal(SIGINT, signalHandler);
    g_debug_level = conf.debug_level;

    if (g_debug_level>0){
        common_util::title("Starting Correlation");
    }

    const bool *img_roi = img_roi_arr.data();

    MultiwindowPyramid multiwindow_l;
    SubsetGrid ss_grid_l;
    SubsetGrid ss_grid_l_0;

    ss_grid_l = build_initial_grid(img_roi,
                                   conf,
                                   mwconf,
                                   saveconf,
                                   multiwindow_l);
    ss_grid_l_0 = ss_grid_l;


    // resize the results based on subset information
    ResultArrays results_def_l(ss_grid_l.num, conf.num_params, false);


    std::unique_ptr<Interpolator> interp_ref_l;
    std::unique_ptr<Interpolator> interp_ref_r;
    std::unique_ptr<Interpolator> interp_def_l;
    std::unique_ptr<Interpolator> interp_def_r;
    interp_ref_l = interp_factory(0); // image 0


    ResultArrays results_ref_l(ss_grid_l.num, conf.num_params, false);
    ResultArrays results_ref_r;
    ResultArrays results_def_r;
    stereo::Geometry stereo_geom;

    auto [basenames_l, basenames_r] = stereo::split_basenames(conf);


    if (conf.stereo){

        // resize the results based on subset information
        results_ref_r = ResultArrays(ss_grid_l.num, conf.num_params, true);
        results_def_r = ResultArrays(ss_grid_l.num, conf.num_params, true);

        // sort out intrinsic and extrinsic matrices into struct
        stereo_geom = stereo::compute_stereo_geometry(calib);

        // Reconstruct the supplied stereo reference pair before processing any
        // deformed images. Physical displacements must be measured from this
        // geometry, rather than from the first deformed image pair.
        const int img_num_ref_r = conf.num_def_img + 1;
        interp_ref_r = interp_factory(img_num_ref_r);
        singlewindow_rg(*interp_ref_l,
                        *interp_ref_r,
                        ss_grid_l_0,
                        conf,
                        0,
                        img_num_ref_r,
                        results_ref_l,
                        results_ref_r,
                        "stereo",
                        stereo_geom.F);

        stereo::pixel_to_world(ss_grid_l_0,
                               calib,
                               results_ref_l,
                               results_ref_r,
                               results_ref_r,
                               stereo_geom.K0,
                               stereo_geom.K1,
                               stereo_geom.R,
                               conf.ss_size,
                               true);

        // The temporal part of this output is the supplied reference image
        // compared with itself. It is not a correlation result, but it is a
        // valid zero-displacement datum for the first saved frame.
        std::fill(results_ref_l.conv.begin(), results_ref_l.conv.end(), 1);
        std::fill(results_ref_l.above_thresh.begin(), results_ref_l.above_thresh.end(), 1);
        std::fill(results_ref_l.cost.begin(), results_ref_l.cost.end(), 1.0);

        results_ref_l.write_to_disk_stereo(results_ref_r,
                                           saveconf,
                                           ss_grid_l_0,
                                           basenames_l[0]);
    }


    int img_num_ref_l = 0;

    // -----------------------------------------------------------------------
    // loop over deformed images and perform DIC
    // -----------------------------------------------------------------------
    for (int img_num = 1; img_num < conf.num_def_img+1; img_num++){

        int img_num_def_l = img_num;
        int img_num_def_r = conf.num_def_img+1+img_num;

        // interpolator for the L image
        interp_def_l = interp_factory(img_num_def_l);

        // interpolator for the R image
        if (conf.stereo) {
            interp_def_r = interp_factory(img_num_def_r);
        }

        switch (conf.scan_method) {
            case util::EScanMethod::RASTER: {
                // ----------------------------------------------------------------------------------------
                // raster scan
                // ----------------------------------------------------------------------------------------
                if (conf.stereo) 
                    throw std::invalid_argument("Unsupported scan method");

                if (conf.incremental)
                    throw std::invalid_argument("Raster scan does not support incremental DIC");

                results_def_l.reset();
                results_def_r.reset();

                raster(*interp_ref_l,
                       *interp_def_l,
                       ss_grid_l,
                       conf, 0,
                       img_num,
                       results_def_l);
                break;
            }

            case util::EScanMethod::MULTIWINDOW: {
                // ----------------------------------------------------------------------------------------
                // multiwindow FFTCC
                // ----------------------------------------------------------------------------------------
                bool update_ref = conf.incremental && should_update_ref(img_num_def_l,
                                                                        results_def_l,
                                                                        conf);
                if (update_ref) {
                    img_num_ref_l = img_num_def_l - 1;
                    results_ref_l = results_def_l;
                    interp_ref_l = interp_factory(img_num_ref_l);

                    std::unique_ptr<bool[]> roi_updated(propagate_roi(img_roi,
                                                                      results_def_l,
                                                                      conf,
                                                                      ss_grid_l));

                    // Update the active finest-level grid before rebuilding the pyramid.
                    if (img_num_def_l > 1){
                        for (int i = 0; i < ss_grid_l.num; i++) {
                            if (!results_ref_l.above_thresh[i]) {
                                ss_grid_l.active_ss[i] = false;
                            }
                        }
                        ss_grid_l.active_total = std::count(ss_grid_l.active_ss.begin(),
                                                            ss_grid_l.active_ss.end(), true);
                    }

                    multiwindow_l.rebuild_for_new_reference(roi_updated.get(),
                                                            ss_grid_l,
                                                            results_ref_l.u,
                                                            results_ref_l.v,
                                                            conf,
                                                            mwconf,
                                                            saveconf,
                                                            mwconf.overlap.size() - 1);

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
                break;
            }

            case util::EScanMethod::SINGLEWINDOW_RG: {
                // ----------------------------------------------------------------------------------------
                // singlewindow FFTCC + RG
                // ----------------------------------------------------------------------------------------
                if (conf.incremental && should_update_ref(img_num_def_l, results_def_l, conf)) {

                    // update left image vars
                    img_num_ref_l = img_num_def_l - 1;
                    results_ref_l = results_def_l;
                    interp_ref_l = interp_factory(img_num_ref_l);

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

                    for (int i = 0; i < ss_grid_l.num; i++) {
                        if (ss_grid_l.active_ss[i]) {
                            ss_grid_l.coords[2*i]   += results_ref_l.u[i];
                            ss_grid_l.coords[2*i+1] += results_ref_l.v[i];
                        }
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

                    singlewindow_rg(*interp_ref_l,
                                    *interp_def_r,
                                    ss_grid_l,
                                    conf,
                                    img_num_ref_l,
                                    img_num_def_r,
                                    results_ref_l,
                                    results_def_r,
                                    "stereo",
                                    stereo_geom.F,
                                    results_def_l);

                    stereo::pixel_to_world(ss_grid_l_0,
                                        calib,
                                        results_def_l,
                                        results_ref_r,
                                        results_def_r,
                                        stereo_geom.K0,
                                        stereo_geom.K1,
                                        stereo_geom.R,
                                        conf.ss_size,
                                        false);
                }
                break;
            }

            case util::EScanMethod::MULTIWINDOW_RG: {
                // ----------------------------------------------------------------------------------------
                // multiwindow FFTCC + reliability Guided
                // ----------------------------------------------------------------------------------------
                bool update_ref = conf.incremental && should_update_ref(img_num_def_l,
                                                                        results_def_l,
                                                                        conf);
                if (update_ref) {

                    // update left image vars
                    img_num_ref_l = img_num_def_l - 1;
                    results_ref_l = results_def_l;
                    interp_ref_l = interp_factory(img_num_ref_l);

                    // update right image vars
                    if (conf.stereo)
                        results_ref_r = results_def_r;


                    std::unique_ptr<bool[]> roi_updated(propagate_roi(img_roi,
                                                                      results_def_l,
                                                                      conf,
                                                                      ss_grid_l));

                    // Update the active finest-level grid before rebuilding the pyramid.
                    if (img_num_def_l > 1){
                        for (int i = 0; i < ss_grid_l.num; i++) {
                            if (!results_ref_l.above_thresh[i]) {
                                ss_grid_l.active_ss[i] = false;
                            }
                        }
                        ss_grid_l.active_total = std::count(ss_grid_l.active_ss.begin(),
                                                            ss_grid_l.active_ss.end(), true);
                    }

                    multiwindow_l.rebuild_for_new_reference(roi_updated.get(),
                                                            ss_grid_l,
                                                            results_ref_l.u,
                                                            results_ref_l.v,
                                                            conf,
                                                            mwconf,
                                                            saveconf,
                                                            mwconf.overlap.size() - 1);

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

                    singlewindow_rg(*interp_ref_l,
                                    *interp_def_r,
                                    multiwindow_l.back().grid,
                                    conf,
                                    img_num_ref_l,
                                    img_num_def_r,
                                    results_ref_l,
                                    results_def_r,
                                    "stereo",
                                    stereo_geom.F,
                                    results_def_l);

                    stereo::pixel_to_world(ss_grid_l_0,
                                        calib,
                                        results_def_l,
                                        results_ref_r,
                                        results_def_r,
                                        stereo_geom.K0,
                                        stereo_geom.K1,
                                        stereo_geom.R,
                                        conf.ss_size,
                                        false);
                }
                break;
            }

            default:
                throw std::invalid_argument("Unsupported scan method");
        }

        if (!conf.stereo){
            results_def_l.write_to_disk_2d(saveconf,
                                           ss_grid_l,
                                           basenames_l[img_num]);
        }
        else {
            results_def_l.write_to_disk_stereo(results_def_r,
                                               saveconf,
                                               ss_grid_l,
                                               basenames_l[img_num]);
        }

        if (stop_request) break;
    }

    raise_on_interrupt();
}
