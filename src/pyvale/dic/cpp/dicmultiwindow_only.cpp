// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <omp.h>
#include <csignal>
#include <optional>
#include <chrono>

// common_cpp headers
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"


// Program Header files
#include "./dicinterp.hpp"
#include "./dicoptimizer.hpp"
#include "./dicutil.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow_util.hpp"


void multiwindow_only(const Image &img_ref,
                        const Image &img_def,
                        const Interpolator &interp_ref,
                        const Interpolator &interp_def,
                        std::vector<WindowLevel> &multiwindow,
                        const util::Config &conf,
                        const int img_num_ref,
                        const int img_num_def,
                        const ResultArrays &results_ref,
                        ResultArrays &results_def){

    // loop over the window sizes and calculate estimates for rigid
    // displacement using FFTCC
    for (int lvl = 0; lvl < multiwindow.size(); lvl++){
        multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
                                                    interp_ref,
                                                    interp_def,
                                                    img_num_ref, img_num_def,
                                                    lvl, multiwindow.size(),
                                                    conf.basenames);
    }

    const subset::Grid &ss_grid = multiwindow.back().layout;
    const int nsizes = multiwindow.size();
    const int last_size = nsizes-1;

    #pragma omp parallel shared(stop_request, results_ref, results_def, multiwindow, ss_grid, conf, interp_ref, interp_def)
    {

        subset::Pixels ss_ref(ss_grid.size_x, ss_grid.size_y);
        subset::Pixels ss_def(ss_grid.size_x, ss_grid.size_y);

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = multiwindow[last_size].layout.num;



        #pragma omp for
        for (int ss = 0; ss < num_ss; ss++){

            // exit the main DIC loop when ctrl+C is hit
            if (stop_request){
                continue;
            }

            // append fourier results to master result vectors
            OptResult res(conf.num_params);
            res.u    = multiwindow.back().u[ss];
            res.p[0] = multiwindow.back().u[ss];
            res.v    = multiwindow.back().v[ss];
            res.p[1] = multiwindow.back().v[ss];

            // get the reference subset
            const double cx_img0 = ss_grid.coords[ss*2];
            const double cy_img0 = ss_grid.coords[ss*2+1];

            // get the reference subset
            subset::fill_from_centre_coords(ss_ref, cx_img0, cy_img0, interp_ref);
            subset::fill_from_shape_params(ss_def, cx_img0, cy_img0, res.p, interp_def, "RIGID");

            // calculate zncc value
            double zncc = 0.0;
            double mean_def = 0.0;
            double mean_ref = 0.0;

            for (int i = 0; i < ss_def.num_px; ++i) {
                mean_ref += ss_ref.vals[i];
                mean_def += ss_def.vals[i];
            }

            mean_ref /= ss_ref.num_px;
            mean_def /= ss_def.num_px;

            double sum_squared_ref = 0.0;
            double sum_squared_def = 0.0;
            for (int i = 0; i < ss_def.num_px; ++i) {
                sum_squared_ref += (ss_ref.vals[i] - mean_ref) * (ss_ref.vals[i] - mean_ref);
                sum_squared_def += (ss_def.vals[i] - mean_def) * (ss_def.vals[i] - mean_def);
            }

            // Bail out if either subset is degenerate (uniform/zero intensity)
            if (sum_squared_def < 1e-12 || sum_squared_ref < 1e-12) {
                zncc = 0.0;
            }
            else {
                const double inv_sum_squared = 1.0 / sqrt(sum_squared_ref*sum_squared_def);

                for (int i = 0; i < ss_def.num_px; ++i) {
                    const double def_norm = (ss_def.vals[i] - mean_def);
                    const double ref_norm = (ss_ref.vals[i] - mean_ref);
                    zncc += ref_norm*def_norm; 
                }
                zncc *= inv_sum_squared;
            }

            res.cost = zncc;
            res.converged=true;

            if (zncc>=conf.threshold)
                res.above_thresh=true;

            res.u += results_ref.u[ss];
            res.v += results_ref.v[ss];

            results_def.append(res, ss);
        }
    }
}
