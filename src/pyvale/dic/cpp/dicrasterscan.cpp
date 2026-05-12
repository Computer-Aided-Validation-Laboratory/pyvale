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
#include "./dicrasterscan.hpp"


void raster(const Image &img_ref,
            const Interpolator &interp_def,
            const subset::Grid &ss_grid,
            const util::Config &conf,
            const int img_num_ref,
            const int img_num_def,
            ResultArrays &result_arrays){

    const int num_ss = ss_grid.num;
    const int ss_size_x = ss_grid.size_x;
    const int ss_size_y = ss_grid.size_y;
    const int results_num = img_num_def-1;

    // progress bar
    std::string bar_title = "Temporal matching for \033[1;4m" + conf.basenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.basenames[img_num_def] + "\033[0m:";
    ProgressBar pbar(bar_title, num_ss);
    std::atomic<int> current_progress = 0;
    int prev_pct = 0;

    // loop over subsets within the ROI
    #pragma omp parallel shared(stop_request)
    {

        // initialise subsets
        subset::Pixels ss_def(ss_size_x, ss_size_y);
        subset::Pixels ss_ref(ss_size_x, ss_size_y);

        // optimization parameters
        Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);


        #pragma omp for
        for (int ss = 0; ss < num_ss; ss++){

            // exit the main DIC loop when ctrl+C is hit
            if (stop_request) continue;

            // subset coordinate list takes central locations. 
            // Converting to top left corner for optimization routine
            double cx = ss_grid.coords[ss*2];
            double cy = ss_grid.coords[ss*2+1];
            const int corner_x = int(cx - ss_size_x/2);
            const int corner_y = int(cy - ss_size_y/2);

            // get the reference subset
            subset::fill_from_img(ss_ref, corner_x, corner_y, conf.px_hori, conf.px_vert, img_ref);

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = 0.0;
            }
            
            // if the reference subset is empty, then skip the optimization and set results to nan
            OptResult res(opt.num_params);
            if (ss_ref.sum!=0) res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

            // append the results for the current subset to result vectors
            result_arrays.append(res, ss);

            // update progress bar
            if (g_debug_level>0){
                int progress = current_progress.fetch_add(1);
                if (omp_get_thread_num()==0) pbar.update(progress);
            }

        }
    }
    if (g_debug_level>0){
        int progress = current_progress;
        pbar.finish();
    }
}
