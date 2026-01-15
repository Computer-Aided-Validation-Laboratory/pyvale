// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include "dicscanmethod.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <atomic>
#include <thread>
#include <cstring>
#include <omp.h>
#include <csignal>

// pybind headers
#include <pybind11/pybind11.h>

// common_cpp headers
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"


// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicutil.hpp"
#include "./dicrg.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"

namespace scanmethod {


    void image(const double *img_ref,
               const Interpolator &interp_def,
               const subset::Grid &ss_grid,
               const util::Config &conf,
               const int img_num,
               OptResultArrays &result_arrays){

        const int num_ss = ss_grid.num;
        const int ss_size = ss_grid.size;
        const int results_num = img_num-1;

        // progress bar
        std::string bar_title = "Correlation for " + conf.filenames[img_num] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress = 0;
        int prev_pct = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            subset::Pixels ss_def(ss_size);
            subset::Pixels ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter,
                                    conf.precision, conf.threshold,
                                    conf.px_vert, conf.px_hori,
                                    conf.corr_crit);

            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request) continue;

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ss_grid.coords[ss*2];
                int ss_y = ss_grid.coords[ss*2+1];

                // get the reference subset
                subset::get_px_from_img(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                for (int i = 0; i < opt.num_params; i++){
                    opt.p[i] = 0.0;
                }

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ss_grid.size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ss_grid.size)/2.0 - 0.5;
                OptResult res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                // append the results for the current subset to result vectors
                result_arrays.append(res, results_num, ss);

                // update progress bar
                int progress = current_progress.fetch_add(1);
                if (omp_get_thread_num()==0) pbar.update(progress);

            }
        }
        int progress = current_progress;
        pbar.finish();
    }

    void multiwindow_reliability_guided(const double *img_ref,
                                       const double *img_def,
                                       const Interpolator &interp_def,
                                       const std::vector<subset::Grid> &ss_grid,
                                       const util::Config &conf,
                                       const int img_num,
                                       OptResultArrays &result_arrays){

        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int seed_x = conf.rg_seed.first;
        const int seed_y = conf.rg_seed.second;
        const int nsizes = ss_grid.size();
        const int last_size = nsizes-1;
        const int num_ss = ss_grid[last_size].num;
        const int ss_size = ss_grid[last_size].size;
        const int ss_step = ss_grid[last_size].step;
        const int results_num = img_num-1;

        fourier::multiwindow(ss_grid, img_ref, img_def, interp_def, conf.fft_mad, conf.fft_mad_scale);

        // progress bar
        std::string bar_title = "Correlation for " + conf.filenames[img_num] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ss_grid[last_size])) {
            return;
        }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid[last_size].mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        std::vector<std::priority_queue<rg::Point>> local_q(omp_get_max_threads());

        // Mutex vector to protect each queue
        std::vector<std::mutex> queue_mutexes(omp_get_max_threads());
        std::mutex steal_mutex;

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::priority_queue<rg::Point>& thread_q = local_q[tid];

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size);
            subset::Pixels ss_ref(ss_size);

            // Optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                      conf.precision, conf.threshold, 
                                      px_vert, px_hori,
                                      conf.corr_crit);

            std::vector<std::unique_ptr<fourier::FFT>> fft_windows;

            for (size_t t = 0; t < ss_grid.size(); ++t) {
                fft_windows.push_back(std::make_unique<fourier::FFT>(ss_grid[t].size));
            }

            // TODO: for the seed location I'm going to overwride the max 
            // number of iterations to make sure we get a good convergence.
            // this is hardcoded for now. Could do with updating so that 
            // the seed location is checked ahead of the main correlation run.

            // TODO: opt.seed_iter exposed to user.
            opt.max_iter = 200;

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int x = seed_x / ss_step;
                int y = seed_y / ss_step;
                int idx = ss_grid[last_size].mask[y * ss_grid[last_size].num_ss_x + x];


                // if the first image. Take the optimization parameters from rigid fourier
                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                opt.p[0] = fourier::shifts[last_size].x[idx];
                opt.p[1] = fourier::shifts[last_size].y[idx];

                // Extract reference subset and solve for starting seed point
                subset::get_px_from_img(ss_ref, seed_x, seed_y, px_hori, px_vert, img_ref);


                double centre_x = seed_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = seed_y + static_cast<double>(ss_size)/2.0 - 0.5;

                OptResult seed_res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                // append the results for the current subset to result vectors
                result_arrays.append(seed_res, results_num, idx);

                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid[last_size].neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid[last_size].neigh[idx][n];

                    int nx = ss_grid[last_size].coords[nidx*2];
                    int ny = ss_grid[last_size].coords[nidx*2+1];

                    subset::get_px_from_img(ss_ref, nx, ny, px_hori, px_vert, img_ref);

                    // get parameter values from fft output or from previous image
                    std::fill(opt.p.begin(), opt.p.end(), 0.0);
                    opt.p[0] = fourier::shifts[last_size].x[nidx];
                    opt.p[1] = fourier::shifts[last_size].y[nidx];

                    // perform optimization for seed point neighbours
                    double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                    double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;
                    OptResult nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                    // append the results for the current subset to result vectors
                    result_arrays.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // Add points to queue
                    local_q[0].push(rg::Point(nidx,nres.cost));

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        pbar.update(progress);
                    }
                }
            }


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier

            // TODO: reset seed location using the last computed point
            opt.max_iter = conf.max_iter;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {
                bool got_point = false;
                int idle_iters = 0;

                // Try own queue safely
                {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    if (!thread_q.empty()) {
                        current = thread_q.top();
                        thread_q.pop();
                        got_point = true;
                    }
                }

                // Steal if nothing in own queue
                if (!got_point) {
                    while (!got_point && idle_iters < max_idle_iters) {
                        {
                            std::lock_guard<std::mutex> lock(steal_mutex);
                            for (size_t i = 0; i < local_q.size(); ++i) {
                                std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                if (!local_q[i].empty()) {
                                    current = local_q[i].top();
                                    local_q[i].pop();
                                    got_point = true;
                                    break;
                                }
                            }
                        }
                        if (!got_point) {
                            ++idle_iters;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                }

                if (!got_point) {
                    break;
                }

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_results = result_arrays.index(current.idx, results_num);
                int idx_results_p = result_arrays.index_parameters(current.idx, results_num);

                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid[last_size].neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid[last_size].neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        int nx = ss_grid[last_size].coords[nidx*2];
                        int ny = ss_grid[last_size].coords[nidx*2+1];

                        // extract subset
                        subset::get_px_from_img(ss_ref, nx, ny, px_hori, px_vert, img_ref);

                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (result_arrays.cost[idx_results] < conf.threshold){
                            std::fill(opt.p.begin(), opt.p.end(), 0.0);
                            opt.p[0] = fourier::shifts[last_size].x[nidx];
                            opt.p[1] = fourier::shifts[last_size].y[nidx];
                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = result_arrays.p[idx_results_p+i];
                            }
                        }

                        // optimize
                        double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                        double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;
                        OptResult nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                        // append results
                        result_arrays.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (omp_get_thread_num()==0) pbar.update(progress);
                        }
                    }
                }

                for (const auto& neigh : temp_neigh) {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    thread_q.push(neigh);
                }
            }
        }
        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }

    void singlewindow_incremental_reliability_guided(const double *img_ref,
                                                   const double *img_def,
                                                   const Interpolator &interp_ref,
                                                   const Interpolator &interp_def,
                                                   const std::vector<subset::Grid> &ss_grid,
                                                   const util::Config &conf,
                                                   const int img_num_ref,
                                                   const int img_num_def,
                                                   OptResultArrays &result_arrays){


        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        int seed_x = conf.rg_seed.first;
        int seed_y = conf.rg_seed.second;
        const int nsizes = ss_grid.size();
        const int last_size = nsizes-1;
        const int num_ss = ss_grid[last_size].num;
        const int ss_size = ss_grid[last_size].size;
        const int ss_step = ss_grid[last_size].step;
        const int results_num = img_num_def-1;

        // get start location of displacements in previous image
        double *prev_img_u = result_arrays.u.data() + result_arrays.index(0,std::max(0,img_num_ref-1));
        double *prev_img_v = result_arrays.v.data() + result_arrays.index(0,std::max(0,img_num_ref-1));

        // get rigid shifts from fourier
        // fourier::single_grid(ss_grid[last_size], prev_img_u, prev_img_v,
        //                      conf.max_disp, img_ref, img_def, interp_def);

        std::string bar_title = "Correlation for " + conf.filenames[img_num_def] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // quick check for the initial seed point
        // if (!rg::is_valid_point(seed_x, seed_y, ss_grid[last_size])) {
        //     return;
        // }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid[last_size].mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        std::vector<std::priority_queue<rg::Point>> local_q(omp_get_max_threads());

        // Mutex vector to protect each queue
        std::vector<std::mutex> queue_mutexes(omp_get_max_threads());
        std::mutex steal_mutex;

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::priority_queue<rg::Point>& thread_q = local_q[tid];

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size);
            subset::Pixels ss_ref(ss_size);

            // Optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                      conf.precision, conf.threshold, 
                                      px_vert, px_hori,
                                      conf.corr_crit);

            std::vector<std::unique_ptr<fourier::FFT>> fft_windows;

            for (size_t t = 0; t < ss_grid.size(); ++t) {
                fft_windows.push_back(std::make_unique<fourier::FFT>(ss_grid[t].size));
            }

            // TODO: opt.seed_iter exposed to user.
            opt.max_iter = 200;

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int x = seed_x / ss_step;
                int y = seed_y / ss_step;
                int idx = ss_grid[last_size].mask[y * ss_grid[last_size].num_ss_x + x];


                // need to add offset based on displacements from previous correlation
                double seed_x_new, seed_y_new;
                if (img_num_ref == 0) {
                    seed_x_new = seed_x;
                    seed_y_new = seed_y;
                } else {
                    seed_x_new = seed_x + prev_img_u[idx];
                    seed_y_new = seed_y + prev_img_v[idx];
                }

                // reference subset based on results from previous image
                subset::get_subpx_from_img(ss_ref, seed_x_new, seed_y_new, interp_ref);


                // if the first image. Take the optimization parameters from rigid fourier
                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                fourier::get_single_window_fftcc_peak(opt.p[0], opt.p[1],
                                                      seed_x_new, seed_y_new,
                                                      ss_size, std::max(conf.max_disp, conf.ss_size),
                                                      img_ref, img_def,
                                                      interp_def);


                //std::cout << "PEAK " << opt.p[0] << " " << opt.p[1] << std::endl;
                double centre_x = seed_x_new + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = seed_y_new + static_cast<double>(ss_size)/2.0 - 0.5;
                
                OptResult seed_res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                if (!seed_res.converged || !seed_res.above_threshold){
                    std::cout << "ERROR: unsuccesful convergence at seed location." << std::endl;
                    std::cout << "Please select a different seed location." << std::endl;
                    exit(EXIT_FAILURE);
                }

                // add deformation from reference image to new results
                if (img_num_ref > 0){
                    seed_res.u += prev_img_u[idx];
                    seed_res.v += prev_img_v[idx];
                }

                // append the results for the current subset to result vectors
                result_arrays.append(seed_res, results_num, idx);

                // mark subset as computed
                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid[last_size].neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid[last_size].neigh[idx][n];

                    double nx = ss_grid[last_size].coords[nidx*2];
                    double ny = ss_grid[last_size].coords[nidx*2+1];
                        

                    // need to add displacements from previous image
                    if (img_num_ref > 0){
                        nx += prev_img_u[nidx];
                        ny += prev_img_v[nidx];
                    }

                    subset::get_subpx_from_img(ss_ref, nx, ny, interp_ref);

                    // get initial guess at parameter values from seed point
                    int index_p = result_arrays.index_parameters(idx,results_num);
                    for (int i = 0; i < opt.num_params; i++){
                        opt.p[i] = result_arrays.p[index_p+i];
                    }

                    // perform optimization for seed point neighbours
                    double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                    double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;
                    OptResult nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                    // add deformation from reference image to new results
                    if (img_num_ref > 0){
                        nres.u += prev_img_u[nidx];
                        nres.v += prev_img_v[nidx];
                    }

                    if (!nres.converged || !nres.above_threshold){
                        std::cout << "ERROR: unsuccesful convergence at neighbouring point to seed." << std::endl;
                        std::cout << "Please select a different seed location." << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    // append the results for the current subset to result vectors
                    result_arrays.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // add this point to queue
                    local_q[0].push(rg::Point(nidx,nres.cost));

                    // update progress bar
                    int progress = current_progress.fetch_add(1);
                    pbar.update(progress+1);
                }
            }


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier

            // TODO: reset seed location using the last computed point
            opt.max_iter = conf.max_iter;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {
                bool got_point = false;
                int idle_iters = 0;

                // Try own queue safely
                {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    if (!thread_q.empty()) {
                        current = thread_q.top();
                        thread_q.pop();
                        got_point = true;
                    }
                }

                // Steal if nothing in own queue
                if (!got_point) {
                    while (!got_point && idle_iters < max_idle_iters) {
                        {
                            std::lock_guard<std::mutex> lock(steal_mutex);
                            for (size_t i = 0; i < local_q.size(); ++i) {
                                std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                if (!local_q[i].empty()) {
                                    current = local_q[i].top();
                                    local_q[i].pop();
                                    got_point = true;
                                    break;
                                }
                            }
                        }
                        if (!got_point) {
                            ++idle_iters;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                }

                if (!got_point) {
                    break;
                }

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_results_def = result_arrays.index(current.idx, results_num);
                int idx_results_def_p = result_arrays.index_parameters(current.idx, results_num);


                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid[last_size].neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid[last_size].neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        double nx = ss_grid[last_size].coords[nidx*2];
                        double ny = ss_grid[last_size].coords[nidx*2+1];

                        // add displacements from reference image
                        if (img_num_ref > 0){
                            nx += prev_img_u[nidx];
                            ny += prev_img_v[nidx];
                        }

                        // temporarily fill p with results from prev img to get
                        int idx_results_p_ref = result_arrays.index_parameters(nidx, img_num_ref);
                        for (int i = 0; i < opt.num_params; i++) opt.p[i] = result_arrays.p[idx_results_p_ref+i];

                        subset::get_subpx_from_shape_params(ss_ref, nx, ny, opt.p, interp_ref);


                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (result_arrays.cost[idx_results_def] < conf.threshold){
                            std::fill(opt.p.begin(), opt.p.end(), 0.0);
                            fourier::get_single_window_fftcc_peak(opt.p[0], opt.p[1],
                                                                  nx, ny,
                                                                  ss_size, std::max(conf.max_disp, conf.ss_size),
                                                                  img_ref, img_def,
                                                                  interp_def);
                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = result_arrays.p[idx_results_def_p+i];
                            }
                        }

                        // optimize
                        double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                        double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;

                        OptResult nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                        // add deformation from reference image to new results
                        if ((nres.converged) && (nres.above_threshold) && (img_num_ref > 0)){
                            nres.u += prev_img_u[nidx];
                            nres.v += prev_img_v[nidx];
                        }
                        else if (img_num_ref > 0){
                            nres.u = prev_img_u[nidx];
                            nres.v = prev_img_v[nidx];
                        }

                        // append results
                        result_arrays.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        int progress = current_progress.fetch_add(1);
                        if (tid==0) pbar.update(progress+1);
                    }
                }

                for (const auto& neigh : temp_neigh) {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    thread_q.push(neigh);
                }
            }
        }
        pbar.update(current_progress+1);
        pbar.finish();
    }

    void multiwindow(const double *img_ref,
                              const double *img_def,
                              const Interpolator &interp_def,
                              const std::vector<subset::Grid> &ss_grid,
                              const util::Config &conf,
                              const int img_num,
                              OptResultArrays &result_arrays){

        // for the first image perform the FFT windowing. later images will be
        // seeded with previous images
        fourier::multiwindow(ss_grid, img_ref, img_def, interp_def, conf.fft_mad, conf.fft_mad_scale);

        const int nsizes = ss_grid.size();
        const int last_size = nsizes-1;
        const int results_num = img_num-1;

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = ss_grid[last_size].num;
        const int ss_size = ss_grid[last_size].size;

        for (int ss = 0; ss < num_ss; ss++){

            // exit the main DIC loop when ctrl+C is hit
            if (stop_request){
                continue;
            }

            // append fourier results to master result vectors
            OptResult res(conf.num_params);
            res.u    = fourier::shifts[last_size].x[ss];
            res.p[0] = fourier::shifts[last_size].x[ss];
            res.v    = fourier::shifts[last_size].y[ss];
            res.p[1] = fourier::shifts[last_size].y[ss];
            res.converged=true;
            res.above_threshold=true;
            result_arrays.append(res, results_num, ss);
        }
    }


    void single_window_fourier(const double *img_ref,
                               const double *img_def,
                               const Interpolator &interp_def,
                               const subset::Grid &ss_grid,
                               const util::Config &conf,
                               const int img_num,
                               OptResultArrays &result_arrays){

        // for the first image perform the FFT windowing. later images will be
        // seeded with previous images
        //fourier::single_grid(ss_grid, 256, img_ref, img_def, interp_def);

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = ss_grid.num;
        const int ss_size = ss_grid.size;
        const int results_num = img_num-1;

        // progress bar
        std::string bar_title = "Correlation for " + conf.filenames[img_num] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            subset::Pixels ss_def(ss_size);
            subset::Pixels ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                    conf.precision, conf.threshold,
                                    conf.px_vert, conf.px_hori,
                                    conf.corr_crit);


            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request){
                    continue;
                }

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ss_grid.coords[ss*2];
                int ss_y = ss_grid.coords[ss*2+1];

                // get the reference subset
                subset::get_px_from_img(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                opt.p[0] = fourier::shifts[0].x[ss];
                opt.p[1] = fourier::shifts[0].y[ss];

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ss_size)/2.0 - 0.5;
                OptResult res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                // append optimization results to results vectors
                result_arrays.append(res, img_num, ss);

                // update progress bar
                int progress = current_progress.fetch_add(1);
                if (omp_get_thread_num()==0) pbar.update(progress);

            }
        }
        pbar.update(current_progress+1);
        pbar.finish();
    }

}
