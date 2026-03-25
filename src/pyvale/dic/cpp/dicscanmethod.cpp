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
#include <mutex>
#include <queue>
#include <atomic>
#include <thread>
#include <cstring>
#include <omp.h>
#include <csignal>

// common_cpp headers
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"


// Program Header files
#include "./dicinterp.hpp"
#include "./dicoptimizer.hpp"
#include "./dicutil.hpp"
#include "./dicrg.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow.hpp"

namespace scanmethod {


    void image(const double *img_ref,
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
        std::string bar_title = "Temporal matching for \033[1;4m" + conf.filenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress = 0;
        int prev_pct = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            subset::Pixels ss_def(ss_size_x, ss_size_x);
            subset::Pixels ss_ref(ss_size_y, ss_size_y);

            // optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);


            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request) continue;

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ss_grid.coords[ss*2];
                int ss_y = ss_grid.coords[ss*2+1];

                // get the reference subset
                subset::fill_from_img(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                for (int i = 0; i < opt.num_params; i++){
                    opt.p[i] = 0.0;
                }

                // perform optimization on subset from deformed image
                OptResult res = opt.solve(ss_x, ss_y, ss_ref, ss_def, interp_def);

                // append the results for the current subset to result vectors
                result_arrays.append(res, results_num, ss);

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

void multiwindow_reliability_guided(const double *img_ref,
                                    const double *img_def,
                                    const Interpolator &interp_def,
                                    std::vector<WindowLevel> &multiwindow,
                                    const util::Config &conf,
                                    const int img_num_ref,
                                    const int img_num_def,
                                    ResultArrays &result_arrays){

        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int seed_x = conf.rg_seed.first;
        const int seed_y = conf.rg_seed.second;

        // subset information
        const subset::Grid &ss_grid = multiwindow.back().layout;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;
        const int results_num = img_num_def-1;

        // loop over the window sizes and calculate estimates for rigid
        // displacement using FFTCC
        for (int lvl = 0; lvl < multiwindow.size(); lvl++){
            multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
                                                      img_ref, img_def,
                                                      interp_def,
                                                      img_num_ref, img_num_def,
                                                      conf.filenames);
        }

        // progress bar
        std::string bar_title = "Temporal matching for \033[1;4m" + conf.filenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ss_grid)) {
            return;
        }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        rg::QueueLocal queue(omp_get_max_threads());

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size_x, ss_size_x);
            subset::Pixels ss_ref(ss_size_y, ss_size_y);

            // Optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);

            std::vector<std::unique_ptr<FFT>> fft_windows;

            for (size_t lvl = 0; lvl < multiwindow.size(); lvl++) {
                fft_windows.push_back(std::make_unique<FFT>(multiwindow[lvl].layout.size_x, 
                                                            multiwindow[lvl].layout.size_y));
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
                int grid_x = seed_x / ss_step;
                int grid_y = seed_y / ss_step;
                int idx = ss_grid.mask[grid_y * ss_grid.num_ss_x + grid_x];

                double cx = ss_grid.coords[2*idx];
                double cy = ss_grid.coords[2*idx+1];

                // if the first image. Take the optimization parameters from rigid fourier
                opt.copy_params_from_fft(idx, multiwindow.back().u, multiwindow.back().v);

                // Extract reference subset and solve for starting seed point
                subset::fill_from_img(ss_ref, seed_x, seed_y, px_hori, px_vert, img_ref);

                OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                // for (int i=0; i < ss_ref.num_px; i++){
                //     std::cout << "temporal_L " << ss_ref.x[i] << " " << ss_ref.y[i] << " " << ss_ref.vals[i] << " ";
                //     std::cout << ss_def.x[i] << " " << ss_def.y[i] << " " << ss_def.vals[i] << std::endl;
                //
                // }
                // std::cout << std::endl;



                rg::check_convergence_or_exit(seed_x, seed_y, seed_res);


                // append the results for the current subset to result vectors
                result_arrays.append(seed_res, results_num, idx);

                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    const double cx = ss_grid.coords[nidx*2];
                    const double cy = ss_grid.coords[nidx*2+1];

                    const int corner_x = int(cx - ss_size_x/2);
                    const int corner_y = int(cy - ss_size_y/2);

                    subset::fill_from_img(ss_ref, corner_x, corner_y, px_hori, px_vert, img_ref);

                    // get parameter values from fft output or from previous image
                    opt.copy_params_from_fft(nidx, multiwindow.back().u, multiwindow.back().v);

                    // perform optimization for seed point neighbours
                    OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                    rg::check_convergence_or_exit(cx, cy, nres);

                    // append the results for the current subset to result vectors
                    result_arrays.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // Add points to queue
                    queue.push(0, {rg::Point(nidx,nres.cost)});

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        if (omp_get_thread_num()==0) pbar.update(progress);
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

            rg::Point current(0, 0);

            while (!stop_request) {

                if (!queue.pop(tid, current))
                    break;

                temp_neigh.clear();

                // index of current point in results arrays
                int idx_results = result_arrays.index(current.idx, results_num);
                int idx_results_p = result_arrays.index_parameters(current.idx, results_num);

                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        int cx = ss_grid.coords[nidx*2];
                        int cy = ss_grid.coords[nidx*2+1];

                        const int corner_x = int(cx - ss_size_x/2);
                        const int corner_y = int(cy - ss_size_y/2);

                        // extract subset
                        subset::fill_from_img(ss_ref, corner_x, corner_y, px_hori, px_vert, img_ref);

                        if (result_arrays.cost[idx_results] < conf.threshold)
                            opt.copy_params_from_fft(nidx,
                                                     multiwindow.back().u,
                                                     multiwindow.back().v);
                        else 
                            opt.copy_params_from_neigh(result_arrays.p, idx_results_p);

                        // optimize
                        OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

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
                queue.push(tid, temp_neigh);
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
                                                   const subset::Grid &ss_grid,
                                                   const util::Config &conf,
                                                   const int img_num_ref,
                                                   const int img_num_def,
                                                   ResultArrays &result_arrays){


        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        int seed_x = conf.rg_seed.first;
        int seed_y = conf.rg_seed.second;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;
        const int results_num = img_num_def-1;

        // get start location of displacements in previous image
        double *prev_img_u = result_arrays.u.data() + result_arrays.index(0,std::max(0,img_num_ref-1));
        double *prev_img_v = result_arrays.v.data() + result_arrays.index(0,std::max(0,img_num_ref-1));

        std::string bar_title = "Temporal matching for \033[1;4m" + conf.filenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // quick check for the initial seed point
        // if (!rg::is_valid_point(seed_x, seed_y, ss_grid)) {
        //     return;
        // }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        rg::QueueLocal queue(omp_get_max_threads()); 

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size_x, ss_size_y);
            subset::Pixels ss_ref(ss_size_x, ss_size_y);

            // Optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);

            // TODO: opt.seed_iter exposed to user.
            opt.max_iter = 200;

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int grid_x = seed_x / ss_step;
                int grid_y = seed_y / ss_step;
                int idx = ss_grid.mask[grid_y * ss_grid.num_ss_x + grid_x];

                double cx = ss_grid.coords[2*idx];
                double cy = ss_grid.coords[2*idx+1];

                // get the reference subset based on the results from the previous image
                int idx_results_prev_p = result_arrays.index_parameters(idx, img_num_ref-1);
                opt.copy_params_from_neigh(result_arrays.p, idx_results_prev_p);

                subset::fill_from_shape_params(ss_ref, cx, cy, opt.p, interp_ref, conf.shape_func);


                // if the first image. Take the optimization parameters from rigid fourier
                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                int window_size = std::max(conf.max_disp, conf.ss_size);
                get_single_window_fftcc_peak(opt.p[0], opt.p[1],
                                             cx, cy,
                                             ss_size_x, ss_size_y,
                                             window_size, window_size,
                                             img_ref, img_def,
                                             interp_def);


                OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                //rg::check_convergence_or_exit(cx, cy, seed_res);

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
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    double cx = ss_grid.coords[nidx*2];
                    double cy = ss_grid.coords[nidx*2+1];

                    // get initial guess at parameter values from seed point
                    int index_p = result_arrays.index_parameters(idx,results_num);
                    opt.copy_params_from_neigh(result_arrays.p, index_p);

                    if (img_num_ref>0){
                        cx += prev_img_u[nidx];
                        cy += prev_img_v[nidx];
                    }

                    subset::fill_from_shape_params(ss_ref, cx, cy, opt.p, interp_ref, conf.shape_func);

                    // perform optimization for seed point neighbours
                    OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                    // add deformation from reference image to new results
                    if (img_num_ref > 0){
                        nres.u += prev_img_u[nidx];
                        nres.v += prev_img_v[nidx];
                    }

                    rg::check_convergence_or_exit(cx, cy, nres);


                    // append the results for the current subset to result vectors
                    result_arrays.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // add this point to queue
                    queue.push(tid, {rg::Point(nidx,nres.cost)});

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        if (omp_get_thread_num()==0) pbar.update(progress+1);
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

            rg::Point current(0, 0);

            while (!stop_request) {

                if (!queue.pop(tid, current))
                    break;

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_results_def = result_arrays.index(current.idx, results_num);
                int idx_results_def_p = result_arrays.index_parameters(current.idx, results_num);


                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        double cx = ss_grid.coords[nidx*2];
                        double cy = ss_grid.coords[nidx*2+1];

                        // add displacements from reference image
                        if (img_num_ref > 0){
                            cx += prev_img_u[nidx];
                            cy += prev_img_v[nidx];
                        }

                        // temporarily fill p with results from prev img to get
                        int idx_results_p_ref = result_arrays.index_parameters(nidx, img_num_ref);

                        opt.copy_params_from_neigh(result_arrays.p, idx_results_p_ref);

                        subset::fill_from_shape_params(ss_ref, cx, cy, opt.p, interp_ref, conf.shape_func);

                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (result_arrays.cost[idx_results_def] < conf.threshold){
                            std::fill(opt.p.begin(), opt.p.end(), 0.0);
                            int window_size = std::max(conf.max_disp, conf.ss_size);
                            get_single_window_fftcc_peak(opt.p[0], opt.p[1],
                                                         cx, cy,
                                                         ss_size_x, ss_size_y,
                                                         window_size, window_size,
                                                         img_ref, img_def,
                                                         interp_def);
                        }
                        else {
                            opt.copy_params_from_neigh(result_arrays.p, idx_results_def_p);
                        }

                        // optimize
                        OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

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
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (tid==0) pbar.update(progress+1);
                        }
                    }
                }

                queue.push(tid, temp_neigh);
            }
        }
        
        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }

    void multiwindow_only(const double *img_ref,
                              const double *img_def,
                              const Interpolator &interp_def,
                              std::vector<WindowLevel> &multiwindow,
                              const util::Config &conf,
                              const int img_num_ref,
                              const int img_num_def,
                              ResultArrays &result_arrays){

        // loop over the window sizes and calculate estimates for rigid
        // displacement using FFTCC
        for (int lvl = 0; lvl < multiwindow.size(); lvl++){
            multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
                                                     img_ref, img_def,
                                                     interp_def,
                                                     img_num_ref, img_num_def,
                                                     conf.filenames);
        }

        const int nsizes = multiwindow.size();
        const int last_size = nsizes-1;
        const int results_num = img_num_ref-1;

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = multiwindow[last_size].layout.num;

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
            res.converged=true;
            res.above_threshold=true;
            result_arrays.append(res, results_num, ss);
        }
    }


 void multiwindow_reliability_guided_r(const double *img_ref,
                                       const double *img_def,
                                       const Interpolator &interp_ref,
                                       const Interpolator &interp_def,
                                       std::vector<WindowLevel> &multiwindow,
                                       const util::Config &conf,
                                       const int img_num_ref,
                                       const int img_num_def,
                                       ResultArrays &stereo_results,
                                       ResultArrays &temporal_results){


        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int seed_x = conf.rg_seed.first;
        const int seed_y = conf.rg_seed.second;

        // subset information
        const subset::Grid &ss_grid = multiwindow.back().layout;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;
        const int results_num = img_num_def-1 - (conf.num_def_img+1);


        // loop over the window sizes and calculate estimates for rigid
        // displacement using FFTCC
        for (int lvl = 0; lvl < multiwindow.size(); lvl++){
            multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
                                                     img_ref, img_def,
                                                     interp_def,
                                                     img_num_ref, img_num_def,
                                                     conf.filenames);
        }

        // progress bar
        std::string bar_title = "Temporal matching for \033[1;4m" + conf.filenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ss_grid)) {
            return;
        }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        rg::QueueLocal queue(omp_get_thread_num());

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size_x, ss_size_x);
            subset::Pixels ss_ref(ss_size_y, ss_size_y);

            // Optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);

            std::vector<std::unique_ptr<FFT>> fft_windows;

            for (size_t lvl = 0; lvl < multiwindow.size(); lvl++) {
                fft_windows.push_back(std::make_unique<FFT>(multiwindow[lvl].layout.size_x, 
                                                            multiwindow[lvl].layout.size_y));
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
                int grid_x = seed_x / ss_step;
                int grid_y = seed_y / ss_step;
                int idx = ss_grid.mask[grid_y * ss_grid.num_ss_x + grid_x];

                double cx = ss_grid.coords[2*idx];
                double cy = ss_grid.coords[2*idx+1];

                // index of stereo results for seed subset
                int idx_stereo = stereo_results.index(idx, 0);
                int idx_stereo_p = stereo_results.index_parameters(idx, 0);

                // if the first image. Take the optimization parameters from rigid fourier
                opt.copy_params_from_fft(idx, multiwindow.back().u,  multiwindow.back().v);

                // Extract REFERENCE subset in the RIGHT image using shape parameters
                std::vector<double> p_stereo(6);
                for (int i = 0; i < conf.num_params; i++){
                    p_stereo[i] = stereo_results.p[idx_stereo_p+i];
                }

                // fill the reference subset based on the shape function
                // parameters that map the seed in the left image to the seed
                // in the right image
                subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);


                OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);


                // for (int i=0; i < ss_ref.num_px; i++){
                //     std::cout << "temporal_R " << ss_ref.x[i] << " " << ss_ref.y[i] << " " << ss_ref.vals[i] << " ";
                //     std::cout << ss_def.x[i] << " " << ss_def.y[i] << " " << ss_def.vals[i] << std::endl;
                //
                // }
                // std::cout << std::endl;
                //
                rg::check_convergence_or_exit(seed_x, seed_y, seed_res);

                // append the results for the current subset to result vectors
                temporal_results.append(seed_res, results_num, idx);

                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    // index of temporal results for seed subset
                    int nidx_stereo = stereo_results.index(nidx, 0);
                    int nidx_stereo_p = stereo_results.index_parameters(nidx, 0);

                    int nx = ss_grid.coords[nidx*2];
                    int ny = ss_grid.coords[nidx*2+1];

                    double cx, cy;
                    subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);

                    // Extract reference subset in the right image
                    // I need to do this based on the shape function parameters 
                    std::vector<double> p_stereo(6);
                    for (int i = 0; i < conf.num_params; i++){
                        p_stereo[i] = stereo_results.p[nidx_stereo_p+i];
                        //std::cout << p_stereo[i] << std::endl;
                    }

                    subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);

                    // get parameter values from fft output or from previous image
                    opt.copy_params_from_fft(nidx, multiwindow.back().u,  multiwindow.back().v);

                    // perform optimization for seed point neighbours
                    OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                    rg::check_convergence_or_exit(cx, cy, nres);

                    // append the results for the current subset to result vectors
                    temporal_results.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // Add points to queue
                    queue.push(tid, {rg::Point(nidx,nres.cost)});

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        if (omp_get_thread_num()==0) pbar.update(progress);
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

            rg::Point current(0, 0);

            while (!stop_request) {

                if (!queue.pop(tid, current))
                    break;

                temp_neigh.clear();

                // index of current point in results arrays
                int idx_results = temporal_results.index(current.idx, results_num);
                int idx_results_p = temporal_results.index_parameters(current.idx, results_num);

                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];



                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        int nx = ss_grid.coords[nidx*2];
                        int ny = ss_grid.coords[nidx*2+1];

                        // index of temporal results for seed subset
                        int nidx_stereo = stereo_results.index(nidx, 0);
                        int nidx_stereo_p = stereo_results.index_parameters(nidx, 0);

                        double cx, cy;
                        subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);

                        // Extract reference subset in the right image
                        // I need to do this based on the shape function parameters
                        std::vector<double> p_stereo(6);
                        for (int i = 0; i < conf.num_params; i++){
                            p_stereo[i] = stereo_results.p[nidx_stereo_p+i];
                        }

                        subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);

                        if (temporal_results.cost[idx_results] < conf.threshold)
                            opt.copy_params_from_fft(nidx,
                                                    multiwindow.back().u,
                                                    multiwindow.back().v);
                        else 
                            opt.copy_params_from_neigh(temporal_results.p, idx_results_p);

                        // optimize
                        OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                        // append results
                        temporal_results.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (omp_get_thread_num()==0) pbar.update(progress);
                        }
                    }
                }
                queue.push(tid, temp_neigh);
            }
        }
        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }


}
