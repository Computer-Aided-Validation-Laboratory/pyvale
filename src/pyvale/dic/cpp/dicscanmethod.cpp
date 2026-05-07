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
#include "./dicrg.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow.hpp"
#include "./dicshapefunc.hpp"
#include "./stereoutil.hpp"

namespace scanmethod {


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

void multiwindow_reliability_guided(const Image &img_ref,
                                    const Image &img_def,
                                    const Interpolator &interp_def,
                                    std::vector<WindowLevel> &multiwindow,
                                    const util::Config &conf,
                                    const int img_num_ref,
                                    const int img_num_def,
                                    ResultArrays &results_def){

        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;

        // subset information
        const subset::Grid &ss_grid = multiwindow.back().layout;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;

        // loop over the window sizes and calculate estimates for rigid
        // displacement using FFTCC
        for (int lvl = 0; lvl < multiwindow.size(); lvl++){
            multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
                                                      img_ref, img_def,
                                                      interp_def,
                                                      img_num_ref, img_num_def,
                                                      conf.basenames);
        }

    // auto res = coarsefine::coarse_to_fine_search(
    //     img_ref, img_def,
    //     conf.px_hori, conf.px_vert,
    //     2000, 2000,  // subset center in reference image
    //     51,                  // subset size
    //     1000,                // max displacement in pixels
    //     true,                // subpixel
    //     "GAUSSIAN_2D"        // peak fit method
    // );
    // if (res.success) {
    //     printf("dx=%.3f  dy=%.3f  peak=%.2f\n", res.disp_x, res.disp_y, res.peak_val);
    // }

        // progress bar
        std::string bar_title = "Temporal matching for \033[1;4m" + conf.basenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.basenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        const auto t0 = std::chrono::steady_clock::now();

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        rg::QueueGlobal queue(omp_get_max_threads());

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size_x, ss_size_x);
            subset::Pixels ss_ref(ss_size_y, ss_size_y);

            // Optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);

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

                // num seeds
                int num_seeds = conf.rg_seeds.size() / 2;


                for (int s = 0; s < num_seeds; s++){

                    int seed_x = conf.rg_seeds[2*s];
                    int seed_y = conf.rg_seeds[2*s+1];

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

                    OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                    rg::check_convergence_or_exit(seed_x, seed_y, seed_res);

                    // append the results for the current subset to result vectors
                    results_def.append(seed_res, idx);

                    computed_mask[idx].store(1);

                    // loop over the neighbours for the initial seed point
                    for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                        if (stop_request) continue;

                        // subset index of neighbour to the current point
                        int nidx = ss_grid.neigh[idx][n];

                        const double cx_img0 = ss_grid.coords[nidx*2];
                        const double cy_img0 = ss_grid.coords[nidx*2+1];

                        const int corner_x = int(cx - ss_size_x/2);
                        const int corner_y = int(cy - ss_size_y/2);

                        subset::fill_from_img(ss_ref, corner_x, corner_y, px_hori, px_vert, img_ref);

                        // get parameter values from fft output or from previous image
                        opt.copy_params_from_fft(nidx, multiwindow.back().u, multiwindow.back().v);

                        // perform optimization for seed point neighbours
                        OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                        rg::check_convergence_or_exit(cx, cy, nres, true);

                        // append the results for the current subset to result vectors
                        results_def.append(nres, nidx);

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

                        if (results_def.cost[current.idx] < conf.threshold)
                            opt.copy_params_from_fft(nidx,
                                                     multiwindow.back().u,
                                                     multiwindow.back().v);
                        else 
                            opt.copy_params_from_neigh(results_def.p, current.idx);

                        // optimize
                        OptResult nres(opt.num_params);
                        if (ss_ref.sum!=0) nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                        // append results
                        results_def.append(nres, nidx);

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

    void singlewindow_incremental_reliability_guided(const Image &img_ref,
                                                     const Image &img_def,
                                                     const Interpolator &interp_ref,
                                                     const Interpolator &interp_def,
                                                     const subset::Grid &ss_grid,
                                                     const util::Config &conf,
                                                     const int img_num_ref,
                                                     const int img_num_def,
                                                     const ResultArrays &results_ref,
                                                     ResultArrays &results_def,
                                                     const std::string &mode,
                                                     const std::optional<Eigen::Matrix3d> &F){





        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;


        auto get_initial_guess = [&](std::vector<double> &p, double &max_val, double cx, double cy) {
            get_single_window_fftcc_peak(p, max_val, cx, cy,
                                            ss_size_x, ss_size_y,
                                            conf.max_disp, conf.max_disp,
                                            img_ref, img_def, interp_def);
        };

        std::string bar_title = "Temporal matching for \033[1;4m" + conf.basenames[img_num_ref] +
                                          "\033[0m and \033[1;4m" + conf.basenames[img_num_def] + 
                                          "\033[0m:";

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
        rg::QueueGlobal queue(omp_get_max_threads()); 

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_def(ss_size_x, ss_size_y);
            subset::Pixels ss_ref(ss_size_x, ss_size_y);
            double max_val = 0.0;

            // Optimization parameters
            Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);

            // TODO: opt.seed_iter exposed to user.
            opt.max_iter = 200;

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {


                // num seeds
                int num_seeds = conf.rg_seeds.size() / 2;

                for (int s = 0; s < num_seeds; s++){

                    int seed_x = conf.rg_seeds[2*s];
                    int seed_y = conf.rg_seeds[2*s+1];

                    // seed coordinates
                    int grid_x = seed_x / ss_step;
                    int grid_y = seed_y / ss_step;
                    int idx = ss_grid.mask[grid_y * ss_grid.num_ss_x + grid_x];



                    // get the centre coordinates for the subset in img k0
                    double cx_img0 = ss_grid.coords[2*idx];
                    double cy_img0 = ss_grid.coords[2*idx+1];


                    // get the centre coordinates for the subset in img k
                    double cx = cx_img0;
                    double cy = cy_img0;
                    if (img_num_ref>0){
                        // displacements are from k0 to k
                        cx += results_ref.u[idx];
                        cy += results_ref.v[idx];
                    }

                    // fill the reference subset
                    subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);

                    // if the first image. Take the optimization parameters from rigid fourier
                    get_initial_guess(opt.p, max_val, cx, cy);

                    // run optimizer
                    OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);
                    rg::check_convergence_or_exit(cx_img0, cy_img0, seed_res);

                    // add deformation from reference image to new results
                    if (img_num_ref > 0){
                        seed_res.u += results_ref.u[idx];
                        seed_res.v += results_ref.v[idx];
                    }

                    // append the results for the current subset to result vectors
                    results_def.append(seed_res, idx);

                    // mark subset as computed
                    computed_mask[idx].store(1);

                    // loop over the neighbours for the initial seed point
                    for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                        if (stop_request) continue;

                        // subset index of neighbour to the current point
                        int nidx = ss_grid.neigh[idx][n];

                        double cx_img0 = ss_grid.coords[nidx*2];
                        double cy_img0 = ss_grid.coords[nidx*2+1];


                        double cx = cx_img0;
                        double cy = cy_img0;
                        if (img_num_ref>0){
                            cx += results_ref.u[nidx];
                            cy += results_ref.v[nidx];
                        }

                        // fill the reference subset
                        subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);


                        // perform optimization for seed point neighbours
                        opt.copy_params_from_neigh(results_def.p, idx);

                        OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                        rg::check_convergence_or_exit(cx_img0, cy_img0, nres, true);

                        // add deformation from reference image to new results
                        if (img_num_ref > 0){
                            nres.u += results_ref.u[nidx];
                            nres.v += results_ref.v[nidx];
                        }


                        // append the results for the current subset to result vectors
                        results_def.append(nres, nidx);

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
            }

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier

            // reset seed location using the last computed point
            opt.max_iter = conf.max_iter;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            rg::Point current(0, 0);

            while (!stop_request) {

                if (!queue.pop(tid, current))
                    break;

                temp_neigh.clear();


                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        double cx_img0 = ss_grid.coords[nidx*2];
                        double cy_img0 = ss_grid.coords[nidx*2+1];

                        // add displacements from base to subset coords in img0
                        double cx = cx_img0;
                        double cy = cy_img0;

                        OptResult nres(opt.num_params);
                        if (((results_ref.above_thresh[nidx]) && (img_num_ref > 0)) || (img_num_ref == 0)){
                            cx += results_ref.u[nidx];
                            cy += results_ref.v[nidx];

                            // fill the reference subset
                            subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);

                            if (results_def.cost[current.idx] < conf.threshold){
                                if (ss_ref.sum!=0) get_initial_guess(opt.p, max_val, cx, cy);
                            }
                            else {
                                opt.copy_params_from_neigh(results_def.p, current.idx);
                            }

                            // optimize
                            if (ss_ref.sum!=0) nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, false);

                            // add deformation from reference image to new results
                            if ((nres.above_threshold) && (img_num_ref > 0)){
                                nres.u += results_ref.u[nidx];
                                nres.v += results_ref.v[nidx];
                            }

                        }

                        // append results
                        results_def.append(nres, nidx);

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

        // // TODO: build a list of subsets using openmp that have correlated poorly but have atleast
        // // one immediate neighbour above_threshold
        // std::vector<int> retry_indices;
        // #pragma omp parallel
        // {
        //     std::vector<int> local_retry;
        //     #pragma omp for nowait
        //     for (int i = 0; i < num_ss; ++i) {
        //         // If the point was reached but failed threshold
        //         if (computed_mask[i].load() == 1 && !results_def.above_thresh[i]) {
        //             // Check if any neighbor was successful
        //             for (int nidx : ss_grid.neigh[i]) {
        //                 if (results_def.above_thresh[nidx]) {
        //                     local_retry.push_back(i);
        //                     break;
        //                 }
        //             }
        //         }
        //     }
        //     #pragma omp critical
        //     retry_indices.insert(retry_indices.end(), local_retry.begin(), local_retry.end());
        // }
        //
        // // TODO: retry the correlation using the parameters for the neighbour that
        // // have successfuly correlated. use openmp
        //
        // #pragma omp parallel
        // {
        //     subset::Pixels ss_def(ss_size_x, ss_size_y);
        //     subset::Pixels ss_ref(ss_size_x, ss_size_y);
        //     Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);
        //
        //     #pragma omp for
        //     for (int i = 0; i < (int)retry_indices.size(); ++i) {
        //         int idx = retry_indices[i];
        //
        //         // Find the neighbor with the lowest cost (best match) to use as new guess
        //         int best_neigh = -1;
        //         double best_cost = -std::numeric_limits<double>::max();
        //         for (int nidx : ss_grid.neigh[idx]) {
        //             if (results_def.above_thresh[nidx] && results_def.cost[nidx] > best_cost) {
        //                 best_cost = results_def.cost[nidx];
        //                 best_neigh = nidx;
        //             }
        //         }
        //
        //         if (best_neigh != -1) {
        //             double cx_img0 = ss_grid.coords[idx*2];
        //             double cy_img0 = ss_grid.coords[idx*2+1];
        //             double cx = cx_img0 + (img_num_ref > 0 ? results_ref.u[idx] : 0);
        //             double cy = cy_img0 + (img_num_ref > 0 ? results_ref.v[idx] : 0);
        //
        //             // Re-fill reference
        //             opt.copy_params_from_neigh(results_ref.p, idx * conf.num_params);
        //             subset::fill_from_shape_params(ss_ref, cx_img0, cy_img0, opt.p, interp_ref, conf.shape_func);
        //
        //             // Use best neighbor's converged parameters as the new starting point
        //             opt.copy_params_from_neigh(results_def.p, best_neigh * conf.num_params);
        //
        //             OptResult retry_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);
        //
        //             // If retry is better than original attempt, update results
        //             if (retry_res.cost > results_def.cost[idx]) {
        //                     std::vector<double> pA(conf.num_params);
        //                     std::vector<double> pB = retry_res.p;
        //                     std::vector<double> pC(conf.num_params);
        //
        //                     // pA: k0 -> k_ref
        //                     std::copy(results_ref.p.begin() + idx*conf.num_params,
        //                             results_ref.p.begin() + idx*conf.num_params + conf.num_params,
        //                             pA.begin());
        //
        //                     if (conf.shape_func == "RIGID") {
        //                         Rigid::compose(pC, pA, pB);
        //                         Rigid::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     else if (conf.shape_func == "AFFINE"){
        //                         Affine::compose(pC, pA, pB);
        //                         Affine::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     else if (conf.shape_func == "QUAD") {
        //                         Quad::compose(pC, pA, pB);
        //                         Quad::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     retry_res.p = pC;
        //
        //                 results_def.append(retry_res, idx); 
        //             }
        //         }
        //     }
        // }

        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }

    void multiwindow_only(const Image &img_ref,
                          const Image &img_def,
                          const Interpolator &interp_ref,
                          const Interpolator &interp_def,
                          std::vector<WindowLevel> &multiwindow,
                          const subset::Grid &ss_grid,
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
                                                     conf.basenames);
        }

        const int nsizes = multiwindow.size();
        const int last_size = nsizes-1;

        #pragma omp parallel shared(stop_request, result_arrays, multiwindow, ss_grid, conf, interp_ref, interp_def)
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
                res.above_threshold=true;
                result_arrays.append(res, ss);
            }
        }
    }


// TODO: This is for matching strategy 1. Need to sort at a later date.
// This function is reliant on a multiwindow setup for the right image.
// I dunno whether this is the best approach.
 void multiwindow_reliability_guided_r(const Image &img_ref,
                                       const Image &img_def,
                                       const Interpolator &interp_ref,
                                       const Interpolator &interp_def,
                                       std::vector<WindowLevel> &multiwindow,
                                       const util::Config &conf,
                                       const int img_num_ref,
                                       const int img_num_def,
                                       ResultArrays &stereo_results,
                                       ResultArrays &temporal_results){

    //
    //     // assign some consts for readability
    //     const int px_hori = conf.px_hori;
    //     const int px_vert = conf.px_vert;
    //     const int seed_x = conf.rg_seed.first;
    //     const int seed_y = conf.rg_seed.second;
    //
    //     // subset information
    //     const subset::Grid &ss_grid = multiwindow.back().layout;
    //     const int num_ss = ss_grid.num;
    //     const int ss_size_x = ss_grid.size_x;
    //     const int ss_size_y = ss_grid.size_y;
    //     const int ss_step = ss_grid.step;
    //     const int results_num = img_num_def-1 - (conf.num_def_img+1);
    //
    //
    //     // loop over the window sizes and calculate estimates for rigid
    //     // displacement using FFTCC
    //     for (int lvl = 0; lvl < multiwindow.size(); lvl++){
    //         multiwindow[lvl].calc_rigid_displacements(multiwindow[std::max(0,lvl-1)],
    //                                                  img_ref, img_def,
    //                                                  interp_def,
    //                                                  img_num_ref, img_num_def,
    //                                                  conf.basenames);
    //     }
    //
    //     // progress bar
    //     std::string bar_title = "Temporal matching for \033[1;4m" + conf.basenames[img_num_ref] + "\033[0m and \033[1;4m" + conf.basenames[img_num_def] + "\033[0m:";
    //     ProgressBar pbar(bar_title, num_ss);
    //     std::atomic<int> current_progress(0);
    //
    //     // quick check for the initial seed point
    //     if (!rg::is_valid_point(seed_x, seed_y, ss_grid)) {
    //         return;
    //     }
    //
    //     // Initialize binary mask for computed points (initialized to 0)
    //     std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
    //     for (auto& val : computed_mask) val.store(0); 
    //
    //     // queue for each thread
    //     rg::QueueLocal queue(omp_get_thread_num());
    //
    //     # pragma omp parallel
    //     {
    //
    //         int tid = omp_get_thread_num();
    //
    //         // Initialize ref and def subsets
    //         subset::Pixels ss_def(ss_size_x, ss_size_x);
    //         subset::Pixels ss_ref(ss_size_y, ss_size_y);
    //
    //         // Optimization parameters
    //         Optimizer opt(conf.shape_func, conf.corr_crit, conf.max_iter, conf.precision, conf.threshold, ss_size_x*ss_size_y);
    //
    //         std::vector<std::unique_ptr<FFT>> fft_windows;
    //
    //         for (size_t lvl = 0; lvl < multiwindow.size(); lvl++) {
    //             fft_windows.push_back(std::make_unique<FFT>(multiwindow[lvl].layout.size_x, 
    //                                                         multiwindow[lvl].layout.size_y));
    //         }
    //
    //         // TODO: for the seed location I'm going to overwride the max 
    //         // number of iterations to make sure we get a good convergence.
    //         // this is hardcoded for now. Could do with updating so that 
    //         // the seed location is checked ahead of the main correlation run.
    //
    //         // TODO: opt.seed_iter exposed to user.
    //         opt.max_iter = 200;
    //
    //         // ---------------------------------------------------------------------------------------------------------------------------
    //         // PROCESS THE SEED SUBSET 
    //         // ---------------------------------------------------------------------------------------------------------------------------
    //         if (tid == 0) {
    //
    //             // seed coordinates
    //             int grid_x = seed_x / ss_step;
    //             int grid_y = seed_y / ss_step;
    //             int idx = ss_grid.mask[grid_y * ss_grid.num_ss_x + grid_x];
    //
    //             double cx = ss_grid.coords[2*idx];
    //             double cy = ss_grid.coords[2*idx+1];
    //
    //             // index of stereo results for seed subset
    //             int idx_stereo = stereo_results.index(idx, 0);
    //             int idx_stereo_p = stereo_results.index_parameters(idx, 0);
    //
    //             // if the first image. Take the optimization parameters from rigid fourier
    //             opt.copy_params_from_fft(idx, multiwindow.back().u,  multiwindow.back().v);
    //
    //             // Extract REFERENCE subset in the RIGHT image using shape parameters
    //             std::vector<double> p_stereo(6);
    //             for (int i = 0; i < conf.num_params; i++){
    //                 p_stereo[i] = stereo_results.p[idx_stereo_p+i];
    //             }
    //
    //             // fill the reference subset based on the shape function
    //             // parameters that map the seed in the left image to the seed
    //             // in the right image
    //             subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);
    //
    //
    //             OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def);
    //
    //
    //             // for (int i=0; i < ss_ref.num_px; i++){
    //             //     std::cout << "temporal_R " << ss_ref.x[i] << " " << ss_ref.y[i] << " " << ss_ref.vals[i] << " ";
    //             //     std::cout << ss_def.x[i] << " " << ss_def.y[i] << " " << ss_def.vals[i] << std::endl;
    //             //
    //             // }
    //             // std::cout << std::endl;
    //             //
    //             rg::check_convergence_or_exit(seed_x, seed_y, seed_res);
    //
    //             // append the results for the current subset to result vectors
    //             temporal_results.append(seed_res, results_num, idx);
    //
    //             computed_mask[idx].store(1);
    //
    //             // loop over the neighbours for the initial seed point
    //             for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {
    //
    //                 // subset index of neighbour to the current point
    //                 int nidx = ss_grid.neigh[idx][n];
    //
    //                 // index of temporal results for seed subset
    //                 int nidx_stereo = stereo_results.index(nidx, 0);
    //                 int nidx_stereo_p = stereo_results.index_parameters(nidx, 0);
    //
    //                 int nx = ss_grid.coords[nidx*2];
    //                 int ny = ss_grid.coords[nidx*2+1];
    //
    //                 double cx, cy;
    //                 subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);
    //
    //                 // Extract reference subset in the right image
    //                 // I need to do this based on the shape function parameters 
    //                 std::vector<double> p_stereo(6);
    //                 for (int i = 0; i < conf.num_params; i++){
    //                     p_stereo[i] = stereo_results.p[nidx_stereo_p+i];
    //                     //std::cout << p_stereo[i] << std::endl;
    //                 }
    //
    //                 subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);
    //
    //                 // get parameter values from fft output or from previous image
    //                 opt.copy_params_from_fft(nidx, multiwindow.back().u,  multiwindow.back().v);
    //
    //                 // perform optimization for seed point neighbours
    //                 OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);
    //
    //                 rg::check_convergence_or_exit(cx, cy, nres);
    //
    //                 // append the results for the current subset to result vectors
    //                 temporal_results.append(nres, results_num, nidx);
    //
    //                 // update mask
    //                 computed_mask[nidx].store(1);
    //
    //                 // Add points to queue
    //                 queue.push(tid, {rg::Point(nidx,nres.cost)});
    //
    //                 // update progress bar
    //                 if (g_debug_level>0){
    //                     int progress = current_progress.fetch_add(1);
    //                     if (omp_get_thread_num()==0) pbar.update(progress);
    //                 }
    //             }
    //         }
    //
    //
    //         // ---------------------------------------------------------------------------------------------------------------------------
    //         // PROCESS ALL OTHER SUBSETS
    //         // ---------------------------------------------------------------------------------------------------------------------------
    //         #pragma omp barrier
    //
    //         // TODO: reset seed location using the last computed point
    //         opt.max_iter = conf.max_iter;
    //
    //         std::vector<rg::Point> temp_neigh;
    //         temp_neigh.reserve(4);
    //
    //         rg::Point current(0, 0);
    //
    //         while (!stop_request) {
    //
    //             if (!queue.pop(tid, current))
    //                 break;
    //
    //             temp_neigh.clear();
    //
    //             // index of current point in results arrays
    //             int idx_results = temporal_results.index(current.idx, results_num);
    //             int idx_results_p = temporal_results.index_parameters(current.idx, results_num);
    //
    //             // loop over neighbouring points
    //             for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {
    //
    //                 // subset index of neighbour to the current point
    //                 int nidx = ss_grid.neigh[current.idx][n];
    //
    //
    //
    //                 int expected = 0;
    //                 expected = computed_mask[nidx].exchange(1);
    //                 if (expected == 0) {
    //
    //                     // coords of neigh
    //                     int nx = ss_grid.coords[nidx*2];
    //                     int ny = ss_grid.coords[nidx*2+1];
    //
    //                     // index of temporal results for seed subset
    //                     int nidx_stereo = stereo_results.index(nidx, 0);
    //                     int nidx_stereo_p = stereo_results.index_parameters(nidx, 0);
    //
    //                     double cx, cy;
    //                     subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);
    //
    //                     // Extract reference subset in the right image
    //                     // I need to do this based on the shape function parameters
    //                     std::vector<double> p_stereo(6);
    //                     for (int i = 0; i < conf.num_params; i++){
    //                         p_stereo[i] = stereo_results.p[nidx_stereo_p+i];
    //                     }
    //
    //                     subset::fill_from_shape_params(ss_ref, cx, cy, p_stereo, interp_ref, conf.shape_func);
    //
    //                     if (temporal_results.cost[idx_results] < conf.threshold)
    //                         opt.copy_params_from_fft(nidx,
    //                                                 multiwindow.back().u,
    //                                                 multiwindow.back().v);
    //                     else 
    //                         opt.copy_params_from_neigh(temporal_results.p, idx_results_p);
    //
    //                     // optimize
    //                     OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);
    //
    //                     // append results
    //                     temporal_results.append(nres, results_num, nidx);
    //
    //                     // add results to temp neighbour results
    //                     temp_neigh.emplace_back(nidx, nres.cost);
    //
    //                     // update progress bar
    //                     if (g_debug_level>0){
    //                         int progress = current_progress.fetch_add(1);
    //                         if (omp_get_thread_num()==0) pbar.update(progress);
    //                     }
    //                 }
    //             }
    //             queue.push(tid, temp_neigh);
    //         }
    //     }
    //     if (g_debug_level>0){
    //         pbar.update(current_progress+1);
    //         pbar.finish();
    //     }
    }


}
