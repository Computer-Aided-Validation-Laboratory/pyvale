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
#include "./dicrg.hpp"
#include "./dicsubset.hpp"
#include "./dicresults.hpp"
#include "./dicmultiwindow_util.hpp"
#include "./dicmultiwindow_rg.hpp"

void multiwindow_rg(const Image &img_ref,
                    const Image &img_def,
                    const Interpolator &interp_ref,
                    const Interpolator &interp_def,
                    std::vector<WindowLevel> &multiwindow,
                    const util::Config &conf,
                    const int img_num_ref,
                    const int img_num_def,
                    const ResultArrays &results_ref,
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
        subset::Pixels ss_def(ss_size_x, ss_size_y);
        subset::Pixels ss_ref(ss_size_x, ss_size_y);

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

                // if a seed subset is no longer active then we've hit a problemo
                if (!ss_grid.active_ss[idx]){
                    throw std::runtime_error("Seed subset (" + std::to_string(seed_x) + "," +
                                             std::to_string(seed_y) + ") is no longer active. The seed subset failed to converge in a previous calculation");
                }

                double cx = ss_grid.coords[2*idx];
                double cy = ss_grid.coords[2*idx+1];


                // if the first image. Take the optimization parameters from rigid fourier
                opt.copy_params_from_fft(idx, multiwindow.back().u, multiwindow.back().v);

                // Extract reference subset and solve for starting seed point
                subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);

                OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                rg::check_convergence(seed_x, seed_y, seed_res);

                if (img_num_ref > 0){
                    seed_res.u += results_ref.u[idx];
                    seed_res.v += results_ref.v[idx];
                }

                // append the results for the current subset to result vectors
                results_def.append(seed_res, idx);

                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    if (stop_request) continue;

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    const double cx = ss_grid.coords[nidx*2];
                    const double cy = ss_grid.coords[nidx*2+1];

                    if (!ss_grid.active_ss[nidx]){
                    throw std::runtime_error("Direct neighbour (" + std::to_string(cx) + "," + 
                                                std::to_string(cy) + ") of seed subset (" + std::to_string(seed_x) + "," +
                                                std::to_string(seed_y) + ") is no longer active. The seed subset failed to converge in a previous calculation");
                    }

                    // fill the reference subset
                    subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);

                    // perform optimization for seed point neighbours
                    opt.copy_params_from_neigh(results_def.p, idx);

                    // perform optimization for seed point neighbours
                    OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                    rg::check_convergence(cx, cy, nres, true);

                    if (img_num_ref > 0){
                        nres.u += results_ref.u[nidx];
                        nres.v += results_ref.v[nidx];
                    }

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
                    double cx = ss_grid.coords[nidx*2];
                    double cy = ss_grid.coords[nidx*2+1];

                    OptResult nres(opt.num_params);

                    // if the subset is no longer active then skip
                    if (!ss_grid.active_ss[nidx]){
                        results_def.append(nres, nidx);
                        continue;
                    }

                    // fill the reference subset
                    subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);

                    if (results_def.above_thresh[current.idx])
                        opt.copy_params_from_neigh(results_def.p, current.idx);
                    else
                        opt.copy_params_from_fft(nidx, multiwindow.back().u, multiwindow.back().v);

                    // optimize
                    if (ss_ref.sum!=0) nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def);

                    if (img_num_ref > 0){
                        nres.u += results_ref.u[nidx];
                        nres.v += results_ref.v[nidx];
                    }

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
