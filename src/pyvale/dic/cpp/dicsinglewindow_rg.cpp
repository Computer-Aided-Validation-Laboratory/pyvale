// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <cmath>
#include <cstdlib>
#include <atomic>
#include <cstring>
#include <omp.h>
#include <csignal>
#include <optional>
#include <stdexcept>

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
#include "./dicsinglewindow_rg.hpp"
#include "./stereoutil.hpp"


void singlewindow_rg(const Interpolator &interp_ref,
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

    auto get_initial_guess_temporal = [&](auto &fft, std::vector<double> &p, double &max_val, double cx, double cy, bool debug) {
        get_single_window_fftcc_peak_centre(fft, p, max_val, 
                                            cx, cy,
                                            0, 0,
                                            ss_size_x, ss_size_y,
                                            std::max(2*conf.max_disp, ss_size_x), 
                                            std::max(2*conf.max_disp, ss_size_y),
                                            interp_ref, interp_def, debug);

    };


    auto get_initial_guess_stereo = [&](std::vector<double> &p, double cx, double cy, bool print) {
            stereo::get_rigid_translation_from_rectified_fft(p, cx, cy, ss_size_x, ss_size_y,
                                                                2*conf.epi_distance, ss_size_y, F.value(),
                                                                interp_ref, interp_def, print);
    };

    std::string bar_title = mode + " \033[1;4m" + conf.basenames[img_num_ref] +
                                        "\033[0m -> \033[1;4m" + conf.basenames[img_num_def] + 
                                        "\033[0m:";

    ProgressBar pbar(bar_title, ss_grid.active_total);
    std::atomic<int> current_progress(0);

    // Initialize binary mask for computed points (initialized to 0)
    std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
    for (auto& val : computed_mask) val.store(0); 

    // queue for each thread
    rg::QueueGlobal queue(omp_get_max_threads()); 

    std::atomic<bool> error_flag(false);
    std::string error_message;

    int count = 0;

    # pragma omp parallel
    {

        int tid = omp_get_thread_num();

        // Initialize ref and def subsets
        subset::Pixels ss_def(ss_size_x, ss_size_y);
        subset::Pixels ss_ref(ss_size_x, ss_size_y);

        // initialize FFT stuff
        std::optional<FFTf> fft_float;
        std::optional<FFT> fft_double;
        if (conf.fft_precision == util::FFTPrecision::FLOAT32) {
            fft_float.emplace(std::max(2*conf.max_disp, ss_size_x), std::max(2*conf.max_disp, ss_size_y), false);
        } else {
            fft_double.emplace(std::max(2*conf.max_disp, ss_size_x), std::max(2*conf.max_disp, ss_size_y), false);
        }

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

                // if a seed subset is no longer active then we've hit a problemo
                if (!ss_grid.active_ss[idx]) {
                    error_message = "Seed subset (" + std::to_string(seed_x) + "," +
                                    std::to_string(seed_y) + ") is no longer active. " +
                                    "The seed subset failed to converge in a previous calculation";
                    error_flag.store(true);
                    break;
                }

                // get the centre coordinates for the subset in img k0
                double cx = ss_grid.coords[2*idx];
                double cy = ss_grid.coords[2*idx+1];

                // std::cout << count << " " << cx << " " << cy << std::endl;
                // count++;

                // fill the reference subset
                subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);
                for (int px = 0; px < ss_ref.num_px; px++) {
                    ss_ref.x[px] -= cx;
                    ss_ref.y[px] -= cy;
                }

                // if the first image. Take the optimization parameters from rigid fourier
                if (mode=="temporal") {
                    if (conf.fft_precision == util::FFTPrecision::FLOAT32) {
                        get_initial_guess_temporal(*fft_float, opt.p, max_val, cx, cy, false);
                    } else {
                        get_initial_guess_temporal(*fft_double, opt.p, max_val, cx, cy, false);
                    }
                }
                if (mode=="stereo") get_initial_guess_stereo(opt.p, cx, cy, false);


                // run optimizer
                OptResult seed_res = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                if (!rg::check_convergence(seed_x, seed_y, seed_res, error_message)) {
                    error_flag.store(true);
                    break;
                }

                // add deformation from reference image to new results
                seed_res.u += results_ref.u[idx];
                seed_res.v += results_ref.v[idx];

                // append the results for the current subset to result vectors
                results_def.append(seed_res, idx);

                // mark subset as computed
                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    if (stop_request) continue;

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    double cx = ss_grid.coords[nidx*2];
                    double cy = ss_grid.coords[nidx*2+1];

                    // std::cout << count << " " << cx << " " << cy << std::endl;
                    // count++;

                    if (!ss_grid.active_ss[nidx]) {
                        error_message = "Direct neighbour (" + std::to_string(cx) + "," + std::to_string(cy) +
                                        ") of seed subset (" + std::to_string(seed_x) + "," + std::to_string(seed_y) +
                                        ") is no longer active. The seed subset failed to converge in a previous calculation";
                        error_flag.store(true);
                        break;
                    }

                    // fill the reference subset
                    subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);
                    for (int px = 0; px < ss_ref.num_px; px++) {
                        ss_ref.x[px] -= cx;
                        ss_ref.y[px] -= cy;
                    }

                    // perform optimization for seed point neighbours
                    opt.copy_params_from_neigh(results_def.p,
                                               results_def.cost,
                                               results_def.above_thresh,
                                               ss_grid.neigh[nidx],
                                               idx);

                    OptResult nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, true);

                    if (!rg::check_convergence(seed_x, seed_y, seed_res, error_message, true)) {
                        error_flag.store(true);
                        break;
                    }

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
                    double cx = ss_grid.coords[nidx*2];
                    double cy = ss_grid.coords[nidx*2+1];

                    // std::cout << count << " " << cx << " " << cy << std::endl;
                    // count++;

                    OptResult nres(opt.num_params);

                    // if the subset is no longer active then skip
                    if (!ss_grid.active_ss[nidx]){
                        results_def.append(nres, nidx);
                        continue;
                    }

                    // fill the reference subset
                    subset::fill_from_centre_coords(ss_ref, cx, cy, interp_ref);
                    for (int px = 0; px < ss_ref.num_px; px++) {
                        ss_ref.x[px] -= cx;
                        ss_ref.y[px] -= cy;
                    }

                    if (results_def.above_thresh[current.idx]){
                        opt.copy_params_from_neigh(results_def.p,
                                                   results_def.cost,
                                                   results_def.above_thresh,
                                                   ss_grid.neigh[nidx],
                                                   current.idx);
                        if (ss_ref.sum!=0) nres = opt.solve(cx, cy, ss_ref, ss_def, interp_def, false);
                    }

                    // add deformation from reference image to new results
                    nres.u += results_ref.u[nidx];
                    nres.v += results_ref.v[nidx];

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

    if (error_flag.load()) {
        throw std::runtime_error(error_message);
    }
}
