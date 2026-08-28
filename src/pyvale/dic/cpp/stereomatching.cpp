// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <omp.h>
#include <vector>
#include <cmath>
#include <thread>

// common_cpp header files
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"

// calibration header files

// dic header files
#include "./stereomatching.hpp"
#include "./stereoutil.hpp"
#include "./dicsubset.hpp"
#include "./dicoptimizer.hpp"
#include "./dicinterp.hpp"
#include "./dicrg.hpp"
#include "./dicshapefunc.hpp"


// Eigen 
#include <Eigen/Dense>

namespace stereo {


void matching(const Image &img_l,
              const Image &img_r,
              const Interpolator &interp_l,
              const Interpolator &interp_r,
              const subset::Grid &ss_grid,
              const util::Config &conf,
              const int img_num_def_l,
              const int img_num_def_r,
              const Eigen::Matrix3d &F,
              const ResultArrays &results_l,
              ResultArrays &results_r){




        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;


        auto get_initial_guess = [&](std::vector<double> &p, double cx, double cy, bool print) {
            stereo::get_rigid_translation_from_rectified_fft(p, cx, cy, ss_size_x, ss_size_y,
                                                             2*conf.max_disp, ss_size_y, F,
                                                             interp_l, interp_r, 0.0, 0.0, print);
        };

        std::string bar_title = "Stereo \033[1;4m" + conf.basenames[img_num_def_l] +
                                    "\033[0m -> \033[1;4m" + conf.basenames[img_num_def_r] + 
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

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize ref and def subsets
            subset::Pixels ss_r(ss_size_x, ss_size_y);
            subset::Pixels ss_l(ss_size_x, ss_size_y);

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
                    double cx_ref_l = ss_grid.coords[2*idx];
                    double cy_ref_l = ss_grid.coords[2*idx+1];

                    // get the centre coordinates for the subset in img k
                    double cx_def_l = cx_ref_l + results_l.u[idx];
                    double cy_def_l = cy_ref_l + results_l.v[idx];

                    // populate the subset for img k using shape function parameters
                    // that map subset in img k0 to k.
                    opt.copy_params_from_neigh(results_l.p, idx);
                    subset::fill_from_shape_params(ss_l, cx_ref_l, cy_ref_l, opt.p, interp_l, conf.shape_func);

                    // if the first image. Take the optimization parameters from rigid fourier
                    get_initial_guess(opt.p, cx_def_l, cy_def_l, false);

                    // run optimizer
                    OptResult seed_res = opt.solve(cx_def_l, cy_def_l, ss_l, ss_r, interp_r, true);
            
                    if (!rg::check_convergence(cx_ref_l, cy_ref_l, seed_res, error_message)) {
                        error_flag.store(true);
                        break;
                    }

                    // add deformation from reference image to new results
                    if (img_num_def_l > 0){
                        std::vector<double> pA(conf.num_params);
                        std::vector<double> pB = seed_res.p;
                        std::vector<double> pC(conf.num_params);


                        // pA: k0 -> k_l
                        std::copy(results_l.p.begin() + idx*conf.num_params,
                                results_l.p.begin() + idx*conf.num_params + conf.num_params,
                                pA.begin());

                        if (conf.shape_func == util::ShapeFunc::RIGID) {
                            Rigid::compose(pC, pA, pB);
                            Rigid::get_displacement(seed_res.u, seed_res.v, 0.0, 0.0, pC);
                        }
                        else if (conf.shape_func == util::ShapeFunc::AFFINE){
                            Affine::compose(pC, pA, pB);
                            Affine::get_displacement(seed_res.u, seed_res.v, 0.0, 0.0, pC);
                        }
                        else if (conf.shape_func == util::ShapeFunc::QUAD) {
                            Quad::compose(pC, pA, pB);
                            Quad::get_displacement(seed_res.u, seed_res.v, 0.0, 0.0, pC);
                        }
                        seed_res.p = pC;
                    }

                    // append the results for the current subset to result vectors
                    results_r.append(seed_res, idx);

                    // mark subset as computed
                    computed_mask[idx].store(1);

                    // loop over the neighbours for the initial seed point
                    for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                        // subset index of neighbour to the current point
                        int nidx = ss_grid.neigh[idx][n];

                        double cx_ref_l = ss_grid.coords[nidx*2];
                        double cy_ref_l = ss_grid.coords[nidx*2+1];

                        double cx_def_l = cx_ref_l + results_l.u[nidx];
                        double cy_def_l = cy_ref_l + results_l.v[nidx];

                        // fill the reference subset using the updated cx,cy and
                        // the shape function parameters for the correlation of
                        // the reference image
                        opt.copy_params_from_neigh(results_l.p, nidx);
                        subset::fill_from_shape_params(ss_l, cx_ref_l, cy_ref_l, opt.p, interp_l, conf.shape_func);

                        // perform optimization for seed point neighbours
                        opt.copy_params_from_neigh(results_r.p, idx);

                        OptResult nres = opt.solve(cx_def_l, cy_def_l, ss_l, ss_r, interp_r, true);

                        if (!rg::check_convergence(cx_def_l, cy_def_l, nres, error_message, true)) {
                            error_flag.store(true);
                            break;
                        }

                        // add deformation from reference image to new results
                        std::vector<double> pA(conf.num_params);
                        std::vector<double> pB = nres.p;
                        std::vector<double> pC(conf.num_params);


                        // pA: k0 -> k_l
                        std::copy(results_l.p.begin() + nidx*conf.num_params,
                                results_l.p.begin() + nidx*conf.num_params + conf.num_params,
                                pA.begin());

                        if (conf.shape_func == util::ShapeFunc::RIGID) {
                            Rigid::compose(pC, pA, pB);
                            Rigid::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                        }
                        else if (conf.shape_func == util::ShapeFunc::AFFINE){
                            Affine::compose(pC, pA, pB);
                            Affine::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                        }
                        else if (conf.shape_func == util::ShapeFunc::QUAD) {
                            Quad::compose(pC, pA, pB);
                            Quad::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                        }
                        nres.p = pC;


                        // append the results for the current subset to result vectors
                        results_r.append(nres, nidx);

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

            while (!stop_request && !error_flag.load()) {

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
                        double cx_ref_l = ss_grid.coords[nidx*2];
                        double cy_ref_l = ss_grid.coords[nidx*2+1];

                        // add displacements from base to subset coords in ref_l
                        double cx_def_l = cx_ref_l + results_l.u[nidx];
                        double cy_def_l = cy_ref_l + results_l.v[nidx];

                        // fill the reference subset using the updated cx,cy and
                        // the shape function parameters for the correlation of
                        // the reference image
                        opt.copy_params_from_neigh(results_l.p, nidx);
                        subset::fill_from_shape_params(ss_l, cx_ref_l, cy_ref_l, opt.p, interp_l, conf.shape_func);

                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (results_r.above_thresh[current.idx])
                            opt.copy_params_from_neigh(results_r.p, current.idx);
                        else
                            get_initial_guess(opt.p, cx_def_l, cy_def_l, false);

                        // optimize
                        OptResult nres = opt.solve(cx_def_l, cy_def_l, ss_l, ss_r, interp_r);

                        // add deformation from reference image to new results
                        if (nres.above_thresh){
                            std::vector<double> pA(conf.num_params);
                            std::vector<double> pB = nres.p;
                            std::vector<double> pC(conf.num_params);

                            // pA: k0 -> k_l
                            std::copy(results_l.p.begin() + nidx*conf.num_params,
                                    results_l.p.begin() + nidx*conf.num_params + conf.num_params,
                                    pA.begin());

                            if (conf.shape_func == util::ShapeFunc::RIGID) {
                                Rigid::compose(pC, pA, pB);
                                Rigid::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                            }
                            else if (conf.shape_func == util::ShapeFunc::AFFINE){
                                Affine::compose(pC, pA, pB);
                                Affine::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                            }
                            else if (conf.shape_func == util::ShapeFunc::QUAD) {
                                Quad::compose(pC, pA, pB);
                                Quad::get_displacement(nres.u, nres.v, 0.0, 0.0, pC);
                            }
                            nres.p = pC;
                        }

                        // append results
                        results_r.append(nres, nidx);

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
        //         if (computed_mask[i].load() == 1 && !results_r.above_thresh[i]) {
        //             // Check if any neighbor was successful
        //             for (int nidx : ss_grid.neigh[i]) {
        //                 if (results_r.above_thresh[nidx]) {
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
        //     subset::Pixels ss_r(ss_size_x, ss_size_y);
        //     subset::Pixels ss_l(ss_size_x, ss_size_y);
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
        //             if (results_r.above_thresh[nidx] && results_r.cost[nidx] > best_cost) {
        //                 best_cost = results_r.cost[nidx];
        //                 best_neigh = nidx;
        //             }
        //         }
        //
        //         if (best_neigh != -1) {
        //             double cx_ref_l = ss_grid.coords[idx*2];
        //             double cy_ref_l = ss_grid.coords[idx*2+1];
        //             double cx = cx_ref_l + (img_num_l > 0 ? results_l.u[idx] : 0);
        //             double cy = cy_ref_l + (img_num_l > 0 ? results_l.v[idx] : 0);
        //
        //             // Re-fill reference
        //             opt.copy_params_from_neigh(results_l.p, idx * conf.num_params);
        //             subset::fill_from_shape_params(ss_l, cx_ref_l, cy_ref_l, opt.p, interp_l, conf.shape_func);
        //
        //             // Use best neighbor's converged parameters as the new starting point
        //             opt.copy_params_from_neigh(results_r.p, best_neigh * conf.num_params);
        //
        //             OptResult retry_res = opt.solve(cx, cy, ss_l, ss_r, interp_r, true);
        //
        //             // If retry is better than original attempt, update results
        //             if (retry_res.cost > results_r.cost[idx]) {
        //                     std::vector<double> pA(conf.num_params);
        //                     std::vector<double> pB = retry_res.p;
        //                     std::vector<double> pC(conf.num_params);
        //
        //                     // pA: k0 -> k_l
        //                     std::copy(results_l.p.begin() + idx*conf.num_params,
        //                             results_l.p.begin() + idx*conf.num_params + conf.num_params,
        //                             pA.begin());
        //
        //                     if (conf.shape_func == util::ShapeFunc::RIGID) {
        //                         Rigid::compose(pC, pA, pB);
        //                         Rigid::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     else if (conf.shape_func == util::ShapeFunc::AFFINE){
        //                         Affine::compose(pC, pA, pB);
        //                         Affine::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     else if (conf.shape_func == util::ShapeFunc::QUAD) {
        //                         Quad::compose(pC, pA, pB);
        //                         Quad::get_displacement(retry_res.u, retry_res.v, 0.0, 0.0, pC);
        //                     }
        //                     retry_res.p = pC;
        //
        //                 results_r.append(retry_res, idx); 
        //             }
        //         }
        //     }
        // }

        if (g_debug_level>0){
            pbar.finish();
        }

        if (error_flag.load()) {
            throw std::runtime_error(error_message);
        }
    }

} // namespace

