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


// Eigen 
#include <Eigen/Dense>

namespace stereo {


    void matching(const double *img_l,
                  const double *img_r,
                  const Interpolator &interp_l,
                  const Interpolator &interp_r,
                  const subset::Grid &ss_grid,
                  const util::Config &conf,
                  const int img_num_l,
                  const int img_num_r,
                  const Eigen::Matrix3d &F,
                  ResultArrays &result_arrays,
                  ResultArrays &matches){


        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        int seed_x = conf.rg_seed.first;
        int seed_y = conf.rg_seed.second;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;
        const int results_num = 0;

        std::string bar_title = "Stereo matching for \033[1;4m" + conf.filenames[img_num_l] + "\033[0m and \033[1;4m" + conf.filenames[img_num_r] + "\033[0m:";        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        std::priority_queue<rg::Point> q_global;
        std::atomic<int> active_threads(omp_get_max_threads()); 

        // Mutex vector to protect each queue
        std::mutex q_mutex;
        std::mutex steal_mutex;

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();

            // Initialize l and r subsets
            subset::Pixels ss_l(ss_size_x, ss_size_y);
            subset::Pixels ss_r(ss_size_x, ss_size_y);

            // Optimization parameters. Dont have quad same convergence as
            // affine otherwise its pointless
            Optimizer opt_affine("AFFINE", "ZNSSD", 40, 0.001, 0.90, ss_size_x*ss_size_y);


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int x = seed_x / ss_step;
                int y = seed_y / ss_step;
                int idx = ss_grid.mask[y * ss_grid.num_ss_x + x];

                double cx, cy;
                subset::get_centre(cx, cy, seed_x, seed_y, ss_size_x, ss_size_y);

                // centre coordinates
                subset::fill_from_img_subpx(ss_l, seed_x, seed_y, interp_l);


                // for (int i=0; i < ss_l.num_px; i++){
                //     std::cout << "matching " << ss_l.x[i] << " " << ss_l.y[i] << " " << ss_l.vals[i] << std::endl;
                //
                // }
                // std::cout << std::endl;

                // equation of epipolar line for the corner
                Eigen::Vector2d closest_point, dir;
                stereo::compute_epi(closest_point, dir, seed_x, seed_y, F);

                // get an estimate for the rigid shift from fft
                get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         seed_x, seed_y,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         200,200,
                                                         img_l, interp_r,false);

                // run optimizer and check convergence 
                OptResult seed_res = opt_affine.solve(cx, cy, ss_l, ss_r, interp_r);
                rg::check_convergence_or_exit(seed_x, seed_y, seed_res);

                // append the results for the current subset to result vectors
                matches.append(seed_res, results_num, idx);

                // mark subset as computed
                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    double nx = ss_grid.coords[nidx*2];
                    double ny = ss_grid.coords[nidx*2+1];

                    double cx, cy;
                    subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);

                    // get subset values
                    subset::fill_from_img_subpx(ss_l, nx, ny, interp_l);

                    // get initial guess at parameter values from seed point
                    int index_p = matches.index_parameters(idx,results_num);

                    // AFFINE
                    for (int i = 0; i < opt_affine.num_params; i++){
                        opt_affine.p[i] = matches.p[index_p+i];
                    }

                    // perform optimization for seed point neighbours
                    OptResult nres = opt_affine.solve(cx, cy, ss_l, ss_r, interp_r);

                    rg::check_convergence_or_exit(nx, ny, nres);

                    // append the results for the current subset to result vectors
                    matches.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // add this point to queue
                    q_global.push(rg::Point(nidx,nres.cost));

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        pbar.update(progress+1);
                    }
                }
            }

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier


            opt_affine.max_iter = 40;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {
                
                //if (!rg::pop_next_point_local(tid, local_q, queue_mutexes, steal_mutex, current))
                if (!rg::pop_next_point_global(q_global, q_mutex, active_threads, current))
                    break;

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_matches = matches.index(current.idx, results_num);
                int idx_matches_p = matches.index_parameters(current.idx, results_num);


                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        double nx = ss_grid.coords[nidx*2];
                        double ny = ss_grid.coords[nidx*2+1];


                        double cx, cy;
                        subset::get_centre(cx, cy, nx, ny, ss_size_x, ss_size_y);

                        subset::fill_from_img_subpx(ss_l, nx, ny, interp_l);

                        // if the neighbouring subset had not met correlation threshold,
                        // then try values from fft windowing
                        if (matches.cost[idx_matches] < conf.threshold){

                            // equation of epipolar line for the corner
                            Eigen::Vector2d closest_point, dir;
                            stereo::compute_epi(closest_point, dir, nx, ny, F);

                            get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         nx, ny,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         400,400,
                                                         img_l, interp_r, false);
                        }
                        else {
                            for (int i = 0; i < opt_affine.num_params; i++){
                                opt_affine.p[i] = matches.p[idx_matches_p+i];
                            }
                        }

                        // optimize
                        subset::fill_from_img_subpx(ss_l, nx, ny, interp_l);
                        OptResult nres = opt_affine.solve(cx, cy, ss_l, ss_r, interp_r);

                        // append results
                        matches.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (tid==0) pbar.update(progress+1);
                        }
                    }
                }

                //rg::push_points_local(tid, local_q, temp_neigh, queue_mutexes);
                rg::push_points_global(q_global, q_mutex, temp_neigh);
            }
        }
        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }



void matching_strategy3(const double *img_l,
                        const double *img_r,
                        const Interpolator &interp_l,
                        const Interpolator &interp_r,
                        const subset::Grid &ss_grid,
                        const util::Config &conf,
                        const int img_num_l,
                        const int img_num_r,
                        const Eigen::Matrix3d &F,
                        ResultArrays &temporal,
                        ResultArrays &matches){


        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        int seed_x = conf.rg_seed.first;
        int seed_y = conf.rg_seed.second;
        const int num_ss = ss_grid.num;
        const int ss_size_x = ss_grid.size_x;
        const int ss_size_y = ss_grid.size_y;
        const int ss_step = ss_grid.step;
        const int results_num = 0;

        std::string bar_title = "Stereo matching 3 for " + conf.filenames[img_num_r] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
        for (auto& val : computed_mask) val.store(0); 

        // queue for each thread
        std::vector<std::priority_queue<rg::Point>> local_q(omp_get_max_threads());
        std::atomic<int> active_threads(omp_get_max_threads());

        // Mutex vector to protect each queue
        std::vector<std::mutex> queue_mutexes(omp_get_max_threads());
        std::mutex steal_mutex;

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::priority_queue<rg::Point>& thread_q = local_q[tid];

            // Initialize l and r subsets
            subset::Pixels ss_l(ss_size_x, ss_size_y);
            subset::Pixels ss_r(ss_size_x, ss_size_y);

            // Optimization parameters. Dont have quad same convergence as
            // affine otherwise its pointless
            Optimizer opt_affine("AFFINE", "ZNSSD", 40, 0.001, 0.90, ss_size_x*ss_size_y);


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int x = seed_x / ss_step;
                int y = seed_y / ss_step;
                int idx = ss_grid.mask[y * ss_grid.num_ss_x + x];


                // need the subset pixel vals in left image
                subset::fill_from_img_subpx(ss_l, seed_x, seed_y, interp_l);

                // index of temporal results for seed subset
                int idx_temporal = temporal.index(idx, img_num_l);
                int idx_temporal_p = temporal.index_parameters(idx, img_num_l);


                // get the shape function parameters from the temporal match
                std::vector<double> p_temporal(6);
                for (int i = 0; i < opt_affine.num_params; i++){
                    p_temporal[i] = temporal.p[idx_temporal_p+i];
                    std::cout << p_temporal[i] << std::endl;
                }

                double cx, cy;
                subset::get_centre(cx, cy, seed_x, seed_y, ss_size_x, ss_size_y);

                // get the reference subset based on the parameters 
                subset::fill_from_shape_params(ss_l, seed_x, seed_y, p_temporal, interp_l, "AFFINE");

                // get the temporal displacement and use that as the initial guess
                const double u = seed_x + temporal.u[idx_temporal];
                const double v = seed_y + temporal.v[idx_temporal];


                // equation of epipolar line for the corner
                Eigen::Vector2d closest_point, dir;
                stereo::compute_epi(closest_point, dir, u, v, F);

                // get an estimate for the rigid shift from fft
                get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         u, v,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         300,300,
                                                         img_l, interp_r, true);

                std::cout << std::endl;
                for (int i = 0; i < opt_affine.num_params; i++){
                    std::cout << opt_affine.p[i] << std::endl;
                }

                // run optimizer and check convergence
                OptResult seed_res = opt_affine.solve(u, v, ss_l, ss_r, interp_r);
                rg::check_convergence_or_exit(seed_x, seed_y, seed_res);

                // append the results for the current subset to result vectors
                matches.append(seed_res, results_num, idx);


                // mark subset as computed
                computed_mask[idx].store(1);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ss_grid.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[idx][n];

                    double nx = ss_grid.coords[nidx*2];
                    double ny = ss_grid.coords[nidx*2+1];

                    // get the shape function parameters from the temporal match
                    for (int i = 0; i < opt_affine.num_params; i++){
                        p_temporal[i] = temporal.p[idx_temporal_p+i];
                    }

                    // get the reference subset based on the parameters 
                    subset::fill_from_shape_params(ss_l, nx, ny, p_temporal, interp_l, "AFFINE");

                    // get initial guess at parameter values from seed point
                    int index_p = matches.index_parameters(idx,results_num);

                    // AFFINE
                    for (int i = 0; i < opt_affine.num_params; i++){
                        opt_affine.p[i] = matches.p[index_p+i];
                    }

                    // perform optimization for seed point neighbours
                    OptResult nres = opt_affine.solve(nx, ny, ss_l, ss_r, interp_r);

                    rg::check_convergence_or_exit(nx, ny, nres);

                    // append the results for the current subset to result vectors
                    matches.append(nres, results_num, nidx);

                    // update mask
                    computed_mask[nidx].store(1);

                    // add this point to queue
                    local_q[0].push(rg::Point(nidx,nres.cost));

                    // update progress bar
                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        pbar.update(progress+1);
                    }
                }
            }

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier


            opt_affine.max_iter = 40;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            std::vector<double> p_temporal(6);

            while (!stop_request) {

                //if (!rg::pop_next_point_local(tid, local_q, queue_mutexes, steal_mutex, current))
                if (!rg::pop_next_point_global(local_q[0], queue_mutexes[0], active_threads, current))
                    break;

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_matches = matches.index(current.idx, results_num);
                int idx_matches_p = matches.index_parameters(current.idx, results_num);


                // loop over neighbouring points
                for (size_t n = 0; n < ss_grid.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ss_grid.neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        double nx = ss_grid.coords[nidx*2];
                        double ny = ss_grid.coords[nidx*2+1];

                        // index of current point in results arrays
                        int nidx_temporal = matches.index(nidx, results_num);
                        int nidx_temporal_p = matches.index_parameters(nidx, results_num);

                        // get the shape function parameters from the temporal match
                        for (int i = 0; i < opt_affine.num_params; i++){
                            p_temporal[i] = temporal.p[nidx_temporal+i];
                        }

                        // get the reference subset based on the parameters 
                        subset::fill_from_shape_params(ss_l, nx, ny, p_temporal, interp_l, "AFFINE");

                        // if the neighbouring subset had not met correlation threshold,
                        // then try values from fft windowing
                        if (matches.cost[idx_matches] < conf.threshold){

                            // equation of epipolar line for the corner
                            Eigen::Vector2d closest_point, dir;
                            stereo::compute_epi(closest_point, dir, nx, ny, F);

                            // get the temporal displacement and use that as the initial guess
                            const double u = seed_x + temporal.u[nidx_temporal];
                            const double v = seed_y + temporal.v[nidx_temporal];

                            get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         u, v,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         200,200,
                                                         img_l, interp_r, false);
                        }
                        else {
                            for (int i = 0; i < opt_affine.num_params; i++){
                                opt_affine.p[i] = matches.p[idx_matches_p+i];
                            }
                        }

                        // optimize
                        subset::fill_from_img_subpx(ss_l, nx, ny, interp_l);
                        OptResult nres = opt_affine.solve(nx, ny, ss_l, ss_r, interp_r);

                        // append results
                        matches.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (tid==0) pbar.update(progress+1);
                        }
                    }
                }

                //rg::push_points_local(tid, local_q, temp_neigh, queue_mutexes);
                rg::push_points_global(local_q[0], queue_mutexes[0], temp_neigh);
            }
        }
        if (g_debug_level>0){
            pbar.update(current_progress+1);
            pbar.finish();
        }
    }




} // namespace

