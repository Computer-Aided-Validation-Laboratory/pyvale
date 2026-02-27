// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <omp.h>
#include <vector>
#include <cmath>

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


        // get start location of displacements in previous image
        double *prev_img_u = result_arrays.u.data() + result_arrays.index(0,std::max(0,img_num_l-1));
        double *prev_img_v = result_arrays.v.data() + result_arrays.index(0,std::max(0,img_num_l-1));

        std::string bar_title = "Stereo matching for " + conf.filenames[img_num_r] + ":";
        ProgressBar pbar(bar_title, num_ss);
        std::atomic<int> current_progress(0);

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ss_grid.mask.size());
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


                // need to add offset based on displacements from previous correlation
                double seed_x_new, seed_y_new;
                if (img_num_l == 0) {
                    seed_x_new = seed_x;
                    seed_y_new = seed_y;
                } else {
                    seed_x_new = seed_x + prev_img_u[idx];
                    seed_y_new = seed_y + prev_img_v[idx];
                }

                // centre coordinates
                subset::get_subpx_from_img(ss_l, seed_x_new, seed_y_new, interp_l);

                // equation of epipolar line for the corner
                Eigen::Vector2d closest_point, dir;
                stereo::compute_epi(closest_point, dir, seed_x_new, seed_y_new, F);

                // get an estimate for the rigid shift from fft
                get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         seed_x_new, seed_y_new,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         100,100,
                                                         img_l, interp_r);

                // run optimizer and check convergence 
                OptResult seed_res = opt_affine.solve(seed_x_new, seed_y_new, ss_l, ss_r, interp_r);
                rg::check_convergence_or_exit(seed_x_new, seed_y_new, seed_res);

                // add translation from prev reference image to new results
                if (img_num_l > 0){
                    seed_res.u += prev_img_u[idx];
                    seed_res.v += prev_img_v[idx];
                }

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

                    // need to add displacements from previous image
                    if (img_num_l > 0){
                        nx += prev_img_u[nidx];
                        ny += prev_img_v[nidx];
                    }

                    // get subset values
                    subset::get_subpx_from_img(ss_l, nx, ny, interp_l);

                    // get initial guess at parameter values from seed point
                    int index_p = matches.index_parameters(idx,results_num);

                    // AFFINE
                    for (int i = 0; i < opt_affine.num_params; i++){
                        opt_affine.p[i] = matches.p[index_p+i];
                    }

                    // perform optimization for seed point neighbours
                    OptResult nres = opt_affine.solve(nx, ny, ss_l, ss_r, interp_r);

                    // add rormation from lerence image to new results
                    if (img_num_l > 0){
                        nres.u += prev_img_u[nidx];
                        nres.v += prev_img_v[nidx];
                    }

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

            while (!stop_request) {
                
                if (!pop_next_point(tid, local_q, queue_mutexes, steal_mutex, current))
                    break;

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_results_r = matches.index(current.idx, results_num);
                int idx_results_r_p = matches.index_parameters(current.idx, results_num);


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


                        // add displacements from lerence image
                        if (img_num_l > 0){
                            nx += prev_img_u[nidx];
                            ny += prev_img_v[nidx];
                        }

                        subset::get_subpx_from_img(ss_l, nx, ny, interp_l);

                        // if the neighbouring subset had not met correlation threshold,
                        // then try values from fft windowing
                        if (matches.cost[idx_results_r] < conf.threshold){

                            // equation of epipolar line for the corner
                            Eigen::Vector2d closest_point, dir;
                            stereo::compute_epi(closest_point, dir, nx, ny, F);

                            get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                         nx, ny,
                                                         ss_size_x,ss_size_y,
                                                         closest_point, dir,
                                                         100,100,
                                                         img_l, interp_r);
                        }
                        else {
                            for (int i = 0; i < opt_affine.num_params; i++){
                                opt_affine.p[i] = matches.p[idx_results_r_p+i];
                            }
                        }

                        // optimize
                        subset::get_subpx_from_img(ss_l, nx, ny, interp_l);
                        OptResult nres = opt_affine.solve(nx, ny, ss_l, ss_r, interp_r);

                        // add rormation from lerence image to new results
                        // if ((nres.converged) && (nres.above_threshold) && (img_num_l > 0)){
                        //     nres.u += prev_img_u[nidx];
                        //     nres.v += prev_img_v[nidx];
                        // }
                        // else if (img_num_l > 0){
                        //     nres.u = prev_img_u[nidx];
                        //     nres.v = prev_img_v[nidx];
                        // }

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

        // get a list of the points that have correlated but have a neighbour
        // that has NOT correlated. 
        int unconv_count = 0;
        for (size_t ss = 0; ss < ss_grid.num; ss++) {

            // subset index of neighbour to the current point
            int unconv_neigh = 0;
            for (size_t n = 0; n < ss_grid.neigh[ss].size(); n++) {
                int nidx = ss_grid.neigh[ss][n];

                double nx = ss_grid.coords[nidx*2];
                double ny = ss_grid.coords[nidx*2+1];

                if (matches.above_thresh[ss] && !matches.above_thresh[nidx]){
                    unconv_neigh++;
                    //std::cout << ss_grid.coords[nidx*2] << " " << ss_grid.coords[nidx*2+1] << " " << matches.cost[nidx] << std::endl;
                }
            }

            if (!matches.above_thresh[ss]){
                computed_mask[ss] = 0;
                unconv_count++;
                std::cout << ss_grid.coords[ss*2] << " " << ss_grid.coords[ss*2+1] << " " << matches.cost[ss] << std::endl;
            }

            if (unconv_neigh>0){
                local_q[0].push(rg::Point(ss,matches.cost[ss]));
            }
        }
        
        // debugging check
        std::cout << std::endl;
        for (size_t ss = 0; ss < ss_grid.num; ss++){
            if (!matches.above_thresh[ss]){
                std::cout << ss_grid.coords[ss*2] << " " << ss_grid.coords[ss*2+1] << " " << matches.cost[ss] << std::endl;
            }
        }
        std::cout << std::endl;

        std::string test_title = "re match for " + conf.filenames[img_num_r] + ":";
        ProgressBar tbar(bar_title, unconv_count);
        current_progress = 0;

        int prev_thread_num = omp_get_num_threads();
        omp_set_num_threads(1);
        #pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::priority_queue<rg::Point>& thread_q = local_q[tid];

            // Initialize l and r subsets
            subset::Pixels ss_l(ss_size_x, ss_size_y);
            subset::Pixels ss_r(ss_size_x, ss_size_y);


            // Optimization parameters. Dont have quad same convergence as
            // affine otherwise its pointless
            Optimizer opt_affine("AFFINE", "ZNSSD", 40, 0.001, 0.90, ss_size_x*ss_size_y);

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {

                if (!pop_next_point(tid, local_q, queue_mutexes, steal_mutex, current))
                    break;

                temp_neigh.clear();


                // index of current point in results arrays
                int idx_results_r = matches.index(current.idx, results_num);
                int idx_results_r_p = matches.index_parameters(current.idx, results_num);


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


                        // add displacements from lerence image
                        if (img_num_l > 0){
                            nx += prev_img_u[nidx];
                            ny += prev_img_v[nidx];
                        }

                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (matches.cost[idx_results_r] < conf.threshold){

                            // equation of epipolar line for the corner
                            Eigen::Vector2d closest_point, dir;
                            stereo::compute_epi(closest_point, dir, nx, ny, F);

                            // get an estimate for the rigid shift
                            // get_rigid_translation_from_rectified_search(opt_affine.p,
                            //                                             nx, ny,
                            //                                             ss_size_x,ss_size_y,
                            //                                             closest_point, dir,
                            //                                             ss_l, interp_l, interp_r);

                            get_rigid_translation_from_rectified_fft(opt_affine.p,
                                                            nx, ny,
                                                            ss_size_x,ss_size_y,
                                                            closest_point, dir,
                                                            100,100,
                                                            img_l, interp_r);
                        }
                        else {
                            for (int i = 0; i < opt_affine.num_params; i++){
                                opt_affine.p[i] = matches.p[idx_results_r_p+i];
                            }
                        }

                        // optimize
                        subset::get_subpx_from_img(ss_l, nx, ny, interp_l);
                        OptResult nres = opt_affine.solve(nx, ny, ss_l, ss_r, interp_r);


                        //std::cout << nx << " " << ny << " " << nres.u << " " << nres.v << " " << nres.cost << " " << nidx << " ";

                        // add rormation from lerence image to new results
                        // if ((nres.converged) && (nres.above_threshold) && (img_num_l > 0)){
                        //     nres.u += prev_img_u[nidx];
                        //     nres.v += prev_img_v[nidx];
                        // }
                        // else if (img_num_l > 0){
                        //     nres.u = prev_img_u[nidx];
                        //     nres.v = prev_img_v[nidx];
                        // }

                        // append results
                        matches.append(nres, results_num, nidx);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        if (g_debug_level>0){
                            int progress = current_progress.fetch_add(1);
                            if (tid==0) tbar.update(progress+1);
                        }
                    }
                }

                for (const auto& neigh : temp_neigh) {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    thread_q.push(neigh);
                }
            }
        }

        // return to previous thread count
        omp_set_num_threads(prev_thread_num);
    }





} // namespace

