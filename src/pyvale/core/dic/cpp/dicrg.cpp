// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <queue>
#include <vector>
#include <utility>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <thread>
#include <iostream>
#include <omp.h>



// Program Header files
#include "./dicoptimizer.hpp"
#include "./dicinterpolator.hpp"
#include "./dicbruteforce.hpp"
#include "./dicutil.hpp"
#include "./defines.hpp"
#include "./dicrg.hpp"


namespace rg {

    void reliability_guided_dic_single_seed(
        const int *image_ref,
        util::Image *image_def,
        const bool *image_roi,
        const int seed_x, const int seed_y,  // Single seed point coordinates
        util::SubsetList *ss_list,
        const int num_def_images,
        const int img_num,
        const int ss_size,
        const int max_iter,
        const double precision,
        const double threshold_lm,
        const double threshold_bf,
        const double range_bf,
        const int num_params) {
        
        // Get image dimensions
        const int px_horizontal = image_def->px_horizontal;
        const int px_vertical = image_def->px_vertical;
        
        // Initialize binary mask for computed points (initialized to 0)
        // std::vector<bool> computed_mask(px_vertical * px_horizontal, false);
        std::vector<std::atomic<bool>> computed_mask(px_vertical * px_horizontal);
        for (int i = 0; i < computed_mask.size(); ++i) {
            computed_mask[i] = false;
        }

        // LOCAL QUEUE
        std::vector<std::priority_queue<CorrelationPoint>> local_queues(omp_get_max_threads());

        // SHARED QUEUE
        std::priority_queue<CorrelationPoint> shared_queue;
        std::mutex queue_mutex;
                      
        // Optimization parameters
        optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);
        
        // Initialize shared priority queue for all threads
        std::priority_queue<CorrelationPoint> point_queue;
        
        // quick check for the initial seed point
        if (!is_valid_point(seed_x, seed_y, ss_list)) {
            return;
        }

        int ss_x = seed_x;
        int ss_y = seed_y;
        // std::cout << ss_x << " " << ss_y << std::endl;

        // Initialize subsets
        util::Subset ss_def(ss_size);
        util::Subset ss_ref(ss_size);

                
        // temp p values for copy from brute force to optimization.
        double ptemp[6] = {0,0,0,0,0,0};
        
        // Extract subset and solve for starting seed point
        util::extract_ss(ss_x, ss_y, image_def, &ss_def);

        // brute force for initial subset
        brute::Parameters brute(threshold_bf, range_bf);
        brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, image_def->px_horizontal, &ss_def, &ss_ref, &brute);

        ptemp[0] = brute.p_rigid[0];
        ptemp[1] = brute.p_rigid[1];
        ptemp[2] = 0.0;
        ptemp[3] = 0.0;
        ptemp[4] = 0.0;
        ptemp[5] = 0.0;


        for (int i = 0; i < opt.num_params; i++){
            opt.p[i] = ptemp[i];
        }

        optimizer::Results seed_results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);

                
        //mark seed as computed
        int index = ss_y * px_horizontal + ss_x;
        computed_mask[index] = true;                     

        
        // Append results for the seed point
        // util::append_results(num_def_images, img_num, 0, &seed_results);
        

        int idx = ss_list->coords_to_index.find({ss_x, ss_y})->second;
        for (int neigh_idx : ss_list->neighbours[idx]) {
            
                    
            // get coordiantes
            int neigh_x = ss_list->coords[neigh_idx*2];
            int neigh_y = ss_list->coords[neigh_idx*2+1];

            util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);

            brute::Parameters brute(threshold_bf, range_bf);
            brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, image_def->px_horizontal, &ss_def, &ss_ref, &brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];
            ptemp[2] = 0.0;
            ptemp[3] = 0.0;
            ptemp[4] = 0.0;
            ptemp[5] = 0.0;

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }

            optimizer::Results neigh_results = optimizer::solve(neigh_x, neigh_y, &ss_def, &ss_ref, &opt);

            util::append_results(num_def_images, img_num, neigh_idx, neigh_results.iter, neigh_results.ftol, neigh_results.xtol, neigh_results.u, neigh_results.v, neigh_results.cost, neigh_results.p);
            
            // Add to priority queue
            computed_mask[neigh_y * px_horizontal + neigh_x] = true;

            // LOCAL QUEUE
            local_queues[0].push(CorrelationPoint(neigh_x, neigh_y, 1 - 0.5 * neigh_results.cost));

            // SHARED QUEUE
            // shared_queue.push(CorrelationPoint(neigh_x, neigh_y, 1 - 0.5 * neigh_results.cost));
                                 

        }

        // LOCAL QUEUE PROCESSING
        #pragma omp parallel firstprivate(ss_def, ss_ref, opt)
        {


            
            auto& my_queue = local_queues[omp_get_thread_num()];
            std::vector<CorrelationPoint> temp_neighbors;
            temp_neighbors.reserve(4);
            int debug_count = 0;


            const int max_idle_iters = 100;
            int idle_iters = 0;

            while (true) {

                // reset correlation values and got_point
                CorrelationPoint current(0, 0, 0);
                bool got_point = false;

                // Try threads own queue
                if (!my_queue.empty()) {
                    current = my_queue.top();
                    my_queue.pop();
                    got_point = true;
                } 
                else {
                    // Try to steal from queue with retries.
                    while (!got_point && idle_iters < max_idle_iters) {
                        for (int i = 0; i < local_queues.size(); ++i) {
                            if (!local_queues[i].empty()) {
                                current = local_queues[i].top();
                                local_queues[i].pop();
                                got_point = true;
                                break;
                            }
                        }

                        if (!got_point) {
                            ++idle_iters;
                            std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        }
                    }
                }

                if (!got_point) {
                    // All queues are empty after retries
                    break;
                }

                temp_neighbors.clear();

                int curr_x = current.x;
                int curr_y = current.y;

                int idx = ss_list->coords_to_index.find({curr_x, curr_y})->second;
                for (int neigh_idx : ss_list->neighbours[idx]) {
                           
                    // get coordiantes
                    int neigh_x = ss_list->coords[neigh_idx*2];
                    int neigh_y = ss_list->coords[neigh_idx*2+1];

                    if (!computed_mask[neigh_y * px_horizontal + neigh_x].exchange(true)) {

                        // extract subset
                        util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);

                        // seed optimization parameters with neighbour value
                        int index  = (img_num * num_def_images + idx);
                        int indexp = index*num_params;

                        // if the neighbouring subset reached the maximum number of iterations, try again from a brute force
                        if (util::niter_arr[index] == opt.max_iter && util::cost_arr[index] > opt.threshold_lm){
                            brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, image_def->px_horizontal, &ss_def, &ss_ref, &brute);
                            ptemp[0] = brute.p_rigid[0];
                            ptemp[1] = brute.p_rigid[1];
                            ptemp[2] = 0.0;
                            ptemp[3] = 0.0;
                            ptemp[4] = 0.0;
                            ptemp[5] = 0.0;

                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = ptemp[i];
                            }
                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = util::p_arr[indexp+i];
                            }
                        }

                        // optimize
                        optimizer::Results neigh_results = optimizer::solve(neigh_x, neigh_y, &ss_def, &ss_ref, &opt);

                        // append results
                        util::append_results(num_def_images, img_num, neigh_idx, neigh_results.iter, neigh_results.ftol, neigh_results.xtol, neigh_results.u, neigh_results.v, neigh_results.cost, neigh_results.p);

                        // // add results to temp neighbour results
                        temp_neighbors.emplace_back(neigh_x, neigh_y, 1.0 - 0.5 * neigh_results.cost);
                    }
                }
                // exit(0);

                for (const auto& neighbor : temp_neighbors) {
                    my_queue.push(neighbor);
                }

                // debug_count++;
                // if (debug_count == 4) exit(0);
            }
        }



        // SHARED QUEUE
        // #pragma omp parallel
        // {
        //     util::Subset ss_def(ss_size);
        //     util::Subset ss_ref(ss_size);
        //     optimizer::Parameters local_opt = opt;

        //     std::vector<CorrelationPoint> temp_neighbors;


        //     while (true) {
        //         CorrelationPoint current(0, 0, 0);
        //         bool got_point = false;

        //         // Grab a point from the shared queue
        //         {
        //             std::lock_guard<std::mutex> lock(queue_mutex);
        //             if (!shared_queue.empty()) {
        //                 current = shared_queue.top();
        //                 shared_queue.pop();
        //                 got_point = true;
        //             }
        //         }

        //         if (!got_point)
        //             break;

        //         int curr_x = current.x;
        //         int curr_y = current.y;

        //         temp_neighbors.clear();

        //         for (int i = 0; i < 4; i++) {
        //             int neigh_x = curr_x + dx[i];
        //             int neigh_y = curr_y + dy[i];

        //             if (is_valid_point(neigh_x, neigh_y, image_roi, px_horizontal, px_vertical, ss_size)) {
        //                 int index = neigh_y * px_horizontal + neigh_x;

        //                 if (!computed_mask[index].exchange(true)) {
        //                     util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);
        //                     optimizer::Results neigh_results = optimizer::solve(neigh_x, neigh_y, &ss_def, &ss_ref, &local_opt);
        //                     temp_neighbors.emplace_back(neigh_x, neigh_y, 1.0 - 0.5 * neigh_results.cost);
        //                 }
        //             }
        //         }

        //         // Push new neighbors back to the shared queue
        //         {
        //             std::lock_guard<std::mutex> lock(queue_mutex);
        //             for (const auto& neighbor : temp_neighbors) {
        //                 shared_queue.push(neighbor);
        //             }
        //         }
        //     }
        // }





    }

    inline bool is_valid_point(int x, int y, util::SubsetList *ss_list) {
        
        auto it = ss_list->coords_to_index.find({x, y});

        // check if coordinates are in the coordinate list
        if (it == ss_list->coords_to_index.end()) {
            std::cerr << "Error: coordinates not found in the coordinate list." << std::endl;
            std::cerr << "Coordinates: " << x << ", " << y << std::endl;
            exit(EXIT_FAILURE);
        }
        else return true;
    }

}


