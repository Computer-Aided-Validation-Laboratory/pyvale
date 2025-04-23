// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <queue>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <iostream>
#include <omp.h>



// Program Header files
#include "./dicoptimizer.hpp"
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
        util::SubsetData *ssdata,
        const int num_def_images,
        const int img_num,
        const int max_iter,
        const double precision,
        const double threshold_lm,
        const double threshold_bf,
        const double range_bf,
        const int num_params) {
        
        // Get image dimensions
        const int px_horizontal = image_def->px_horizontal;
        const int px_vertical = image_def->px_vertical;

        int idx_dx[4] = {1, 0, -1, 0};
        int idx_dy[4] = {0, 1, 0, -1};
        int ss_dx[4]  = {ssdata->step, 0, -ssdata->step, 0};
        int ss_dy[4]  = {0, ssdata->step, 0, -ssdata->step};
        
        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<bool>> computed_mask(ssdata->mask.size());
        for (int i = 0; i < computed_mask.size(); ++i) {
            computed_mask[i] = false;
        }

        // LOCAL QUEUE
        std::vector<std::priority_queue<CorrelationPoint>> local_queues(omp_get_max_threads());

        // SHARED QUEUE
        std::priority_queue<CorrelationPoint> shared_queue;
        std::mutex queue_mutex;
                      
        // Optimization parameters
        optimizer::Parameters opt(num_params, max_iter, precision, threshold_lm, px_vertical, px_horizontal);

        // Initialize shared priority queue for all threads
        std::priority_queue<CorrelationPoint> point_queue;

        // quick check for the initial seed point
        if (!is_valid_point(seed_x, seed_y, ssdata)) {
            return;
        }

        int ss_x = seed_x;
        int ss_y = seed_y;
        int ss_x_idx  = seed_x / ssdata->step;
        int ss_y_idx  = seed_y / ssdata->step;

        // Initialize subsets
        util::Subset ss_def(ssdata->size);
        util::Subset ss_ref(ssdata->size);

        // temp p values for copy from brute force to optimization.
        double ptemp[6] = {0,0,0,0,0,0};
        
        // Extract subset and solve for starting seed point
        util::extract_ss(ss_x, ss_y, image_def, &ss_def);

        // brute force for initial subset
        brute::Parameters brute(threshold_bf, range_bf);
        brute::expanding_wavefront(ss_x, ss_y, image_ref, px_vertical, px_horizontal, &ss_def, &ss_ref, &brute);

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
        int idx = (ss_y_idx * ssdata->num_ss_x + ss_x_idx);
        computed_mask[idx] = true;                     

        
        // Append results for the seed point
        // util::append_results(num_def_images, img_num, 0, &seed_results);
        
        // loop over seed subset neighbours
        for (int i = 0; i < 4; ++i) {

            int neigh_x_ss = seed_x + ss_dx[i];
            int neigh_y_ss = seed_y + ss_dy[i];
            int neigh_x_idx = seed_x / ssdata->step + idx_dx[i];
            int neigh_y_idx = seed_y / ssdata->step + idx_dy[i];
            int neigh_idx = neigh_y_idx*ssdata->num_ss_x + neigh_x_idx;

            util::extract_ss(neigh_x_ss, neigh_y_ss, image_def, &ss_def);

            brute::Parameters brute(threshold_bf, range_bf);
            brute::expanding_wavefront(ss_x, ss_y, image_ref, px_vertical, px_horizontal, &ss_def, &ss_ref, &brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];
            ptemp[2] = 0.0;
            ptemp[3] = 0.0;
            ptemp[4] = 0.0;
            ptemp[5] = 0.0;

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }

            optimizer::Results neigh_results = optimizer::solve(neigh_x_ss, neigh_y_ss, &ss_def, &ss_ref, &opt);

            util::append_results(num_def_images, img_num, neigh_idx, neigh_results.iter, neigh_results.ftol, neigh_results.xtol, neigh_results.u, neigh_results.v, neigh_results.cost, neigh_results.p);
            
            // Add to priority queue
            computed_mask[neigh_idx] = true;

            // LOCAL QUEUE
            local_queues[0].push(CorrelationPoint(neigh_x_idx, neigh_y_idx, 1.0 - 0.5*neigh_results.cost));

            // SHARED QUEUE
            // shared_queue.push(CorrelationPoint(neigh_x, neigh_y, 1 - 0.5 * neigh_results.cost));
                                 

        }

        // LOCAL QUEUE PROCESSING
        #pragma omp parallel firstprivate(ss_def, ss_ref, opt)
        {

            auto& my_queue = local_queues[omp_get_thread_num()];
            std::vector<CorrelationPoint> temp_neighbours;
            temp_neighbours.reserve(4);
            int debug_count = 0;

            const int max_idle_iters = 100;

            while (true) {

                // reset correlation values and got_point
                CorrelationPoint current(0, 0, 0);
                bool got_point = false;
                int idle_iters = 0;

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
                    std::cerr << "Thread " << omp_get_thread_num() << " exiting. Can't find a point to process." << std::endl;
                    break;
                }

                temp_neighbours.clear();

                int idx_x = current.x;
                int idx_y = current.y;
                int idx = idx_y * ssdata->num_ss_x + idx_x;

                // loop over subset neighbours
                for (int i = 0; i < 4; ++i) {

                    int neigh_x_idx = idx_x + idx_dx[i];
                    int neigh_y_idx = idx_y + idx_dy[i];
                    int neigh_x_ss = neigh_x_idx * ssdata->step;
                    int neigh_y_ss = neigh_y_idx * ssdata->step;
                    int neigh_idx = neigh_y_idx * ssdata->num_ss_x + neigh_x_idx;

                    // if its a valid subset and its not already been computed
                    if ((ssdata->mask[neigh_idx]) && (!computed_mask[neigh_idx].exchange(true))) {

                        // extract subset
                        util::extract_ss(neigh_x_ss, neigh_y_ss, image_def, &ss_def);

                        // array index in results arrays
                        int results_index  = (img_num * num_def_images + idx);
                        int results_indexp = results_index*opt.num_params;

                        // if the neighbouring subset reached the maximum number of iterations, try again from a brute force
                        if (util::niter_arr[results_index] == opt.max_iter && util::cost_arr[results_index] > opt.threshold_lm){

                            brute::expanding_wavefront(ss_x, ss_y, image_ref, px_vertical, px_horizontal, &ss_def, &ss_ref, &brute);
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
                                opt.p[i] = util::p_arr[results_indexp+i];
                            }

                        }

                        // optimize
                        optimizer::Results neigh_results = optimizer::solve(neigh_x_ss, neigh_y_ss, &ss_def, &ss_ref, &opt);

                        // append results
                        util::append_results(num_def_images, img_num, neigh_idx, neigh_results.iter, neigh_results.ftol, neigh_results.xtol, neigh_results.u, neigh_results.v, neigh_results.cost, neigh_results.p);

                        // // add results to temp neighbour results
                        temp_neighbours.emplace_back(neigh_x_idx, neigh_y_idx, 1.0 - 0.5 * neigh_results.cost);
                    }
                }

                for (const auto& neighbour : temp_neighbours) {
                    my_queue.push(neighbour);
                }

                // debug_count++;
                // if (debug_count == 4) exit(0);
            }
        }



        // SHARED QUEUE
        // #pragma omp parallel
        // {
        //     util::Subset ss_def(ssdata->size);
        //     util::Subset ss_ref(ssdata->size);
        //     optimizer::Parameters local_opt = opt;

        //     std::vector<CorrelationPoint> temp_neighbours;


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

        //         temp_neighbours.clear();

        //         for (int i = 0; i < 4; i++) {
        //             int neigh_x = curr_x + dx[i];
        //             int neigh_y = curr_y + dy[i];

        //             if (is_valid_point(neigh_x, neigh_y, image_roi, px_horizontal, px_vertical, ssdata->size)) {
        //                 int index = neigh_y * px_horizontal + neigh_x;

        //                 if (!computed_mask[index].exchange(true)) {
        //                     util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);
        //                     optimizer::Results neigh_results = optimizer::solve(neigh_x, neigh_y, &ss_def, &ss_ref, &local_opt);
        //                     temp_neighbours.emplace_back(neigh_x, neigh_y, 1.0 - 0.5 * neigh_results.cost);
        //                 }
        //             }
        //         }

        //         // Push new neighbours back to the shared queue
        //         {
        //             std::lock_guard<std::mutex> lock(queue_mutex);
        //             for (const auto& neighbour : temp_neighbours) {
        //                 shared_queue.push(neighbour);
        //             }
        //         }
        //     }
        // }





    }

    inline bool is_valid_point(int ss_x, int ss_y, util::SubsetData *ssdata) {

        int x = ss_x / ssdata-> step;
        int y = ss_y / ssdata-> step;

        int idx = y * ssdata->num_ss_x + x;

        if ((ss_x % ssdata->step) || (ss_y % ssdata->step)){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "Subset step size: " << ssdata->step << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else if (!ssdata->mask[idx]){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "subset mask index: " << idx << std::endl;
            std::cerr << "subset mask value: " << idx << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else return true;

        //auto it = ssdata->coords_to_index.find({x, y});

        //// check if coordinates are in the coordinate list
        //if (it == ssdata->coords_to_index.end()) {
        //    std::cerr << "Error: coordinates not found in the coordinate list." << std::endl;
        //    std::cerr << "Coordinates: " << x << ", " << y << std::endl;
        //    exit(EXIT_FAILURE);
        //}
        //else return true;
    }

}


