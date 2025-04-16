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
        const int num_def_images,
        const int img_num,
        const int ss_size,
        const int max_iter,
        const double precision,
        const double threshold_lm,
        const double threshold_bf,
        const double range_bf,
        const std::string &shape_func) {
        
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
        optimizer::Parameters opt(0, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);
        if (shape_func == "affine") opt = optimizer::Parameters(6, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);
        else if (shape_func == "rigid") opt = optimizer::Parameters(2, max_iter, precision, threshold_lm, image_def->px_vertical, image_def->px_horizontal);
        
        // Initialize shared priority queue for all threads
        std::priority_queue<CorrelationPoint> point_queue;
        
        // quick check for the initial seed point
        if (!is_valid_point(seed_x, seed_y, image_roi, px_horizontal, px_vertical, ss_size)) {
            return;
        }

        int ss_x = seed_x - ss_size / 2;
        int ss_y = seed_y - ss_size / 2;
        // std::cout << ss_x << " " << ss_y << std::endl;

        // Initialize subsets
        util::Subset ss_def(ss_size);
        util::Subset ss_ref(ss_size);
        
        // Extract subset and solve for starting seed point
        util::extract_ss(ss_x, ss_y, image_def, &ss_def);
        optimizer::Results seed_results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);

                
        //mark seed as computed
        int index = ss_y * px_horizontal + ss_x;
        computed_mask[index] = true;                     

        
        // Append results for the seed point
        // util::append_results(num_def_images, img_num, 0, &seed_results);
        
        // The four neighboring points
        const int dx[4] = {10, 0, -10, 0};  // Right, Up, Left, Down
        const int dy[4] = {0, 10, 0, -10};
        
        // temp p values for copy from brute force to optimization.
        double ptemp[6] = {0,0,0,0,0,0};


        for (int i = 0; i < 4; i++) {

            int neigh_x = ss_x + dx[i];
            int neigh_y = ss_y + dy[i];
            
            // quick check to see if neighbours are within image bounds.
            if (is_valid_point(neigh_x, neigh_y, image_roi, px_horizontal, px_vertical, ss_size)) {


                util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);

                brute::Parameters brute(threshold_bf, range_bf);
                brute::expanding_wavefront(ss_x, ss_y, image_ref, image_def->px_vertical, image_def->px_horizontal, &ss_def, &ss_ref, &brute);
                // std::cout << brute.p_rigid[0] << " " << brute.p_rigid[1] << std::endl;

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
                
                // Add to priority queue
                computed_mask[neigh_y * px_horizontal + neigh_x] = true;

                // LOCAL QUEUE
                local_queues[0].push(CorrelationPoint(neigh_x, neigh_y, 1 - 0.5 * neigh_results.cost));

                // SHARED QUEUE
                // shared_queue.push(CorrelationPoint(neigh_x, neigh_y, 1 - 0.5 * neigh_results.cost));
                
                // Mark as computed
                // std::cout << "idx: " << neigh_y * px_horizontal + neigh_x << std::endl;
                // std::cout << "processing condition: " << computed_mask[neigh_y * px_horizontal + neigh_x] << std::endl;                        

            }
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

                int curr_x = current.x;
                int curr_y = current.y;

                temp_neighbors.clear();

                for (int i = 0; i < 4; i++) {
                    int neigh_x = curr_x + dx[i];
                    int neigh_y = curr_y + dy[i];

                    if (is_valid_point(neigh_x, neigh_y, image_roi, px_horizontal, px_vertical, ss_size)) {
                        if (!computed_mask[neigh_y * px_horizontal + neigh_x].exchange(true)) {
                            util::extract_ss(neigh_x, neigh_y, image_def, &ss_def);
                            // std::cout << "thread : " << omp_get_thread_num() << ", queue size: " << my_queue.size() << " ";
                            optimizer::Results neigh_results = optimizer::solve(neigh_x, neigh_y, &ss_def, &ss_ref, &opt);
                            temp_neighbors.emplace_back(neigh_x, neigh_y, 1.0 - 0.5 * neigh_results.cost);
                        }
                    }
                }

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

    inline bool is_valid_point(int x, int y, const bool *image_roi, int px_horizontal, int px_vertical, int ss_size) {
        return (x >= ss_size/2 && y >= ss_size/2 && x < px_horizontal - ss_size/2 && y < px_vertical - ss_size/2 && image_roi[y * px_horizontal + x]) ;
    }

}


