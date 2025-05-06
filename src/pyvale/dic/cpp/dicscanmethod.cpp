// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <queue>
#include <atomic>
#include <thread>
#include <cstring>
#include <omp.h>

// Program Header files
#include "./dicbruteforce.hpp"
#include "./dicoptimizer.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicrg.hpp"


namespace scanmethod {



    void image(double *image_ref, 
                    double *image_def, 
                    bool *image_roi,
                    util::SubsetData &ssdata, 
                    util::Config &conf,
                    int img_num){
    
        // initialise subsets
        util::Subset ss_def(ssdata.size);
        util::Subset ss_ref(ssdata.size);
    
        // optimization parameters
        optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                  conf.precision, conf.threshold_lm,
                                  conf.px_vertical, conf.px_horizontal);
    
        // loop over subsets within the ROI
        #pragma omp parallel for firstprivate(ss_def, ss_ref, opt)
        for (int ss = 0; ss < ssdata.num; ss++){
    
            // subset coordinate list takes central locations. 
            // Converting to top left corner for optimization routine
            int ss_x = ssdata.coords[ss*2];
            int ss_y = ssdata.coords[ss*2+1];
    
            // get the deformed subset
            util::extract_ss(ss_def, ss_x, ss_y, 
                             conf.px_horizontal,
                             conf.px_vertical,
                             image_def); 
    
    
            // perform optimization on subset from deformed image
            optimizer::Results results;
            results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);
    
    
            // append the results for the current subset to result vectors
            util::append_results(conf.num_def_images, img_num, ssdata.num, ss, 
                                 results.iter, results.ftol, results.xtol, 
                                 results.u, results.v, results.cost, results.p);
    
        }
    }



void image_with_bf(double *image_ref, 
                        double *image_def, 
                        bool *image_roi,
                        util::SubsetData &ssdata, 
                        util::Config &conf,
                        int img_num){

    // initialise subsets
    util::Subset ss_def(ssdata.size);
    util::Subset ss_ref(ssdata.size);

    // optimization parameters
    optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                              conf.precision, conf.threshold_lm,
                              conf.px_vertical, conf.px_horizontal);


    // brute force scan parameters
    brute::Parameters brute(conf.threshold_bf, conf.range_bf);

    // perform optimization on subset from deformed image
    optimizer::Results results;
    results.iter = 0;

    // counter for each thread
    int ss_thread_num = 0;      

    // temp p values for copy from brute force to optimization.
    double ptemp[6] = {0,0,0,0,0,0};


    // loop over subsets within the ROI
    #pragma omp parallel for firstprivate(ss_def, ss_ref, ss_thread_num, opt, brute, results)
    for (int ss = 0; ss < ssdata.num; ss++){

        // subset coordinate list contains central locations.
        // Converting to top left corner for optimization routine
        int ss_x = ssdata.coords[ss*2];
        int ss_y = ssdata.coords[ss*2+1];

        // get the deformed subset
        util::extract_ss(ss_def, ss_x, ss_y, 
                         conf.px_horizontal,
                         conf.px_vertical,
                             image_def); 


        // if first subset in the loop or prev subset was a poor match
        // start search with a brute force scan using the last set of 
        // brute force params that gave a good match.
        if ((ss_thread_num == 0) || (results.iter == opt.max_iter)){

            brute::expanding_wavefront(ss_x, ss_y, image_ref, 
                                       conf.px_vertical, 
                                       conf.px_horizontal, 
                                       ss_def, ss_ref, brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }
        }

        results = optimizer::solve(ss_x, ss_y, &ss_def, &ss_ref, &opt);

        // append the results for the current subset to result vectors
        util::append_results(conf.num_def_images, img_num, ssdata.num, ss, 
                             results.iter, results.ftol, results.xtol,
                             results.u, results.v, results.cost, results.p);

        ss_thread_num++;

    }
}






    void reliability_guided(double *image_ref, 
                            double *image_def, 
                            bool *image_roi,
                            util::SubsetData &ssdata, 
                            util::Config &conf,
                            int img_num){

         int seed_x = 500; // in corner coordinates
         int seed_y = 500; // in corner coodinates

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ssdata)) {
            return;
        }


        // assign some consts for readability
        const int px_horizontal = conf.px_horizontal;
        const int px_vertical = conf.px_vertical;

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<bool>> computed_mask(ssdata.mask.size());
        for (size_t i = 0; i < computed_mask.size(); ++i) {
            computed_mask[i] = false;
        }

        // queue for each thread
        int max_threads = omp_get_max_threads();
        std::vector<std::priority_queue<rg::Point>> local_queues(max_threads);

        // Initialize ref and def subsets
        util::Subset ss_def(ssdata.size);
        util::Subset ss_ref(ssdata.size);

        // Extract subset and solve for starting seed point
        util::extract_ss(ss_def, seed_x, seed_y, 
                         px_horizontal, px_vertical,
                         image_def);



        // temp p values for copy from brute force to optimization.
        double ptemp[6] = {0,0,0,0,0,0};

        // brute force for initial subset
        brute::Parameters brute(conf.threshold_bf, 
                                conf.range_bf);

        brute::expanding_wavefront(seed_x, seed_y, image_ref, 
                                   px_vertical, px_horizontal, 
                                   ss_def, ss_ref, brute);

        // Optimization parameters
        optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                  conf.precision, conf.threshold_lm, 
                                  px_vertical, px_horizontal);

        opt.p[0] = brute.p_rigid[0];
        opt.p[1] = brute.p_rigid[1];

        for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
        }

        optimizer::Results seed_results = optimizer::solve(seed_x, seed_y, 
                                                           &ss_def, &ss_ref, &opt);



        //mark seed as computed
        int idx = ssdata.coords_to_idx.find({seed_x, seed_y})->second;

        util::append_results(conf.num_def_images, img_num, 
                             ssdata.num, idx, seed_results.iter, 
                             seed_results.ftol, seed_results.xtol, 
                             seed_results.u, seed_results.v, 
                             seed_results.cost, seed_results.p);

        computed_mask[idx] = true;

        // loop over seed neighbours
        for (int nidx : ssdata.neighbours[idx]) {

            int nx = ssdata.coords[nidx*2];
            int ny = ssdata.coords[nidx*2+1];

            util::extract_ss(ss_def, nx, ny,
                             px_horizontal, px_vertical,
                             image_def);

            brute::expanding_wavefront(nx, ny, image_ref, 
                                       px_vertical, px_horizontal, 
                                       ss_def, ss_ref, brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }

            optimizer::Results nresults = optimizer::solve(nx, ny,
                                                                &ss_def, &ss_ref, &opt);

            util::append_results(conf.num_def_images, img_num, 
                                 ssdata.num, nidx, nresults.iter, 
                                 nresults.ftol, nresults.xtol, 
                                 nresults.u, nresults.v, 
                                 nresults.cost, nresults.p);

            // Add to priority queue
            computed_mask[nidx] = true;

            // LOCAL QUEUE
            local_queues[0].push(rg::Point(nx, ny, 1.0 - 0.5*nresults.cost));
        }

        // LOCAL QUEUE PROCESSING
        #pragma omp parallel firstprivate(ss_def, ss_ref, opt, brute)
        {

            auto& my_queue = local_queues[omp_get_thread_num()];
            std::vector<rg::Point> temp_neighbours;
            temp_neighbours.reserve(4);
            double ptemp[6] = {0,0,0,0,0,0};

            const int max_idle_iters = 100;
            rg::Point current(0, 0, 0);

            while (true) {

                // reset correlation values and got_point
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
                        for (size_t i = 0; i < local_queues.size(); ++i) {
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

                // LIST subset
                int curr_x = current.x;
                int curr_y = current.y;


                // LIST loop
                int idx = ssdata.coords_to_idx.find({curr_x, curr_y})->second;
                int idx_results = (img_num * ssdata.num + idx);
                int idx_resultsp = idx_results*opt.num_params;

                for (int nidx : ssdata.neighbours[idx]) {

                    // LIST get coordiantes
                    int nx = ssdata.coords[nidx*2];
                    int ny = ssdata.coords[nidx*2+1];

                    // LIST
                    if (!computed_mask[nidx].exchange(true)) {

                        // extract subset
                        util::extract_ss(ss_def, nx, ny,
                                         px_horizontal, px_vertical,
                                         image_def);

                        // if the neighbouring subset reached the maximum number of iterations, try again from a brute force
                        if (util::niter_arr[idx] == opt.max_iter && util::cost_arr[idx] > opt.threshold_lm){

                            brute::expanding_wavefront(nx, ny,
                                                       image_ref, px_vertical, 
                                                       px_horizontal, ss_def, 
                                                       ss_ref, brute);

                            ptemp[0] = brute.p_rigid[0];
                            ptemp[1] = brute.p_rigid[1];

                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = ptemp[i];
                            }

                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = util::p_arr[idx_resultsp+i];
                            }
                        }

                        // optimize
                        optimizer::Results nresults = optimizer::solve(nx, ny, &ss_def, &ss_ref, &opt);

                        // append results
                        util::append_results(conf.num_def_images, img_num, 
                                             ssdata.num, nidx, 
                                             nresults.iter, 
                                             nresults.ftol, 
                                             nresults.xtol, 
                                             nresults.u, 
                                             nresults.v, 
                                             nresults.cost, 
                                             nresults.p);

                        // // add results to temp neighbour results
                        temp_neighbours.emplace_back(nx, ny, 1.0 - 0.5 * nresults.cost);
                    }
                }

                for (const auto& neighbour : temp_neighbours) {
                    my_queue.push(neighbour);
                }
            }
        }



    }


}
