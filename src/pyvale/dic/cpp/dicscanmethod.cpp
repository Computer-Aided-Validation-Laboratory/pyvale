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
#include <csignal>

// Program Header files
#include "./dicbruteforce.hpp"
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicrg.hpp"
#include "./indicators.hpp"
#include "./dicfourier.hpp"

namespace scanmethod {


    // for graceful exit
    volatile bool stop_request(false);


    void reset_indicators(indicators::ProgressBar &bar,
                          int img_num, int num_ss){
        //Hide cursor
        indicators::show_console_cursor(false);
        bar.set_option(indicators::option::BarWidth{50});
        bar.set_option(indicators::option::Start{" ["});
        bar.set_option(indicators::option::Fill{"#"});
        bar.set_option(indicators::option::Lead{"#"});
        bar.set_option(indicators::option::Remainder{"-"});
        bar.set_option(indicators::option::End{"]"});
        bar.set_option(indicators::option::PrefixText{"Deformed Image " + std::to_string(img_num)});
        bar.set_option(indicators::option::ShowPercentage{true});
        bar.set_option(indicators::option::ShowElapsedTime{true});
        bar.set_option(indicators::option::FontStyles{
          std::vector<indicators::FontStyle>{indicators::FontStyle::bold}});

    }

    void update_bar(indicators::ProgressBar &bar, int i, int num_ss, int &prev_pct) {

        int curr_pct = static_cast<float>(i) / static_cast<float>(num_ss) * 100;
        if (curr_pct > prev_pct){
            bar.set_progress(curr_pct);
        }
        prev_pct = curr_pct;
    }



    void signalHandler(int signal) {
        if (signal == SIGINT) {
            stop_request = true;
        }
    }

    void image(const Interpolator &interp_ref,
               const double *img_def,
               const util::SubsetData &ssdata, 
               const util::Config &conf,
               const int img_num){

        const int ss_num = ssdata.num;
        const int ss_size = ssdata.size;

        // progress bar
        indicators::ProgressBar bar;
        reset_indicators(bar, img_num, ss_num);
        int current_progress = 0;
        int prev_pct = 0;

        // initialise subsets
        util::Subset ss_def(ss_size);
        util::Subset ss_ref(ss_size);

        // optimization parameters
        optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                  conf.precision, conf.threshold_lm,
                                  conf.px_vert, conf.px_hori);

        // loop over subsets within the ROI
        #pragma omp parallel for firstprivate(ss_def, ss_ref, opt) shared(stop_request)
        for (int ss = 0; ss < ss_num; ss++){

            // exit the main DIC loop when ctrl+C is hit
            if (stop_request){
                continue;
            }

            // subset coordinate list takes central locations. 
            // Converting to top left corner for optimization routine
            int ss_x = ssdata.coords[ss*2];
            int ss_y = ssdata.coords[ss*2+1];

            // get the deformed subset
            util::extract_ss(ss_def, ss_x, ss_y, conf.px_hori, conf.px_vert, img_def);

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = 0.0;
            }

            // perform optimization on subset from deformed image
            double centre_x = ss_x + static_cast<double>(ssdata.size)/2.0 - 0.5;
            double centre_y = ss_y + static_cast<double>(ssdata.size)/2.0 - 0.5;
            util::Results res = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);


            // append the results for the current subset to result vectors
            util::append_results(img_num, ss, res, ss_num);

            // update progress bar
            #pragma omp critical
            {
                update_bar(bar, current_progress, ss_num, prev_pct);
                current_progress++;
            }

        }

        bar.mark_as_completed();
    }



void image_with_bf(const Interpolator &interp_ref, 
                   const double *img_ref,
                   const double *img_def,
                   const util::SubsetData  &ssdata, 
                   const util::Config &conf,
                   const int img_num){

    const int ss_num = ssdata.num;
    const int ss_size = ssdata.size;

    // progress bar
    indicators::ProgressBar bar;
    reset_indicators(bar, img_num, ss_num);
    int current_progress = 0;
    int prev_pct = 0;

    // initialise subsets
    util::Subset ss_def(ss_size);
    util::Subset ss_ref(ss_size);

    // optimization parameters
    optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                              conf.precision, conf.threshold_lm,
                              conf.px_vert, conf.px_hori);


    // brute force scan parameters
    brute::Parameters brute(conf.threshold_bf, conf.range_bf);

    // perform optimization on subset from deformed image
    util::Results res(conf.num_params);

    // counter for each thread
    int ss_thread_num = 0;

    // temp p values for copy from brute force to optimization.
    double ptemp[6] = {0,0,0,0,0,0};

    // loop over subsets within the ROI
    #pragma omp parallel for firstprivate(ss_def, ss_ref, ss_thread_num, opt, brute, res, ptemp)
    for (int ss = 0; ss < ss_num; ss++){

        // exit the main DIC loop when ctrl+C is hit
        if (stop_request){
            continue;
        }


        // subset coordinate list contains central locations.
        // Converting to top left corner for optimization routine
        int ss_x = ssdata.coords[ss*2];
        int ss_y = ssdata.coords[ss*2+1];

        // get the deformed subset
        util::extract_ss(ss_def, ss_x, ss_y, conf.px_hori, conf.px_vert, img_def); 


        // if first subset in the loop or prev subset was a poor match
        // start search with a brute force scan using the last set of 
        // brute force params that gave a good match.
        if ((ss_thread_num == 0) || (res.cost > opt.threshold_lm)){

            brute::expanding_wavefront(ss_x, ss_y, img_ref, 
                                       conf.px_hori, 
                                       conf.px_vert, 
                                       ss_def, ss_ref, brute);

            ptemp[0] = brute.p_rigid[0];
            ptemp[1] = brute.p_rigid[1];

            for (int i = 0; i < opt.num_params; i++){
                opt.p[i] = ptemp[i];
            }
        }

        double centre_x = ss_x + static_cast<double>(ssdata.size)/2.0 - 0.5;
        double centre_y = ss_y + static_cast<double>(ssdata.size)/2.0 - 0.5;
        util::Results res = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);

        // append the results for the current subset to result vectors
        util::append_results(img_num, ss, res, ss_num);

        ss_thread_num++;

        // update progress bar
        #pragma omp critical 
        {
            update_bar(bar, current_progress, ss_num, prev_pct);
            current_progress++;
        }

    }
    bar.mark_as_completed();
}






    void reliability_guided(const Interpolator &interp_ref,
                            const double *img_ref,
                            const double *img_def,
                            const util::SubsetData &ssdata,
                            const util::Config &conf,
                            const int img_num){
        
        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int ss_num = ssdata.num;
        const int ss_size = ssdata.size;
        const int seed_x = conf.rg_seed.first;
        const int seed_y = conf.rg_seed.second;

        // progress bar
        indicators::ProgressBar bar;
        reset_indicators(bar, img_num, ss_num);
        int current_progress = 0;
        int prev_pct = 0;

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ssdata)) {
            return;
        }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<bool>> computed_mask(ssdata.mask.size());
        for (size_t i = 0; i < computed_mask.size(); ++i) {
            computed_mask[i] = false;
        }

        // queue for each thread
        std::vector<std::priority_queue<rg::Point>> local_q(omp_get_max_threads());

        // initialise the fft search
        int largest_fft_window = rg::next_pow2(conf.range_bf);
        std::vector<int> windows = rg::pow2_between(largest_fft_window, conf.ss_size.back());

        # pragma omp parallel 
        {
            // Initialize ref and def subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // Optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, conf.precision, conf.threshold_lm, px_vert, px_hori);

            brute::Parameters brute(conf.threshold_bf, conf.range_bf);

            // I now need an array of ssdatas for each window size.
            //std::vector<util::SubsetData> window_data;
            //std::vector<util::Subset> fft_window_ref;
            //std::vector<util::Subset> fft_window_def;

            //// Reserve to be safe
            //fft_window_ref.reserve(windows.size());
            //fft_window_def.reserve(windows.size());

            //for (size_t t = 0; t < windows.size(); t++){
            //    fft_window_ref.emplace_back(util::Subset(windows[t]));
            //    fft_window_def.emplace_back(util::Subset(windows[t]));
            //}

            std::vector<std::unique_ptr<fourier::FFT>> fft_windows;

            for (size_t t = 0; t < windows.size(); ++t) {
                fft_windows.push_back(std::make_unique<fourier::FFT>(windows[t]));
            }

            double prev_x = 0.0;
            double prev_y = 0.0;
            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (omp_get_thread_num() == 0) {

                double shift_x, shift_y;
                rg::get_rigid_shift(shift_x, shift_y, seed_x, seed_y, fft_windows, interp_ref, img_def);

                opt.p[0] = -shift_x;
                opt.p[1] = -shift_y;

                // Extract subset and solve for starting seed point
                util::extract_ss(ss_def, seed_x, seed_y, px_hori, px_vert, img_def);

                // brute force for initial subset
                //brute::expanding_wavefront(seed_x, seed_y, img_ref, px_hori, px_vert, ss_def, ss_ref, brute);
                //opt.p[0] = brute.p_rigid[0];
                //opt.p[1] = brute.p_rigid[1];


                double centre_x = seed_x + static_cast<double>(ssdata.size)/2.0 - 0.5;
                double centre_y = seed_y + static_cast<double>(ssdata.size)/2.0 - 0.5;
                util::Results seed_res = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);

                // seed coordinates
                int x = seed_x / ssdata.step;
                int y = seed_y / ssdata.step;
                int idx = ssdata.mask[y * ssdata.num_ss_x + x];

                // append the results for the current subset to result vectors
                util::append_results(img_num, idx, seed_res, ss_num);

                computed_mask[idx] = true;

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ssdata.neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ssdata.neigh[idx][n];

                    int nx = ssdata.coords[nidx*2];
                    int ny = ssdata.coords[nidx*2+1];

                    util::extract_ss(ss_def, nx, ny, px_hori, px_vert, img_def);

                    double shift_x, shift_y;
                    rg::get_rigid_shift(shift_x, shift_y, nx, ny, fft_windows, interp_ref, img_def);
                    // replace brute force with fft approach
                    //brute::expanding_wavefront(nx, ny, img_ref, px_hori, px_vert, ss_def, ss_ref, brute);

                    double ptemp[6] = {0,0,0,0,0,0};
                    ptemp[0] = -shift_x; //brute.p_rigid[0];
                    ptemp[1] = -shift_y; //brute.p_rigid[1];

                    for (int i = 0; i < opt.num_params; i++){
                        opt.p[i] = ptemp[i];
                    }

                    // perform optimization for seed point neighbours
                    double centre_x = nx + static_cast<double>(ssdata.size)/2.0 - 0.5;
                    double centre_y = ny + static_cast<double>(ssdata.size)/2.0 - 0.5;
                    util::Results nres = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);

                    // append the results for the current subset to result vectors
                    util::append_results(img_num, nidx, nres, ss_num);

                    // update mask
                    computed_mask[nidx] = true;

                    // add this point to queue
                    local_q[0].push(rg::Point(nidx,1.0-0.5*nres.cost));
                }
            }


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            std::priority_queue<rg::Point>& thread_q = local_q[omp_get_thread_num()];
            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);
            double ptemp[6] = {0,0,0,0,0,0};

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {

                // reset correlation values and got_point
                bool got_point = false;
                int idle_iters = 0;

                // Try threads own queue
                if (!thread_q.empty()) {
                    current = thread_q.top();
                    thread_q.pop();
                    got_point = true;
                } 
                else {
                    // Try to steal from top of other local queues
                    while (!got_point && idle_iters < max_idle_iters) {
                        for (size_t i = 0; i < local_q.size(); ++i) {
                            if (!local_q[i].empty()) {
                                current = local_q[i].top();
                                local_q[i].pop();
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
                    break;
                }

                temp_neigh.clear();

                // get the mask idx of point in subset list
                int idx_res = (img_num * ss_num + current.idx);
                int idx_resp = idx_res*opt.num_params;

                // loop over neighbouring points
                for (size_t n = 0; n < ssdata.neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ssdata.neigh[current.idx][n];

                    // coords of neigh
                    int nx = ssdata.coords[nidx*2];
                    int ny = ssdata.coords[nidx*2+1];

                    if (!computed_mask[nidx].exchange(true)) {

                        // extract subset
                        util::extract_ss(ss_def, nx, ny, px_hori, px_vert, img_def);

                        // if the neighbouring subset had not met correlation threshold
                        // then start brute force again
                        if (util::cost_arr[idx_res] > opt.threshold_lm){

                            double shift_x, shift_y;
                            rg::get_rigid_shift(shift_x, shift_y, nx, ny, fft_windows, interp_ref, img_def);

                            ptemp[0] = -shift_x;
                            ptemp[1] = -shift_y;

                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = ptemp[i];
                            }

                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = util::p_arr[idx_resp+i];
                            }
                        }

                        // optimize
                        double centre_x = nx + static_cast<double>(ssdata.size)/2.0 - 0.5;
                        double centre_y = ny + static_cast<double>(ssdata.size)/2.0 - 0.5;
                        util::Results nres = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);

                        // append results
                        util::append_results(img_num, nidx, nres, ss_num);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, 1.0-0.5*nres.cost);

                        // update progress bar
                        #pragma omp critical 
                        {
                            update_bar(bar, current_progress, ss_num, prev_pct);
                            current_progress++;
                        }

                    }
                }

                for (const auto& neigh : temp_neigh) {
                    thread_q.push(neigh);
                }
            }
        }
        bar.mark_as_completed();
    }


    void multi_window_fourier(const Interpolator &interp_ref, 
                              const double *img_ref,
                              const double *img_def, 
                              const std::vector<util::SubsetData> &ssdata,
                              const util::Config &conf,
                              const int img_num){
    
        fourier::mgwd(ssdata, interp_ref, img_ref, img_def, conf.px_hori, conf.px_vert);

        const int nsizes = ssdata.size();
        const int last_size = nsizes-1;

        // get number of subsets and the size for the smalllest window size
        const int ss_num  = ssdata[last_size].num;
        const int ss_size = ssdata[last_size].size;

        // progress bar
        indicators::ProgressBar bar;
        reset_indicators(bar, img_num, ss_num);
        int current_progress = 0;
        int prev_pct = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                    conf.precision, conf.threshold_lm,
                                    conf.px_vert, conf.px_hori);

            double ptemp[6] = {0,0,0,0,0,0};

            #pragma omp for
            for (int ss = 0; ss < ss_num; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request){
                    continue;
                }

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ssdata[last_size].coords[ss*2];
                int ss_y = ssdata[last_size].coords[ss*2+1];

                // get the deformed subset
                util::extract_ss(ss_def, ss_x, ss_y, 
                                conf.px_hori,
                                conf.px_vert,
                                img_def); 

                ptemp[0] = -fourier::shifts[last_size].x[ss];
                ptemp[1] = -fourier::shifts[last_size].y[ss];
                
                //#pragma omp critical
                //{
                //    std::cout << ss_x << " " << ss_y << " ";
                //    std::cout << ptemp[0] << " " << ptemp[1] << std::endl;
                //}

                for (int i = 0; i < opt.num_params; i++){
                    opt.p[i] = ptemp[i];
                }

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ss_size)/2.0 - 0.5;
                util::Results res = optimizer::solve(centre_x, centre_y, ss_def, ss_ref, interp_ref, opt);

                // append optimization results to results vectors
                util::append_results(img_num, ss, res, ss_num);
                //exit(0);

                // update progress bar
                #pragma omp critical
                {
                    update_bar(bar, current_progress, ss_num, prev_pct);
                    current_progress++;
                }

            }
        }
        bar.mark_as_completed();
    }




}
