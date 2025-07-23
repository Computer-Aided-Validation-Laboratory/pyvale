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
#include "./cursor_control.hpp"
#include "./dicfourier.hpp"

namespace scanmethod {


    // for graceful exit
    std::atomic<bool> stop_request(false);

    void signalHandler(int signal) {
        if (signal == SIGINT) {
            stop_request = true;
        }
    }

    void image(const double *img_ref,
               const Interpolator &interp_def,
               const util::SubsetData &ssdata, 
               const util::Config &conf,
               const int img_num){

        const int num_ss = ssdata.num;
        const int ss_size = ssdata.size;

        // progress bar
        indicators::ProgressBar bar;
        util::create_progress_bar(bar, conf.filenames, img_num, 100);
        std::atomic<int> current_progress = 0;
        std::atomic<int> prev_pct = 0;

               // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter,
                                    conf.precision, conf.opt_threshold,
                                    conf.px_vert, conf.px_hori);

            // if using SSD then not going to use opt_threshold. It can take
            // any value. Convergence will be checked against precision only
            if (conf.corr_crit=="SSD")
                opt.opt_threshold = std::numeric_limits<double>::max();


            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request){
                    continue;
                }

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ssdata.coords[ss*2];
                int ss_y = ssdata.coords[ss*2+1];

                // get the reference subset
                util::extract_ss(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                for (int i = 0; i < opt.num_params; i++){
                    opt.p[i] = 0.0;
                }

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ssdata.size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ssdata.size)/2.0 - 0.5;
                util::Results res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                if (conf.corr_crit!="SSD")
                    res.cost = 1-res.cost;

                // append the results for the current subset to result vectors
                util::append_results(img_num, ss, res, num_ss);

                // update progress bar
                int progress = current_progress.fetch_add(1);
                util::update_progress_bar(bar, progress, num_ss, prev_pct);

            }
        }
        bar.mark_as_completed();
        indicators::show_console_cursor(true);

    }



    void image_with_bf(const double *img_ref,
                    const double *img_def,
                    const Interpolator &interp_def,
                    const util::SubsetData  &ssdata, 
                    const util::Config &conf,
                    const int img_num){

        const int num_ss = ssdata.num;
        const int ss_size = ssdata.size;

        // progress bar
        indicators::ProgressBar bar;
        util::create_progress_bar(bar, conf.filenames, img_num, num_ss);
        std::atomic<int> current_progress = 0;
        std::atomic<int> prev_pct = 0;

        // initialise subsets
        util::Subset ss_def(ss_size);
        util::Subset ss_ref(ss_size);

        // optimization parameters
        optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                conf.precision, conf.opt_threshold,
                                conf.px_vert, conf.px_hori);

        // if using SSD then not going to use opt_threshold. It can take
        // any value. Convergence will be checked against precision only
        if (conf.corr_crit=="SSD")
            opt.opt_threshold = std::numeric_limits<double>::max();


        // brute force scan parameters
        brute::Parameters brute(conf.bf_threshold, conf.max_disp);

        // perform optimization on subset from deformed image
        util::Results res(conf.num_params);

        // counter for each thread
        int ss_thread_num = 0;

        // temp p values for copy from brute force to optimization.
        double ptemp[6] = {0,0,0,0,0,0};

        // loop over subsets within the ROI
        #pragma omp parallel for firstprivate(ss_ref, ss_def, ss_thread_num, opt, brute, res, ptemp)
        for (int ss = 0; ss < num_ss; ss++){

            // exit the main DIC loop when ctrl+C is hit
            if (stop_request){
                continue;
            }


            // subset coordinate list contains central locations.
            // Converting to top left corner for optimization routine
            int ss_x = ssdata.coords[ss*2];
            int ss_y = ssdata.coords[ss*2+1];

            // get the reference subset values from the reference img
            util::extract_ss(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref); 


            // if first subset in the loop or prev subset was a poor match
            // start search with a brute force scan using the last set of 
            // brute force params that gave a good match.
            if ((ss_thread_num == 0) || (res.cost > opt.opt_threshold)){

                brute::expanding_wavefront(ss_x, ss_y, img_ref, 
                                        conf.px_hori, 
                                        conf.px_vert, 
                                        ss_ref, ss_def, brute);

                ptemp[0] = brute.p_rigid[0];
                ptemp[1] = brute.p_rigid[1];

                for (int i = 0; i < opt.num_params; i++){
                    opt.p[i] = ptemp[i];
                }
            }

            double centre_x = ss_x + static_cast<double>(ssdata.size)/2.0 - 0.5;
            double centre_y = ss_y + static_cast<double>(ssdata.size)/2.0 - 0.5;
            util::Results res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);


            // if its not SSD, then we need to flip the cost values so that 1.0
            // is a perfect match rather than 0.0
            if (conf.corr_crit!="SSD")
                res.cost = 1-res.cost;

            // append the results for the current subset to result vectors
            util::append_results(img_num, ss, res, num_ss);

            ss_thread_num++;

            // update progress bar
            int progress = current_progress.fetch_add(1);
            util::update_progress_bar(bar, progress, num_ss, prev_pct);

        }
        bar.mark_as_completed();
        indicators::show_console_cursor(true);
    }






    void reliability_guided(const double *img_ref,
                            const double *img_def,
                            const Interpolator &interp_def,
                            const std::vector<util::SubsetData> &ssdata,
                            const util::Config &conf,
                            const int img_num,
                            const bool save_at_end){

        // assign some consts for readability
        const int px_hori = conf.px_hori;
        const int px_vert = conf.px_vert;
        const int seed_x = conf.rg_seed.first;
        const int seed_y = conf.rg_seed.second;
        const int nsizes = ssdata.size();
        const int last_size = nsizes-1;
        const int num_ss = ssdata[last_size].num;
        const int ss_size = ssdata[last_size].size;
        const int ss_step = ssdata[last_size].step;


        fourier::mgwd(ssdata, img_ref, img_def, interp_def, 
                      conf.fft_mad, conf.fft_mad_scale);

        // progress bar
        indicators::ProgressBar bar;
        util::create_progress_bar(bar, conf.filenames, img_num, num_ss);
        std::atomic<int> current_progress(0);
        std::atomic<int> prev_pct(0);

        // quick check for the initial seed point
        if (!rg::is_valid_point(seed_x, seed_y, ssdata[last_size])) {
            return;
        }

        // Initialize binary mask for computed points (initialized to 0)
        std::vector<std::atomic<int>> computed_mask(ssdata[last_size].mask.size());
        for (auto& val : computed_mask) val.store(0);

        // queue for each thread
        std::vector<std::priority_queue<rg::Point>> local_q(omp_get_max_threads());

        // Mutex vector to protect each queue
        std::vector<std::mutex> queue_mutexes(omp_get_max_threads());

        # pragma omp parallel
        {

            int tid = omp_get_thread_num();
            std::priority_queue<rg::Point>& thread_q = local_q[tid];

            // Initialize ref and def subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // Optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                      conf.precision, conf.opt_threshold, 
                                      px_vert, px_hori);

            // if using SSD then not going to use opt_threshold. It can take
            // any value. Convergence will be checked against precision only
            if (conf.corr_crit=="SSD")
                opt.opt_threshold = std::numeric_limits<double>::max();

            brute::Parameters brute(conf.bf_threshold, conf.max_disp);

            std::vector<std::unique_ptr<fourier::FFT>> fft_windows;

            for (size_t t = 0; t < ssdata.size(); ++t) {
                fft_windows.push_back(std::make_unique<fourier::FFT>(ssdata[t].size));
            }

            // TODO: for the seed location I'm going to overwride the max 
            // number of iterations to make sure we get a good convergence.
            // this is hardcoded for now. Could do with updating so that 
            // the seed location is checked ahead of the main correlation run.
            opt.max_iter = 200;

            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS THE SEED SUBSET 
            // ---------------------------------------------------------------------------------------------------------------------------
            if (tid == 0) {

                // seed coordinates
                int x = seed_x / ss_step;
                int y = seed_y / ss_step;
                int idx = ssdata[last_size].mask[y * ssdata[last_size].num_ss_x + x];


                // if the first image. Take the optimization parameters from rigid fourier
                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                opt.p[0] = fourier::shifts[last_size].x[idx];
                opt.p[1] = fourier::shifts[last_size].y[idx];

                // Extract reference subset and solve for starting seed point
                util::extract_ss(ss_ref, seed_x, seed_y, px_hori, px_vert, img_ref);


                double centre_x = seed_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = seed_y + static_cast<double>(ss_size)/2.0 - 0.5;

                util::Results seed_res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);

                // if its not SSD, then we need to flip the cost values so that 1.0
                // is a perfect match rather than 0.0
                if (conf.corr_crit!="SSD")
                    seed_res.cost = 1.0-seed_res.cost;

                // append the results for the current subset to result vectors
                util::append_results(img_num, idx, seed_res, num_ss);

                computed_mask[idx].store(1);

                // int progress = current_progress.fetch_add(1);
                // util::update_progress_bar(bar, progress, num_ss, prev_pct);

                // loop over the neighbours for the initial seed point
                for (size_t n = 0; n < ssdata[last_size].neigh[idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ssdata[last_size].neigh[idx][n];

                    int nx = ssdata[last_size].coords[nidx*2];
                    int ny = ssdata[last_size].coords[nidx*2+1];

                    util::extract_ss(ss_ref, nx, ny, px_hori, px_vert, img_ref);

                    // get parameter values from fft output or from previous image
                    std::fill(opt.p.begin(), opt.p.end(), 0.0);
                    opt.p[0] = fourier::shifts[last_size].x[nidx];
                    opt.p[1] = fourier::shifts[last_size].y[nidx];

                    // perform optimization for seed point neighbours
                    double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                    double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;
                    util::Results nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);
                    
                    // if its not SSD, then we need to flip the cost values so that 1.0
                    // is a perfect match rather than 0.0
                    if (conf.corr_crit!="SSD")
                        nres.cost = 1.0-nres.cost;

                    // append the results for the current subset to result vectors
                    util::append_results(img_num, nidx, nres, num_ss);

                    // update mask
                    computed_mask[idx].store(1);

                    // add this point to queue
                    // Protect push with mutex
                    {
                        std::lock_guard<std::mutex> lock(queue_mutexes[0]);
                        local_q[0].push(rg::Point(nidx,nres.cost));
                    }

                    // update progress bar
                    // int progress = current_progress.fetch_add(1);
                    // util::update_progress_bar(bar, progress, num_ss, prev_pct);
                }
            }


            // ---------------------------------------------------------------------------------------------------------------------------
            // PROCESS ALL OTHER SUBSETS
            // ---------------------------------------------------------------------------------------------------------------------------
            #pragma omp barrier

            opt.max_iter = conf.max_iter;

            std::vector<rg::Point> temp_neigh;
            temp_neigh.reserve(4);

            const int max_idle_iters = 100;
            rg::Point current(0, 0);

            while (!stop_request) {
                bool got_point = false;
                int idle_iters = 0;

                // Try own queue safely
                {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    if (!thread_q.empty()) {
                        current = thread_q.top();
                        thread_q.pop();
                        got_point = true;
                    }
                }

                // Steal if nothing in own queue
                if (!got_point) {
                    while (!got_point && idle_iters < max_idle_iters) {
                        #pragma omp critical
                        {
                            for (size_t i = 0; i < local_q.size(); ++i) {
                                std::lock_guard<std::mutex> lock(queue_mutexes[i]);
                                if (!local_q[i].empty()) {
                                    current = local_q[i].top();
                                    local_q[i].pop();
                                    got_point = true;
                                    break;
                                }
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


                // index of current point in results arrays
                int idx_results = save_at_end ? img_num * num_ss + current.idx : current.idx;
                int idx_results_p = idx_results * opt.num_params;

                // loop over neighbouring points
                for (size_t n = 0; n < ssdata[last_size].neigh[current.idx].size(); n++) {

                    // subset index of neighbour to the current point
                    int nidx = ssdata[last_size].neigh[current.idx][n];

                    int expected = 0;
                    expected = computed_mask[nidx].exchange(1);
                    if (expected == 0) {

                        // coords of neigh
                        int nx = ssdata[last_size].coords[nidx*2];
                        int ny = ssdata[last_size].coords[nidx*2+1];

                        // extract subset
                        util::extract_ss(ss_ref, nx, ny, px_hori, px_vert, img_ref);

                        // if the neighbouring subset had not met correlation threshold then try values from fft windowing
                        if (util::cost_arr[idx_results] < opt.opt_threshold){
                            std::fill(opt.p.begin(), opt.p.end(), 0.0);
                            opt.p[0] = fourier::shifts[last_size].x[nidx];
                            opt.p[1] = fourier::shifts[last_size].y[nidx];
                        }
                        else {
                            for (int i = 0; i < opt.num_params; i++){
                                opt.p[i] = util::p_arr[idx_results_p+i];
                            }
                        }

                        // optimize
                        double centre_x = nx + static_cast<double>(ss_size)/2.0 - 0.5;
                        double centre_y = ny + static_cast<double>(ss_size)/2.0 - 0.5;
                        util::Results nres = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);


                        // if its not SSD, then we need to flip the cost values so that 1.0
                        // is a perfect match rather than 0.0
                        if (conf.corr_crit!="SSD")
                            nres.cost = 1.0-nres.cost;

                        // append results
                        #pragma omp critical
                            util::append_results(img_num, nidx, nres, num_ss);

                        // add results to temp neighbour results
                        temp_neigh.emplace_back(nidx, nres.cost);

                        // update progress bar
                        int progress = current_progress.fetch_add(1);
                        util::update_progress_bar(bar, progress, num_ss, prev_pct);

                    }
                }

                for (const auto& neigh : temp_neigh) {
                    std::lock_guard<std::mutex> lock(queue_mutexes[tid]);
                    thread_q.push(neigh);
                }
            }
        }
        bar.mark_as_completed();
        indicators::show_console_cursor(true);

    }


    void multi_window_fourier(const double *img_ref,
                              const double *img_def,
                              const Interpolator &interp_def,
                              const std::vector<util::SubsetData> &ssdata,
                              const util::Config &conf,
                              const int img_num){

        // for the first image perform the FFT windowing. later images will be
        // seeded with previous images
        fourier::mgwd(ssdata, img_ref, img_def, interp_def, 
                      conf.fft_mad, conf.fft_mad_scale);

        const int nsizes = ssdata.size();
        const int last_size = nsizes-1;

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = ssdata[last_size].num;
        const int ss_size = ssdata[last_size].size;

        // progress bar
        indicators::ProgressBar bar;
        util::create_progress_bar(bar, conf.filenames, img_num, num_ss);
        std::atomic<int> current_progress = 0;
        std::atomic<int> prev_pct = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                    conf.precision, conf.opt_threshold,
                                    conf.px_vert, conf.px_hori);
            

            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request){
                    continue;
                }

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ssdata[last_size].coords[ss*2];
                int ss_y = ssdata[last_size].coords[ss*2+1];

                // get the reference subset
                util::extract_ss(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                opt.p[0] = fourier::shifts[last_size].x[ss];
                opt.p[1] = fourier::shifts[last_size].y[ss];

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ss_size)/2.0 - 0.5;
                util::Results res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);
                

                // if its not SSD, then we need to flip the cost values so that 1.0
                // is a perfect match rather than 0.0
                if (conf.corr_crit!="SSD")
                    res.cost = 1.0-res.cost;

                // append optimization results to results vectors
                #pragma omp critical
                    util::append_results(img_num, ss, res, num_ss);

                // update progress bar
                int progress = current_progress.fetch_add(1);
                util::update_progress_bar(bar, progress, num_ss, prev_pct);

            }
        }
        bar.mark_as_completed();
        indicators::show_console_cursor(true);
    }


    void single_window_fourier(const double *img_ref,
                              const double *img_def,
                              const Interpolator &interp_def,
                              const util::SubsetData &ssdata,
                              const util::Config &conf,
                              const int img_num){

        // for the first image perform the FFT windowing. later images will be
        // seeded with previous images
        fourier::sgwd(ssdata, 256, img_ref, img_def, interp_def);

        // get number of subsets and the size for the smalllest window size
        const int num_ss  = ssdata.num;
        const int ss_size = ssdata.size;

        // progress bar
        indicators::ProgressBar bar;
        util::create_progress_bar(bar, conf.filenames, img_num, num_ss);
        std::atomic<int> current_progress = 0;
        std::atomic<int> prev_pct = 0;

        // loop over subsets within the ROI
        #pragma omp parallel shared(stop_request)
        {

            // initialise subsets
            util::Subset ss_def(ss_size);
            util::Subset ss_ref(ss_size);

            // optimization parameters
            optimizer::Parameters opt(conf.num_params, conf.max_iter, 
                                    conf.precision, conf.opt_threshold,
                                    conf.px_vert, conf.px_hori);


            #pragma omp for
            for (int ss = 0; ss < num_ss; ss++){

                // exit the main DIC loop when ctrl+C is hit
                if (stop_request){
                    continue;
                }

                // subset coordinate list takes central locations. 
                // Converting to top left corner for optimization routine
                int ss_x = ssdata.coords[ss*2];
                int ss_y = ssdata.coords[ss*2+1];

                // get the reference subset
                util::extract_ss(ss_ref, ss_x, ss_y, conf.px_hori, conf.px_vert, img_ref);

                std::fill(opt.p.begin(), opt.p.end(), 0.0);
                opt.p[0] = fourier::shifts[0].x[ss];
                opt.p[1] = fourier::shifts[0].y[ss];

                // perform optimization on subset from deformed image
                double centre_x = ss_x + static_cast<double>(ss_size)/2.0 - 0.5;
                double centre_y = ss_y + static_cast<double>(ss_size)/2.0 - 0.5;
                util::Results res = optimizer::solve(centre_x, centre_y, ss_ref, ss_def, interp_def, opt, conf.corr_crit);
                
                // if its not SSD, then we need to flip the cost values so that 1.0
                // is a perfect match rather than 0.0
                if (conf.corr_crit!="SSD")
                    res.cost = 1.0-res.cost;

                // append optimization results to results vectors
                util::append_results(img_num, ss, res, num_ss);

                // update progress bar
                int progress = current_progress.fetch_add(1);
                util::update_progress_bar(bar, progress, num_ss, prev_pct);

            }
        }
        bar.mark_as_completed();
        indicators::show_console_cursor(true);
    }

}
