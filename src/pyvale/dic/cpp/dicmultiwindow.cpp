// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include "dicutil.hpp"
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <csignal>

// Common Header files
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"

// DIC Header files
#include "./dicmultiwindow.hpp"
#include "./dicfourier.hpp"
#include "./dicsubset.hpp"
#include "./dicinterp.hpp"


void multiwindow_init(std::vector<WindowLevel> &level, 
                      const bool *img_roi, 
                      const util::Config &conf){

    // timer for the initialisation
    //Timer timer("entire FFT initislisation");
    
    std::vector<int> ss_sizes;
    std::vector<int> ss_steps;

    int power = util::next_pow2(conf.max_disp);
    while (power > conf.ss_size) {
        ss_sizes.push_back(power);
        ss_steps.push_back(power / 2);
        power /= 2;
    }
    ss_sizes.push_back(conf.ss_size);
    ss_steps.push_back(conf.ss_step);

    for (size_t lvl = 0; lvl < ss_sizes.size(); lvl++) {

        const bool is_last = (lvl == ss_sizes.size() - 1);
        const subset::Grid *prev = (lvl > 0) ? &level[lvl-1].layout : nullptr;

        level.emplace_back(img_roi, 
                           ss_steps[lvl], ss_sizes[lvl],
                           conf.px_hori, conf.px_vert, 
                           !is_last, lvl, 
                           conf.fft_mad, conf.fft_mad_scale,
                           prev);

    }
}

void WindowLevel::gen_neighlist(const subset::Grid &layout_prev) {

    //Timer timer("nearest neighbour collection for :");

    const int prev_step = layout_prev.step;


    // a list containing the number of neighbours from the previous
    // window size for each subset in the current window size
    num_neigh_list.resize(layout.num);

    // we know the neigh_list is going to be a max size of
    // max_neigh*num_ss. we can resize this later once populated
    neigh_list.resize(max_num_neigh*layout.num);

    // For each subset, find 4 nearest neighbours in layout_prev
    #pragma omp parallel for
    for (int ss = 0; ss < layout.num; ++ss) {

        // corner of subset
        const int ss_x = layout.coords[2*ss];
        const int ss_y = layout.coords[2*ss+1];

        // Vector to store pairs of (distance, index)
        std::vector<std::pair<double, int>> dist_index_list;

        // loop over a 10x10 section from the previous window
        int idx_x = (ss_x / prev_step);
        int idx_y = (ss_y / prev_step);

        // range of neighbour search
        int min_x = std::max(0,idx_x-5);
        int min_y = std::max(0,idx_y-5);
        int max_x = std::min(layout_prev.num_ss_x,idx_x+6);
        int max_y = std::min(layout_prev.num_ss_y,idx_y+6);

        for (int y = min_y; y < max_y; y++){
            for (int x = min_x; x < max_x; x++){

            // check if point is a valid subset
            int nss_idx = layout_prev.mask[y*layout_prev.num_ss_x+x];
            if (nss_idx == -1) continue;

            int nss_x = layout_prev.coords[2*nss_idx];
            int nss_y = layout_prev.coords[2*nss_idx+1];

            double dx = (nss_x) - ss_x;
            double dy = (nss_y) - ss_y;
            double dist_sq = dx*dx + dy*dy;

            dist_index_list.emplace_back(dist_sq, nss_idx);
            }
        }

        // either use max_num_neigh or size of list if less than max_num_neigh
        int num_neigh = std::min(max_num_neigh, dist_index_list.size());

        // can't find any neighbours.
        if (num_neigh == 0){
            std::cerr << "Could not find any neighbours from the previous FFT window size for point (" << ss_x << ", " << ss_y << ")." << std::endl;
            std::cerr << "Number of neighbours: " << dist_index_list.size() << std::endl;
            std::cerr << "Neighbours from previous window: " << std::endl;
            for (size_t n = 0; n < dist_index_list.size(); n++){
                int nss_idx = dist_index_list[n].second;
                int nss_x = layout_prev.coords[2*nss_idx];
                int nss_y = layout_prev.coords[2*nss_idx+1];
                std::cerr << "(" << nss_x << ", " << nss_y << "), ";
            }
            std::cerr << std::endl;
            exit(EXIT_FAILURE);
        }

        num_neigh_list[ss] = num_neigh;
        std::nth_element(dist_index_list.begin(), dist_index_list.begin() + num_neigh, dist_index_list.end());
        dist_index_list.resize(num_neigh);

        // Store neighbours indices into neighlist
        for (size_t i = 0; i < num_neigh; ++i) {
            neigh_list[ss*max_num_neigh+i] = dist_index_list[i].second;
        }

    }
}

void WindowLevel::remove_outliers(std::vector<double> &u,
                                  const double mad_scale) {

    std::vector<double> u_new = u;

    int radius = 2;

    for (int ss = 0; ss < layout.num; ss++) {
        
        // subset coords
        int ss_x = layout.coords[2*ss];
        int ss_y = layout.coords[2*ss+1];

        // subset x and y index in 2d mask
        int idx_x = ss_x / layout.step;
        int idx_y = ss_y / layout.step;

        std::vector<double> neigh_vals;

        int min_x = std::max(0, idx_x-radius);
        int min_y = std::max(0, idx_y-radius);
        int max_y = std::min(layout.num_ss_y, idx_y+radius+1);
        int max_x = std::min(layout.num_ss_x, idx_x+radius+1);

        for (int y = min_y; y < max_y; ++y) {
            for (int x = min_x; x < max_x; ++x) {

                // index of neighbour 
                int nss_idx = layout.mask[y*layout.num_ss_x+x];

                // check if invalid neigh
                if (nss_idx == -1 || nss_idx == ss) continue; 

                neigh_vals.push_back(u[nss_idx]);
            }
        }

        if (neigh_vals.size() < 4) continue;

        // Median
        std::sort(neigh_vals.begin(), neigh_vals.end());
        size_t sz = neigh_vals.size();
        double median = (sz % 2 == 0) ? 0.5 * (neigh_vals[sz/2 - 1] + neigh_vals[sz/2]) : neigh_vals[sz/2];

        // MAD
        std::vector<double> abs_devs;
        abs_devs.reserve(sz);
        for (double v : neigh_vals) abs_devs.push_back(std::abs(v - median));

        std::sort(abs_devs.begin(), abs_devs.end());
        double mad = (sz % 2 == 0) ? 0.5 * (abs_devs[sz/2 - 1] + abs_devs[sz/2]) : abs_devs[sz/2];

        if (mad < 1e-12) continue;

        if (std::abs(u[ss] - median) > mad_scale * mad) {
            u_new[ss] = median;
        }
    }
    u = std::move(u_new);
}

void WindowLevel::calc_rigid_displacements(const WindowLevel &prev,
                                           const Image &img_ref,
                                           const Image &img_def,
                                           const Interpolator &interp_def,
                                           const int img_num_ref,
                                           const int img_num_def,
                                           const std::vector<std::string> &filenames){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;

        // TODO: Add a proper flag for this 
        bool subpx = true;


        // consts
        const int ss_size_x = layout.size_x;
        const int ss_size_y = layout.size_y;
        const int num_ss  = layout.num;

        // set all displacements for multiwindow level to 0
        std::fill(u.begin(), u.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);

        // progress bar initialisation
        std::string bar_title = "FFT windowing " + std::to_string(ss_size_x) + "x" + std::to_string(ss_size_y) + " for \033[1;4m" + filenames[img_num_ref] + "\033[0m and \033[1;4m" + filenames[img_num_def] + "\033[0m:";
        ProgressBar pbar(bar_title, layout.num);
        std::atomic<int> current_progress = 0;


        #pragma omp parallel shared(stop_request, level, prev, interp_def, ss_size_x, ss_size_y)
        {


            // class for FFT
            FFT fft(ss_size_x, ss_size_y);

            // loop over subsets for each size/step
            #pragma omp for schedule(dynamic,10)
            for (int ss = 0; ss < layout.num; ss++){

                // exit when ctrl+C
                if (stop_request) continue;

                const double cx = layout.coords[2*ss];
                const double cy = layout.coords[2*ss+1];

                const int corner_x = int(cx - ss_size_x/2);
                const int corner_y = int(cy - ss_size_y/2);

                // get the seed for the new window size
                double prev_u = 0.0;
                double prev_v = 0.0;

                if (level>0)
                    get_displacement_from_prev_window(prev_u, prev_v, prev, ss, cx, cy);

                double corner_x_shft = corner_x+prev_u;
                double corner_y_shft = corner_y+prev_v;

                // populate fft.ss_ref with reference subset values
                subset::fill_from_img(fft.ss_ref,corner_x, corner_y, px_hori, px_vert, img_ref);

                // populate fft.ss_def with interpolator value
                subset::fill_from_img_subpx(fft.ss_def, corner_x_shft, corner_y_shft, interp_def);

                // zero normalise the subsets
                bool normalised = fft.zero_norm_subsets(fft.ss_ref.vals, fft.ss_def.vals, ss_size_x, ss_size_y);

                // get peaks from the cross correlation
                double peak_x = 0, peak_y = 0, temp_max = 0.0;

                if (normalised){
                    fft.correlate();
                    fft.get_peak(peak_x, peak_y, temp_max, subpx, "GAUSSIAN_2D");
                }

                u[ss] = prev_u+peak_x;
                v[ss] = prev_v+peak_y;

                // this isn't essential. storing peak amplitude and cost value for level
                //subset::get_subpx_from_img(fft.ss_def, ss_x+level[i].x[ss], ss_y+level[i].y[ss], interp_def);
                //level[i].cost[ss] = debugcost(fft.ss_ref,fft.ss_def);
                max_val[ss] = temp_max;


                if (g_debug_level>1){
                    int progress = current_progress.fetch_add(1);
                    if (omp_get_thread_num()==0) pbar.update(progress+1);
                }
            }
        }

        // remove outliers in fft
        if (mad_filter){
            remove_outliers(u, mad_scale);
            remove_outliers(v, mad_scale);
        }

        //smooth_field(level[i].x, current_level, 7.0, 5);
        //smooth_field(level[i].y, current_level, 7.0, 5);

        // debugging
        // for (int ss = 0; ss < layout.num; ss++){
        //     std::cout << layout.coords[2*ss] << " " << layout.coords[2*ss+1] << " ";
        //     std::cout << u[ss] << " " << v[ss] << " ";
        //     std::cout << max_val[ss] << std::endl;
        //     //std::cout << level[i].cost[ss] << std::endl;
        // }
        // std::cout << std::endl;

        if (g_debug_level>1){
            pbar.update(current_progress);
            pbar.finish();
        }
    }


void WindowLevel::get_displacement_from_prev_window(double &prev_x, 
                                                    double &prev_y,
                                                    const WindowLevel &prev,
                                                    const int ss,
                                                    const double cx, 
                                                    const double cy) {

    const double epsilon = 10.0;
    double weight_sum_x = 0.0;
    double weight_sum_y = 0.0;
    double weight_tot = 0.0;
    double sum_x = 0;
    double sum_y = 0;

    // weighted average of 4 nearest neighbours
    for (size_t j = 0; j < num_neigh_list[ss]; ++j) {

        int nidx = neigh_list[ss*max_num_neigh+j];
        double cx_neigh = prev.layout.coords[2*nidx];
        double cy_neigh = prev.layout.coords[2*nidx+1];

        double dx = cx - cx_neigh;
        double dy = cy - cy_neigh;
        double dist_sq = dx * dx + dy * dy;

        double weight = 1.0 / (dist_sq + epsilon);

        //sum_x += level[i-1].x[nidx];
        //sum_y += level[i-1].y[nidx];
        weight_sum_x += prev.u[nidx] * weight;
        weight_sum_y += prev.v[nidx] * weight;
        weight_tot += weight;
    }

    //prev_x = sum_x / level[i].num_neigh_list[ss];
    //prev_y = sum_y / level[i].num_neigh_list[ss];
    prev_x = weight_sum_x / weight_tot;
    prev_y = weight_sum_y / weight_tot;

}
