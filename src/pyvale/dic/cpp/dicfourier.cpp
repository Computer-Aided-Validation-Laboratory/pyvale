// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
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
#include "dicfourier.hpp"
#include "dicsubset.hpp"
#include "dicinterpolator.hpp"

namespace fourier {

    std::vector<Shift> shifts;

    void init(std::vector<subset::Grid> &window_data,
              std::vector<int> &ss_sizes,
              std::vector<int> &ss_steps,
              const bool *img_roi, const util::Config &conf){

        // timer for the initialisation
        //Timer timer("entire FFT initislisation");
        
        // loop over the window sizes
        for (size_t i = 0; i < ss_sizes.size(); i++) {

            const int window_size = ss_sizes[i];
            const int window_step = ss_steps[i];

            // generate subset information for each window size.
            // for the last size all subsets need to sit within the ROI
            if (i==ss_sizes.size()-1)
                window_data.push_back(subset::create_grid(img_roi, window_step,
                                                        window_size, conf.px_hori, 
                                                        conf.px_vert, false));
            else 
                window_data.push_back(subset::create_grid(img_roi, window_step, 
                                                        window_size, conf.px_hori, 
                                                        conf.px_vert, true));

            // shifts for each subset size
            Shift shift;
            shift.max_num_neigh = 4;

            // resize vectors
            shift.x.resize(window_data[i].num);
            shift.y.resize(window_data[i].num);
            shift.cost.resize(window_data[i].num);
            shift.max_val.resize(window_data[i].num);

            // we need the neighbours in the previous window size for all sizes 
            // except the first
            if (i > 0){
                shift.gen_neighlist(window_data[i], window_data[i-1]);
            }

        // add the shifts for the current window to the vector
            shifts.push_back(shift);
        }
    }

    void remove_outliers(std::vector<double>& shift,
                         const subset::Grid &ss_grid,
                         const double mad_scale) {

        std::vector<double> updated = shift;

        int radius = 2;

        for (int ss = 0; ss < ss_grid.num; ss++) {
            
            // subset coords
            int ss_x = ss_grid.coords[2*ss];
            int ss_y = ss_grid.coords[2*ss+1];

            // subset x and y index in 2d mask
            int idx_x = ss_x / ss_grid.step;
            int idx_y = ss_y / ss_grid.step;

            std::vector<double> neigh_vals;

            int min_x = std::max(0, idx_x-radius);
            int min_y = std::max(0, idx_y-radius);
            int max_y = std::min(ss_grid.num_ss_y, idx_y+radius+1);
            int max_x = std::min(ss_grid.num_ss_x, idx_x+radius+1);

            for (int y = min_y; y < max_y; ++y) {
                for (int x = min_x; x < max_x; ++x) {

                    // index of neighbour 
                    int nss_idx = ss_grid.mask[y*ss_grid.num_ss_x+x];

                    // check if invalid neigh
                    if (nss_idx == -1 || nss_idx == ss) continue; 

                    neigh_vals.push_back(shift[nss_idx]);
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

            if (std::abs(shift[ss] - median) > mad_scale * mad) {
                updated[ss] = median;
            }
        }
        shift = std::move(updated);
    }

    void multiwindow(const std::vector<subset::Grid> &ss_grid, 
                     const double *img_ref,
                     const double *img_def,
                     const Interpolator &interp_def,
                     const bool fft_mad, 
                     const double fft_mad_scale){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // Loop over window size
        for (size_t i = 0; i < ss_grid.size(); i++){

            //Timer timer("FFT windowing for subset size: " + std::to_string(ss_grid[i].size));

            const int ss_size = ss_grid[i].size;
            const int num_ss  = ss_grid[i].num;

            std::fill(shifts[i].x.begin(), shifts[i].x.end(), 0.0);
            std::fill(shifts[i].y.begin(), shifts[i].y.end(), 0.0);

            std::string bar_title = "FFT windowing for size " + std::to_string(ss_size) + ":";
            ProgressBar pbar(bar_title, ss_grid[i].num);
            std::atomic<int> current_progress = 0;


            #pragma omp parallel shared(stop_request, shifts, ss_grid, interp_def, ss_size)
            {


                // class for FFT
                fourier::FFT fft(ss_size);

                // loop over subsets for each size/step
                #pragma omp for schedule(dynamic,10)
                for (int ss = 0; ss < ss_grid[i].num; ss++){

                    // exit when ctrl+C
                    if (stop_request) continue;

                    const int ss_x = ss_grid[i].coords[2*ss];
                    const int ss_y = ss_grid[i].coords[2*ss+1];

                    // get the seed for the new window size
                    auto [prev_x, prev_y] = get_prev_shift(i, ss, ss_x, ss_y, shifts, ss_grid);
                    double ss_x_shft = ss_x+prev_x;
                    double ss_y_shft = ss_y+prev_y;

                    // populate fft.ss_ref with reference subset values
                    subset::get_px_from_img(fft.ss_ref,ss_x, ss_y, px_hori, px_vert, img_ref);

                    // populate fft.ss_def with interpolator value
                    subset::get_subpx_from_img(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);

                    // zero normalise the subsets
                    zero_norm_subsets(fft.ss_ref.vals, fft.ss_def.vals, ss_size);

                    // get peaks from the cross correlation
                    double peak_x = 0, peak_y = 0, max_val = 0.0;
                    fft.correlate();
                    fft.find_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");


                    // update the shift arrays
                    if (i == 0){
                        shifts[i].x[ss] = peak_x;
                        shifts[i].y[ss] = peak_y;
                    }
                    else {
                        shifts[i].x[ss] = prev_x+peak_x;
                        shifts[i].y[ss] = prev_y+peak_y;
                    }

                    // this isn't essential. storing peak amplitude and cost value for shifts
                    //subset::get_subpx_from_img(fft.ss_def, ss_x+shifts[i].x[ss], ss_y+shifts[i].y[ss], interp_def);
                    //shifts[i].cost[ss] = debugcost(fft.ss_ref,fft.ss_def);
                    shifts[i].max_val[ss] = max_val;


                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        if (omp_get_thread_num()==0) pbar.update(progress+1);
                    }
                }
            }

            // remove outliers in fft
            if (fft_mad){
                remove_outliers(shifts[i].x, ss_grid[i], fft_mad);
                remove_outliers(shifts[i].y, ss_grid[i], fft_mad);
            }

            //smooth_field(shifts[i].x, ss_grid[i], 7.0, 5);
            //smooth_field(shifts[i].y, ss_grid[i], 7.0, 5);

            //for (int ss = 0; ss < ss_grid[i].num; ss++){
            //    std::cout << ss_grid[i].coords[2*ss] << " " << ss_grid[i].coords[2*ss+1] << " ";
            //    std::cout << shifts[i].x[ss] << " " << shifts[i].y[ss] << " ";
            //    std::cout << shifts[i].max_val[ss] << " ";
            //    std::cout << shifts[i].cost[ss] << std::endl;
            //}
            //std::cout << std::endl;

            if (g_debug_level>0){
                pbar.update(current_progress);
                pbar.finish();
            }
        }



    }


     std::pair<double,double> get_prev_shift(const int i, const int ss,
                                       const double ss_x, const double ss_y,
                                       const std::vector<Shift>& shifts,
                                       const std::vector<subset::Grid>& ss_grid) {
        const double epsilon = 10.0;
        double weight_sum_x = 0.0;
        double weight_sum_y = 0.0;
        double weight_tot = 0.0;
        double prev_x = 0;
        double prev_y = 0;
        double sum_x = 0;
        double sum_y = 0;

        // assign values for all subset sizes EXCEPT first
        if (i > 0){

            // weighted average of 4 nearest neighbours
            for (size_t j = 0; j < shifts[i].num_neigh_list[ss]; ++j) {

                int nidx = shifts[i].neigh_list[ss*shifts[i].max_num_neigh+j];
                int neigh_x = ss_grid[i-1].coords[2*nidx];
                int neigh_y = ss_grid[i-1].coords[2*nidx+1];

                double dx = ss_x - neigh_x;
                double dy = ss_y - neigh_y;
                double dist_sq = dx * dx + dy * dy;

                double weight = 1.0 / (dist_sq + epsilon);

                //sum_x += shifts[i-1].x[nidx];
                //sum_y += shifts[i-1].y[nidx];
                weight_sum_x += shifts[i-1].x[nidx] * weight;
                weight_sum_y += shifts[i-1].y[nidx] * weight;
                weight_tot += weight;
            }

            //prev_x = sum_x / shifts[i].num_neigh_list[ss];
            //prev_y = sum_y / shifts[i].num_neigh_list[ss];
            prev_x = weight_sum_x / weight_tot;
            prev_y = weight_sum_y / weight_tot;
        }
        return {prev_x, prev_y};
    }


    void smooth_field(std::vector<double>& shift,
                    const subset::Grid& ss_grid,
                    double sigma = 1.0,
                    int radius = 2) {

        std::vector<double> smoothed = shift;

        const int width = ss_grid.num_ss_x;
        const int height = ss_grid.num_ss_y;

        // Precompute Gaussian weights
        std::vector<std::vector<double>> weights(2 * radius + 1, std::vector<double>(2 * radius + 1));
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                double dist2 = dx * dx + dy * dy;
                weights[dy + radius][dx + radius] = std::exp(-dist2 / (2.0 * sigma * sigma));
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                int center_idx = ss_grid.mask[y * width + x];
                if (center_idx == -1) continue;

                double sum = 0.0;
                double weight_sum = 0.0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;

                        int n_idx = ss_grid.mask[ny * width + nx];
                        if (n_idx == -1) continue;

                        double val = shift[n_idx];
                        double weight = weights[dy + radius][dx + radius];

                        sum += val * weight;
                        weight_sum += weight;
                    }
                }

                if (weight_sum > 0.0) {
                    smoothed[center_idx] = sum / weight_sum;
                }
            }
        }

        shift = std::move(smoothed);
    }


    double debugcost(const subset::Pixels &ss_ref, const subset::Pixels &ss_def){
        const int num_px = ss_def.num_px;
        double cost = 0.0;
        double mean_ref = 0.0;
        double mean_def = 0.0;

        // loop over pixel values in reference image
        for (int i = 0; i < num_px; i++){
            mean_ref += ss_ref.vals[i];
            mean_def += ss_def.vals[i];
        }

        mean_ref /= num_px;
        mean_def /= num_px;

        // get cost function denominators
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_ref += (ss_ref.vals[i] - mean_ref)*
                               (ss_ref.vals[i] - mean_ref);
            sum_squared_def += (ss_def.vals[i] - mean_def)*
                               (ss_def.vals[i] - mean_def);
        }
        double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
        double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);


        // calcualte cost 
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            cost += (ref_norm - def_norm) * (ref_norm - def_norm);
        }
        return cost;
    }





    void zero_norm_subsets(std::vector<double>& ref_vals, std::vector<double>& def_vals, const int ss_size) {
        const int total_px = ss_size * ss_size;

        // Compute means
        double mean_def = 0.0;
        double mean_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            mean_def += def_vals[i];
            mean_ref += ref_vals[i];
        }
        mean_def /= total_px;
        mean_ref /= total_px;

        // Compute standard deviations
        double std_def = 0.0;
        double std_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            std_def += std::pow(def_vals[i] - mean_def, 2);
            std_ref += std::pow(ref_vals[i] - mean_ref, 2);
        }
        std_def = std::sqrt(std_def / total_px);
        std_ref = std::sqrt(std_ref / total_px);

        // Normalize
        for (int i = 0; i < total_px; ++i) {
            def_vals[i] = (def_vals[i] - mean_def) / std_def;
            ref_vals[i] = (ref_vals[i] - mean_ref) / std_ref;
        }
    }
    


    void zero_norm_subsets_single(std::vector<double>& ref_vals, int ss_size) {
        const int total_px = ss_size * ss_size;

        // Compute means
        double mean_def = 0.0;
        double mean_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            mean_ref += ref_vals[i];
        }
        mean_def /= total_px;
        mean_ref /= total_px;

        // Compute standard deviations
        double std_def = 0.0;
        double std_ref = 0.0;
        for (int i = 0; i < total_px; ++i) {
            std_ref += std::pow(ref_vals[i] - mean_ref, 2);
        }
        std_def = std::sqrt(std_def / total_px);
        std_ref = std::sqrt(std_ref / total_px);

        // Normalize
        for (int i = 0; i < total_px; ++i) {
            ref_vals[i] = (ref_vals[i] - mean_ref) / std_ref;
        }
    }



    void single_grid(const subset::Grid &ss_grid,
                     const double *prev_img_u,
                     const double *prev_img_v,
                     const int window_size,
                     const double *img_ref,
                     const double *img_def,
                     const Interpolator &interp_def){


        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        
        const int ss_size = ss_grid.size;

        const int window_half = window_size/2;

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // Loop over window size
        #pragma omp parallel shared(stop_request, shifts, ss_grid, interp_def, ss_size)
        {

            // class for FFT
            int tid = omp_get_thread_num();
            fourier::FFT fft(window_size);

            // loop over subsets for each size/step
            #pragma omp for schedule(dynamic,10)
            for (int ss = 0; ss < ss_grid.num; ss++){

                int ss_x = ss_grid.coords[2*ss] + prev_img_u[ss];
                int ss_y = ss_grid.coords[2*ss+1] + prev_img_v[ss];

                // Clamp the deformed window to image range
                int ss_x_shft, ss_y_shft;
                ss_x_shft = std::clamp(ss_x - window_size/2 + ss_size/2, 0, px_hori - window_size);
                ss_y_shft = std::clamp(ss_y - window_size/2 + ss_size/2, 0, px_vert - window_size);


                // Offsets to center the subset within the larger window
                int offset = window_half - ss_size/2;
                int offset_x, offset_y;
                offset_x = std::min(offset, ss_x);
                offset_y = std::min(offset, ss_y);


                for (int row = 0; row < ss_size; ++row) {
                    for (int col = 0; col < ss_size; ++col) {

                        // Source coordinates in img_def
                        int px_y = ss_y + row;
                        int px_x = ss_x + col;

                        if (px_x >= px_hori || px_y >= px_vert) {
                            std::cout << "original subset coords: " << ss_grid.coords[2*ss] << ss_grid.coords[2*ss+1] << std::endl;
                            std::cout << "previous displacements: " << prev_img_u[ss] << prev_img_v[ss] << std::endl;
                            std::cout << "Image access out of bounds!" << std::endl;
                            exit(0);
                            continue;
                        }

                        // Target coordinates in ss_ref
                        int target_y = offset_y + row;
                        int target_x = offset_x + col;

                        int idx_img = px_y * px_hori + px_x;
                        int idx_window = target_y * window_size + target_x;

                        if (idx_window >= window_size*window_size){
                            std::cout << "idx_window out of bounds: " << idx_window << " ss_x: " << ss_x << " ss_y: " << ss_y << " target_x: " << target_x << " target_y: " << target_y << std::endl;
                            exit(0);
                        }

                        // On-the-fly 2D Hann window
                        const double hann_row = 0.5 * (1.0 - cos(2.0 * M_PI * row / (ss_size - 1)));
                        const double hann_col = 0.5 * (1.0 - cos(2.0 * M_PI * col / (ss_size - 1)));
                        const double window_val = hann_row * hann_col;

                        fft.ss_ref.vals[idx_window] = img_ref[idx_img] * window_val;
                    }
                }

                // populate fft.ss_def with interpolator values
                subset::get_subpx_from_img(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);

                // TODO: Make hanning window calc part of the initialization process.
                for (int row = 0; row < window_size; ++row) {
                    for (int col = 0; col < window_size; ++col) {
                        // On-the-fly 2D Hann window
                        const double hann_row = 0.5 * (1.0 - cos(2.0 * M_PI * row / (window_size - 1)));
                        const double hann_col = 0.5 * (1.0 - cos(2.0 * M_PI * col / (window_size - 1)));
                        const double window_val = hann_row * hann_col;
                        fft.ss_def.vals[row*window_size+col]*=window_val;
                    }
                }

                // get peaks from the cross correlation
                double peak_x = 0, peak_y = 0, max_val = 0.0;
                fft.correlate_phase();
                fft.find_peak_offset(peak_x, peak_y, max_val, subpx, "gaussian_2d");

                // resetting values in fft_ref to 0.0.
                for (int row = 0; row < ss_size; ++row) {
                    for (int col = 0; col < ss_size; ++col) {

                        // Target coordinates in ss_ref
                        int target_y = offset_y + row;
                        int target_x = offset_x + col;
                        int idx_window = target_y * window_size + target_x;

                        if (idx_window >= window_size*window_size){
                            std::cout << "idx_window out of bounds: " << idx_window << " ss_x: " << ss_x << " ss_y: " << ss_y << " target_x: " << target_x << " target_y: " << target_y << std::endl;
                            exit(0);
                        }
                        fft.ss_ref.vals[idx_window] = 0.0;
                    }
                }
                // update the shift arrays
                shifts.back().x[ss] = peak_x;
                shifts.back().y[ss] = peak_y;
            }
        }
        // remove outliers in fft
        //if (fft_mad){
        //    remove_outliers(shifts[i].x, ss_grid[i], fft_mad);
        //    remove_outliers(shifts[i].y, ss_grid[i], fft_mad);
        //}
    }



   void get_single_window_fftcc_peak(double &peak_x, double &peak_y,
                                     const int ss_x, const int ss_y,
                                     const int ss_size, const int window_size,
                                     const double *img_ref, const double *img_def,
                                     const Interpolator &interp_def){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        const int window_half = window_size/2;
        const int ss_half = ss_size/2;

        // class for FFT
        fourier::FFT fft(window_size);

        // TODO: Add a proper flag for this 
        bool subpx = true;


        // Clamp the deformed window to image range
        int ss_x_shft, ss_y_shft;
        ss_x_shft = std::clamp(ss_x - window_half + ss_half, 0, px_hori - window_size);
        ss_y_shft = std::clamp(ss_y - window_half + ss_half, 0, px_vert - window_size);

        // Offsets to center the subset within the larger window
        int offset = window_half - ss_half;
        int offset_x, offset_y;
        offset_x = std::min(offset, ss_x);
        offset_y = std::min(offset, ss_y);


        for (int row = 0; row < ss_size; ++row) {
            for (int col = 0; col < ss_size; ++col) {

                // Source coordinates in img_def
                int px_y = ss_y + row;
                int px_x = ss_x + col;

                // Target coordinates in ss_ref
                int target_y = offset_y + row;
                int target_x = offset_x + col;

                int idx_img = px_y * px_hori + px_x;
                int idx_window = target_y * window_size + target_x;

                if (idx_window >= window_size*window_size){
                    std::cout << "idx_window out of bounds: " << idx_window << " ss_x: " << ss_x << " ss_y: " << ss_y << " target_x: " << target_x << " target_y: " << target_y << std::endl;
                    continue;
                }

                if (px_x >= px_hori || px_y >= px_vert) {
                    std::cout << "Subset coords: (" << ss_x << ", " << ss_y << "). " <<  std::endl;
                    std::cout << "Image access out of bounds!" << std::endl;
                    continue;
                }

                // On-the-fly 2D Hann window
                const double hann_row = 0.5 * (1.0 - cos(2.0 * M_PI * row / (ss_size - 1)));
                const double hann_col = 0.5 * (1.0 - cos(2.0 * M_PI * col / (ss_size - 1)));
                const double window_val = hann_row * hann_col;

                fft.ss_ref.vals[idx_window] = img_ref[idx_img] * window_val;
            }
        }

        // populate fft.ss_def with interpolator values
        subset::get_subpx_from_img(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);


        // TODO: Make hanning window calc part of the initialization process.
        for (int row = 0; row < window_size; ++row) {
            for (int col = 0; col < window_size; ++col) {
                // On-the-fly 2D Hann window
                const double hann_row = 0.5 * (1.0 - cos(2.0 * M_PI * row / (window_size - 1)));
                const double hann_col = 0.5 * (1.0 - cos(2.0 * M_PI * col / (window_size - 1)));
                const double window_val = hann_row * hann_col;
                fft.ss_def.vals[row*window_size+col]*=window_val;
            }
        }


        // get peaks from the cross correlation
        double max_val = 0.0;
        fft.correlate_phase();
        fft.find_peak(peak_x, peak_y, max_val, subpx, "gaussian_2d");



        // resetting values in fft_ref to 0.0.
        for (int row = 0; row < ss_size; ++row) {
            for (int col = 0; col < ss_size; ++col) {

                // Target coordinates in ss_ref
                int target_y = offset_y + row;
                int target_x = offset_x + col;

                //int idx_img = px_y * px_hori + px_x;
                int idx_window = target_y * window_size + target_x;

                if (idx_window >= window_size*window_size){
                    std::cout << "idx_window out of bounds: " << idx_window << " ss_x: " << ss_x << " ss_y: " << ss_y << " target_x: " << target_x << " target_y: " << target_y << std::endl;
                    exit(0);
                }

                // IF DUBUGGING WITH THE BELOW COMMENT THIS
                fft.ss_ref.vals[idx_window] = 0.0;
            }
        }

    }
}



