// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicfourier.hpp"

namespace fourier {

    std::vector<Shift> shifts;

    void init(std::vector<util::SubsetData> &window_data,
              std::vector<int> &ss_sizes,
              std::vector<int> &ss_steps,
              const bool *img_roi, const util::Config &conf){

        // timer for the initialisation
        //util::Timer timer("entire FFT initislisation");
        
        // loop over the window sizes
        for (size_t i = 0; i < ss_sizes.size(); i++) {

            const int window_size = ss_sizes[i];
            const int window_step = ss_steps[i];

            // generate subset information for each window size.
            // for the last size all subsets need to sit within the ROI
            if (i==ss_sizes.size()-1)
                window_data.push_back(util::gen_ss_list(img_roi, window_step,
                                                        window_size, conf.px_hori, 
                                                        conf.px_vert, false));
            else 
                window_data.push_back(util::gen_ss_list(img_roi, window_step, 
                                                        window_size, conf.px_hori, 
                                                        conf.px_vert, true));
            

            // shifts for each subset size
            Shift shift;
            shift.num_neigh = 4;

            // resize vectors
            shift.x.resize(window_data[i].num);
            shift.y.resize(window_data[i].num);
            shift.cost.resize(window_data[i].num);
            shift.max_val.resize(window_data[i].num);

            // we need the neighbours in the previous window size for all sizes 
            // except the first
            if (i > 0){
                shift.neighlist.resize(shift.num_neigh*window_data[i].num);
                shift.gen_neighlist(window_data[i], window_data[i-1]);
            }

            // add the shifts for the current window to the vector
            shifts.push_back(shift);

        }
    }

    void remove_outliers(std::vector<double>& shift,
                         const util::SubsetData &ssdata,
                         const double mad_scale) {

        std::vector<double> updated = shift;

        int radius = 2;

        for (int ss = 0; ss < ssdata.num; ss++) {
            
            // subset coords
            int ss_x = ssdata.coords[2*ss];
            int ss_y = ssdata.coords[2*ss+1];

            // subset x and y index in 2d mask
            int idx_x = ss_x / ssdata.step;
            int idx_y = ss_y / ssdata.step;

            std::vector<double> neigh_vals;

            int min_x = std::max(0, idx_x-radius);
            int min_y = std::max(0, idx_y-radius);
            int max_y = std::min(ssdata.num_ss_y, idx_y+radius+1);
            int max_x = std::min(ssdata.num_ss_x, idx_x+radius+1);

            for (int y = min_y; y < max_y; ++y) {
                for (int x = min_x; x < max_x; ++x) {

                    // index of neighbour 
                    int nss_idx = ssdata.mask[y*ssdata.num_ss_x+x];

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

    void mgwd(const std::vector<util::SubsetData> &ssdata, 
              const double *img_ref, const double *img_def, 
              const Interpolator &interp_def, const bool fft_mad, 
              const double fft_mad_scale){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // Loop over window size
        for (size_t i = 0; i < ssdata.size(); i++){

            //util::Timer timer("FFT windowing for subset size: " + std::to_string(ssdata[i].size));

            const int ss_size = ssdata[i].size;

            std::fill(shifts[i].x.begin(), shifts[i].x.end(), 0.0);
            std::fill(shifts[i].y.begin(), shifts[i].y.end(), 0.0);

            #pragma omp parallel
            {


                // class for FFT
                fourier::FFT fft(ss_size);

                // loop over subsets for each size/step
                #pragma omp for
                for (int ss = 0; ss < ssdata[i].num; ss++){

                    int ss_x = ssdata[i].coords[2*ss];
                    int ss_y = ssdata[i].coords[2*ss+1];

                    // get the seed for the new window size
                    auto [prev_x, prev_y] = get_prev_shift(i, ss, ss_x, ss_y, shifts, ssdata);
                    double ss_x_shft = ss_x+prev_x;
                    double ss_y_shft = ss_y+prev_y;

                    // populate fft.ss_ref with reference subset values
                    util::extract_ss(fft.ss_ref,ss_x, ss_y, px_hori, px_vert, img_ref);

                    // populate fft.ss_def with interpolator value
                    util::extract_ss_subpx(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);

                    // zero normalise the subsets
                    zero_norm_subsets(fft.ss_ref.vals, fft.ss_def.vals, ss_size);

                    // get peaks from the cross correlation
                    double peak_x = 0, peak_y = 0, max_val = 0.0;
                    fft.correlate();
                    fft.find_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");


                    // debugging print cross correlation
                    //if (ss == 0){
                    //    for (int y = 0; y < fft.ss_ref.size; y++){
                    //        for (int x = 0; x < fft.ss_ref.size; x++){
                    //            std::cout << x << " " << y << " "  << fft.ss_ref.vals[y*fft.ss_ref.size + x] << " " << fft.ss_def.vals[y*fft.ss_def.size + x] << " " << fft.cross_corr[y*fft.ss_ref.size+x] << std::endl;
                    //        }
                    //    }
                    //    std::cout << std::endl;
                    //}
                    //exit(0);

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
                    //util::extract_ss_subpx(fft.ss_def, ss_x+shifts[i].x[ss], ss_y+shifts[i].y[ss], interp_def);
                    //shifts[i].cost[ss] = debugcost(fft.ss_ref,fft.ss_def);
                    shifts[i].max_val[ss] = max_val;
                }
            }

            // remove outliers in fft
            if (fft_mad){
                remove_outliers(shifts[i].x, ssdata[i], fft_mad);
                remove_outliers(shifts[i].y, ssdata[i], fft_mad);
            }


            //smooth_field(shifts[i].x, ssdata[i], 7.0, 5);
            //smooth_field(shifts[i].y, ssdata[i], 7.0, 5);

            //for (int ss = 0; ss < ssdata[i].num; ss++){
            //    std::cout << ssdata[i].coords[2*ss] << " " << ssdata[i].coords[2*ss+1] << " ";
            //    std::cout << shifts[i].x[ss] << " " << shifts[i].y[ss] << " ";
            //    std::cout << shifts[i].max_val[ss] << " ";
            //    std::cout << shifts[i].cost[ss] << std::endl;
            //}
            //std::cout << std::endl;
        }


    }


     std::pair<double,double> get_prev_shift(const int i, const int ss,
                                       const double ss_x, const double ss_y,
                                       const std::vector<Shift>& shifts,
                                       const std::vector<util::SubsetData>& ssdata) {
        const double epsilon = 1.0e-8;
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
            for (size_t j = 0; j < shifts[i].num_neigh; ++j) {

                int nidx = shifts[i].neighlist[ss*shifts[i].num_neigh+j];
                int neigh_x = ssdata[i-1].coords[2*nidx];
                int neigh_y = ssdata[i-1].coords[2*nidx+1];

                double dx = ss_x - neigh_x;
                double dy = ss_y - neigh_y;
                double dist_sq = dx * dx + dy * dy;

                double weight = 1.0 / (dist_sq + epsilon);

                //std::cout << nidx << " " << neigh_x << " " << neigh_y << " " << dx << " " << dy << " " << dist_sq << " " << weight << std::endl;
                sum_x += shifts[i-1].x[nidx];
                sum_y += shifts[i-1].y[nidx];
                //weight_sum_x += shifts[i-1].x[nidx] * weight;
                //weight_sum_y += shifts[i-1].y[nidx] * weight;
                //weight_tot += weight;
            }

            //std::cout << std::endl;
            prev_x = sum_x / shifts[i].num_neigh;
            prev_y = sum_y / shifts[i].num_neigh;
            //prev_x = weight_sum_x / weight_tot;
            //prev_y = weight_sum_y / weight_tot;
        }
        return {prev_x, prev_y};
    }


    void smooth_field(std::vector<double>& shift,
                    const util::SubsetData& ssdata,
                    double sigma = 1.0,
                    int radius = 2) {

        std::vector<double> smoothed = shift;

        const int width = ssdata.num_ss_x;
        const int height = ssdata.num_ss_y;

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

                int center_idx = ssdata.mask[y * width + x];
                if (center_idx == -1) continue;

                double sum = 0.0;
                double weight_sum = 0.0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;

                        int n_idx = ssdata.mask[ny * width + nx];
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


    double debugcost(const util::Subset &ss_ref, const util::Subset &ss_def){
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



   void sgwd(const util::SubsetData &ssdata, const int window_size, const double *img_ref, const double *img_def, const Interpolator &interp_def){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        
        const int ss_size = ssdata.size;

        // TODO: Add a proper flag for this 
        bool subpx = true;

        // shifts for each subset size
        Shift shift;
        shift.num_neigh = 4;

        // resize vectors
        shift.x.resize(ssdata.num);
        shift.y.resize(ssdata.num);
        shift.cost.resize(ssdata.num);
        shift.max_val.resize(ssdata.num);

        shifts.push_back(shift);

        // Loop over window size
        #pragma omp parallel
        {

            // class for FFT
            fourier::FFT fft(window_size);

            // loop over subsets for each size/step
            #pragma omp for
            for (int ss = 0; ss < ssdata.num; ss++){
                std::cout << ss << std::endl;
                int ss_x = ssdata.coords[2*ss];
                int ss_y = ssdata.coords[2*ss+1];

                // get the seed for the new window size
                int ss_x_shft, ss_y_shft;
                ss_x_shft = std::clamp(ss_x - window_size/2, 0, px_hori - window_size);
                ss_y_shft = std::clamp(ss_y - window_size/2, 0, px_vert - window_size);

                // Offsets to center the subset within the larger window
                int offset = window_size/2 - ss_size/2;
                int offset_x, offset_y;
                offset_x = std::min(offset, ss_x);
                offset_y = std::min(offset, ss_y);

                //std::fill(fft.ss_ref.vals.begin(), fft.ss_ref.vals.end(), 0.0);

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
                        fft.ss_ref.vals[idx_window] = img_ref[idx_img];
                    }
                }

                // populate fft.ss_def with interpolator values
                util::extract_ss_subpx(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);

                // zero normalise the subsets
                //zero_norm_subsets(fft.ss_ref.vals, fft.ss_def.vals, ss_size);

                // get peaks from the cross correlation
                double peak_x = 0, peak_y = 0, max_val = 0.0;
                fft.correlate();
                fft.find_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
                for (int row = 0; row < ss_size; ++row) {
                    for (int col = 0; col < ss_size; ++col) {

                        // Source coordinates in img_def
                        //int px_y = ss_y + row;
                        //int px_x = ss_x + col;

                        // Target coordinates in ss_ref
                        int target_y = offset_y + row;
                        int target_x = offset_x + col;

                        //int idx_img = px_y * px_hori + px_x;
                        int idx_window = target_y * window_size + target_x;
                        fft.ss_ref.vals[idx_window] = 0.0;
                    }
                }

                // debugging print cross correlation
                //if (ss == 0){
                    // for (int y = 0; y < fft.ss_ref.size; y++){
                    //     for (int x = 0; x < fft.ss_ref.size; x++){
                    //         std::cout << x << " " << y << " "  << fft.ss_ref.vals[y*fft.ss_ref.size + x] << " " << fft.ss_def.vals[y*fft.ss_def.size + x] << " " << fft.cross_corr[y*fft.ss_ref.size+x] << std::endl;
                    //     }
                    // }
                    // std::cout << std::endl;
                //}


                // update the shift arrays
                shifts[0].x[ss] = peak_x;
                shifts[0].y[ss] = peak_y;

                // this isn't essential. storing peak amplitude and cost value for shifts
                //util::extract_ss_subpx(fft.ss_def, ss_x+shifts[i].x[ss], ss_y+shifts[i].y[ss], interp_def);
                //shifts[i].cost[ss] = debugcost(fft.ss_ref,fft.ss_def);
            }
        }

        // remove outliers in fft
        //remove_outliers(shifts[0].x, ssdata[i], 4.0);
        //remove_outliers(shifts[0].y, ssdata[i], 4.0);

        for (int ss = 0; ss < ssdata.num; ss++){
            std::cout << ssdata.coords[2*ss] << " " << ssdata.coords[2*ss+1] << " ";
            std::cout << shifts[0].x[ss] << " " << shifts[0].y[ss] << " ";
            std::cout << shifts[0].max_val[ss] << " ";
            std::cout << shifts[0].cost[ss] << std::endl;
        }
        std::cout << std::endl;
        exit(0);
    }

   void test(double &peak_x, double &peak_y, int ss_x, int ss_y, const int window_size, const double *img_ref, const double *img_def, const Interpolator &interp_def){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        const int ss_size = 31;
        bool subpx = true;

        // class for FFT
        fourier::FFT fft(window_size);

        // get the seed for the new window size
        int ss_x_shft, ss_y_shft;
        ss_x_shft = std::clamp(ss_x - window_size/2 + ss_size/2, 0, px_hori - window_size);
        ss_y_shft = std::clamp(ss_y - window_size/2 + ss_size/2, 0, px_vert - window_size);

        // Offsets to center the subset within the larger window
        int offset = window_size/2 - ss_size/2;
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
                fft.ss_ref.vals[idx_window] = img_ref[idx_img];
            }
        }

        // populate fft.ss_def with interpolator values
        util::extract_ss_subpx(fft.ss_def, ss_x_shft, ss_y_shft, interp_def);

        // zero normalise the subsets
        fourier::zero_norm_subsets(fft.ss_ref.vals, fft.ss_def.vals, window_size);

        // get peaks from the cross correlation
        double max_val = 0.0;
        fft.correlate();
        fft.find_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
        
        // debugging print cross correlation
        //if (ss == 0){
            // for (int y = 0; y < fft.ss_ref.size; y++){
            //     for (int x = 0; x < fft.ss_ref.size; x++){
            //         std::cout << x << " " << y << " "  << fft.ss_ref.vals[y*fft.ss_ref.size + x] << " " << fft.ss_def.vals[y*fft.ss_def.size + x] << " " << fft.cross_corr[y*fft.ss_ref.size+x] << std::endl;
            //     }
            // }
            // std::cout << std::endl;
            // std::cout << peak_x << " " << peak_y << std::endl;
            // exit(0);
        //}
        peak_x*=-1.0;
        peak_y*=-1.0;
        std::cout << peak_x << " " << peak_y << std::endl;
    }





}



