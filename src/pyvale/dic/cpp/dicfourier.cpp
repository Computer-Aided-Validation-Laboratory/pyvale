// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <fftw3.h>
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

    void init(std::vector<util::SubsetData> &ssdata,
              const bool *img_roi, const util::Config conf){

        // timer for the initialisation
        util::Timer timer("entire FFT initislisation");
        
        // loop over the window sizes
        for (size_t i = 0; i < conf.ss_size.size(); i++) {

            const int ss_size = conf.ss_size[i];
            const int ss_step = conf.ss_step[i];

            // generate subset information for each window
            ssdata.push_back(util::gen_ss_list(img_roi, ss_step, ss_size,
                                          conf.px_hori, conf.px_vert));

            // shifts for each subset size
            Shift shift;
            shift.num_neigh = 8;

            // resize vectors
            shift.x.resize(ssdata[i].num);
            shift.y.resize(ssdata[i].num);
            shift.cost.resize(ssdata[i].num);
            shift.max_val.resize(ssdata[i].num);

            // we need the neighbours in the previous window size for all sizes 
            // except the first
            if (i > 0){
                shift.neighlist.resize(shift.num_neigh*ssdata[i].num);
                shift.gen_neighlist(ssdata[i], ssdata[i-1]);
            }

            // add the shifts for the current window to the vector
            shifts.push_back(shift);

        }
    }

    void remove_outliers(std::vector<double>& shift,
                         const util::SubsetData &ssdata,
                         const double mad_scale = 3.0) {

        std::vector<double> updated = shift;

        for (int ss = 0; ss < ssdata.num; ss++) {
            int ss_x = ssdata.coords[2*ss];
            int ss_y = ssdata.coords[2*ss+1];

            int idx_x = ss_x / ssdata.step;
            int idx_y = ss_y / ssdata.step;

            std::vector<double> neigh_vals;

            int min_x = std::max(0, idx_x-2);
            int min_y = std::max(0, idx_y-2);
            int max_y = std::min(ssdata.num_ss_y, idx_y+3);
            int max_x = std::min(ssdata.num_ss_x, idx_x+3);

            for (int y = min_y; y < max_y; ++y) {
                for (int x = min_x; x < max_x; ++x) {
                    int nss_idx = ssdata.mask[y*ssdata.num_ss_x+x];
                    if (nss_idx == -1 || nss_idx == ss) continue; // skip invalid or self

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
              const Interpolator &interp_ref,
              const double *img_ref, const double *img_def,
              const int px_hori, const int px_vert){


        // TODO: Add a proper flag for this 
        bool subpx = true;

        // Loop over window size
        for (size_t i = 0; i < ssdata.size(); i++){

            util::Timer timer("FFT windowing for subset size: " 
                              + std::to_string(ssdata[i].size));

            const int ss_size = ssdata[i].size;

            #pragma omp parallel
            {

                // subsets for FFTs
                util::Subset ss_def(ss_size);
                util::Subset ss_ref(ss_size);

                // class for FFT
                fourier::FFT fft(ss_size, ss_def.vals.data(), ss_ref.vals.data());

                // loop over subsets for each size/step
                #pragma omp for
                for (int ss = 0; ss < ssdata[i].num; ss++){

                    int ss_x = ssdata[i].coords[2*ss];
                    int ss_y = ssdata[i].coords[2*ss+1];
                    
                    // get the deformed subset values
                    util::extract_ss(ss_def,ss_x, ss_y, px_hori, px_vert, img_def);

                    // get the seed for the new window size
                    auto [prev_x, prev_y] = get_prev_shift(i, ss, ss_x, ss_y, shifts, ssdata);
                    double ss_x_shft = ss_x-prev_x;
                    double ss_y_shft = ss_y-prev_y;
                
                    //std::cout << "prev:  " << prev_x << " " << prev_y << std::endl;
                    //std::cout << "shift: " << ss_x_shft << " " << ss_y_shft << std::endl;

                    // get the reference subset values
                    if (subpx) util::extract_ss_subpx(ss_ref, ss_x_shft, ss_y_shft, interp_ref);
                    else util::extract_ss(ss_ref, ss_x_shft, ss_y_shft, px_hori, px_vert, img_ref);

                    // zero normalise the subsets
                    zero_norm_subsets(ss_def.vals, ss_ref.vals, ss_size);


                    // get peaks from the cross correlation
                    double peak_x = 0, peak_y = 0;
                    double max_val = 0.0;
                    fft.correlate();
                    fft.find_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
                    //std::cout << "peak:  " << peak_x << " " << peak_y << std::endl;
                    //exit(0);

                    // update the shift arrays
                    if (i == 0){
                        shifts[i].x[ss] = peak_x;
                        shifts[i].y[ss] = peak_y;
                    }
                    else {
                        shifts[i].x[ss] = prev_x + peak_x;
                        shifts[i].y[ss] = prev_y + peak_y;
                    }

                    shifts[i].cost[ss] = debugcost(ss_def, ss_ref);
                    shifts[i].max_val[ss] = max_val;
                }
            }

            // remove outliers in fft
            remove_outliers(shifts[i].x, ssdata[i], 3.0);
            remove_outliers(shifts[i].y, ssdata[i], 3.0);

            for (size_t ss = 0; ss < ssdata[i].num; ss++){
                std::cout << ssdata[i].coords[2*ss] << " " << ssdata[i].coords[2*ss+1] << " ";
                std::cout << shifts[i].x[ss] << " " << shifts[i].y[ss] << " ";
                std::cout << shifts[i].max_val[ss] << " ";
                std::cout << shifts[i].cost[ss] << std::endl;
            }

            std::cout << std::endl;
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

        // assign values for all subset sizes EXCEPT first
        if (i > 0){

            // weighted average of 4 nearest neighbours
            for (int j = 0; j < shifts[i].num_neigh; ++j) {

                int nidx = shifts[i].neighlist[ss*shifts[i].num_neigh+j];
                int neigh_x = ssdata[i-1].coords[2*nidx];
                int neigh_y = ssdata[i-1].coords[2*nidx+1];

                double dx = ss_x - neigh_x;
                double dy = ss_y - neigh_y;
                double dist_sq = dx * dx + dy * dy;

                double weight = 1.0 / (dist_sq + epsilon);
                weight_sum_x += shifts[i-1].x[nidx] * weight;
                weight_sum_y += shifts[i-1].y[nidx] * weight;
                weight_tot += weight;
            }

            prev_x = weight_sum_x / weight_tot;
            prev_y = weight_sum_y / weight_tot;
        }
        return {prev_x, prev_y};
    }






    inline void destroy_fftw_plans(std::vector<fftw_plan>& plans) {
        for (auto& plan : plans) {
            if (plan != nullptr) {
                fftw_destroy_plan(plan);
            }
        }
        plans.clear();
    }





    inline void free_fftw_arrays(std::vector<fftw_complex*>& vec) {
        for (auto& ptr : vec) {
            if (ptr != nullptr) {
                fftw_free(ptr);
            }
        }
        vec.clear();
    }






    double debugcost(util::Subset &ss_ref, util::Subset &ss_def){
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
            cost += (def_norm - ref_norm) * (def_norm - ref_norm);
        }
        return cost;
    }





    void zero_norm_subsets(std::vector<double>& def_vals, std::vector<double>& ref_vals, int ss_size) {
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

}
