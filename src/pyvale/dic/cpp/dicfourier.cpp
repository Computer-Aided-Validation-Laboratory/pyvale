// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <fftw3.h>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <algorithm>

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicfourier.hpp"

namespace fourier {

    std::vector<util::Subset> ss_def;
    std::vector<util::Subset> ss_ref;

    // forward fft
    std::vector<fftw_plan> plans_def;
    std::vector<fftw_plan> plans_ref;
    std::vector<fftw_complex*> fft_def;
    std::vector<fftw_complex*> fft_ref;

    std::vector<std::vector<double>> shift_x;
    std::vector<std::vector<double>> shift_y;

    // inverse fft
    std::vector<fftw_plan> plans_inv;
    std::vector<std::vector<double>> ifft_out;

    std::vector<std::vector<int>> neighlist;


    void init(std::vector<util::SubsetData> ssdata, 
              const bool *img_roi, const util::Config conf){
        
        util::Timer timer("entire FFT initislisation");


        for (size_t i = 0; i < conf.windows.size(); i++) {

            int w = conf.windows[i];

            // generate subset information
            ssdata.push_back(util::gen_ss_list(img_roi, w/2, w,
                                          conf.px_hori, conf.px_vert));

            ss_def.push_back(util::Subset(w));
            ss_ref.push_back(util::Subset(w));
            fft_def.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));
            fft_ref.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));

            // forward fftw plans
            plans_def.push_back(fftw_plan_dft_r2c_2d(w, w, &ss_def[i].vals[0], fft_def[i], FFTW_ESTIMATE));
            plans_ref.push_back(fftw_plan_dft_r2c_2d(w, w, &ss_ref[i].vals[0], fft_ref[i], FFTW_ESTIMATE));
            
            // inverse fftw plans
            ifft_out.push_back(std::vector<double>(w*w));
            plans_inv.push_back(fftw_plan_dft_c2r_2d(w, w, fft_def[i], ifft_out[i].data(), FFTW_ESTIMATE));


            shift_x.push_back(std::vector<double>(ssdata[i].num, 0));
            shift_y.push_back(std::vector<double>(ssdata[i].num, 0));

            // TODO: shifts for each window size
            //shift_x.push_back(std::vector<double>(ssdata[i].num_ss_x*ssdata[i].num_ss_y, NAN));
            //shift_y.push_back(std::vector<double>(ssdata[i].num_ss_x*ssdata[i].num_ss_y, NAN));


            // precompute neighbouring windows
            if (i > 0){
                neighlist.push_back(std::vector<int>(4*ssdata[i].num, 0));
                get_4nn(neighlist[i-1], ssdata[i], ssdata[i-1]);
            }

        }
        std::cout << "Finished FFT initialisation" << std::endl;
    }

    void remove_outliers_auto(std::vector<double>& shift,
                              util::SubsetData ssdata,
                              double mad_scale = 3.0) {
        
        std::vector<double> updated = shift;
    
        for (const auto& [idx, neighbors] : ssdata.neigh) {
            if (neighbors.size() < 2) continue;
    
            std::vector<double> neigh_vals;
            for (int n : neighbors) {
                neigh_vals.push_back(shift[n]);
            }
    
            // Compute median
            std::sort(neigh_vals.begin(), neigh_vals.end());
            double median = neigh_vals[neigh_vals.size() / 2];
    
            // Median Absolute Deviation
            std::vector<double> abs_devs;
            for (double v : neigh_vals) {
                abs_devs.push_back(std::abs(v - median));
            }
            std::sort(abs_devs.begin(), abs_devs.end());
            double mad = abs_devs[abs_devs.size() / 2];

            if (mad == 0) continue; // no variation among neighbors

            // If current shift deviates significantly, replace
            if (std::abs(shift[idx] - median) > mad_scale * mad) {
                updated[idx] = median;
            }
        }
        //debugging
        //for (int ss = 0; ss < ssdata.num; ss++){
        //    std::cout << ssdata.coords[ss*2] << " " << ssdata.coords[ss*2+1] << " " << shift[ss] << " " << updated[ss] << std::endl;
        //}
        shift = std::move(updated);

    }


    void mgwd(const std::vector<util::SubsetData> ssdata,
              const double *img_def, const double *img_ref,
              const util::Config conf){


        // Loop over window size
        for (size_t i = 0; i < conf.windows.size(); i++){

            int w = conf.windows[i];

            int ss_x, ss_y;
            int ss_x_centre, ss_y_centre;
            int shift_x_prev = 0;
            int shift_y_prev = 0;

            // loop over subsets for each window size
            for (int ss = 0; ss < ssdata[i].num; ss++){

                ss_x = ssdata[i].coords[2*ss];
                ss_y = ssdata[i].coords[2*ss+1];
                ss_x_centre = ss_x + ssdata[i].step / 2.0;
                ss_y_centre = ss_y + ssdata[i].step / 2.0;
                    

                // window has to always be decreasing in size
                if (i > 0) {

                    //shift_x_prev = shift_x[i-1][neighlist[i-1][ss]];
                    //shift_y_prev = shift_y[i-1][neighlist[i-1][ss]];
                    
                    // take a weighted sum of 4 nearest neighbours
                    const double epsilon = 1e-8;
                    double weight_sum_x = 0.0;
                    double weight_sum_y = 0.0;
                    double weight_tot = 0.0;
                    
                    for (int j = 0; j < 4; ++j) {
                        int nidx = neighlist[i - 1][ss * 4 + j];

                        int neigh_x = ssdata[i-1].coords[2*nidx];
                        int neigh_y = ssdata[i-1].coords[2*nidx + 1];
                        double neigh_x_centre = neigh_x + ssdata[i-1].step / 2.0;
                        double neigh_y_centre = neigh_y + ssdata[i-1].step / 2.0;

                        double dx = ss_x_centre - neigh_x_centre;
                        double dy = ss_y_centre - neigh_y_centre;
                        double dist_sq = dx * dx + dy * dy;

                        double weight = 1.0 / (dist_sq + epsilon);
                        weight_sum_x += shift_x[i-1][nidx] * weight;
                        weight_sum_y += shift_y[i-1][nidx] * weight;
                        weight_tot += weight;
                    }

                    shift_x_prev = weight_sum_x / weight_tot;
                    shift_y_prev = weight_sum_y / weight_tot;

                    // debugging
                    //std::cout << ss_x << " " << ss_y << " " << " " << neighlist[i][ss] << " " << shift_x_prev << " " << shift_y_prev << std::endl;

                    // debugging
                    //if ((ss_x == 768) && (ss_y == 288)){
                    //    std::cout << "subset: " << " (" << ss_x << ", " << ss_y << ")" << std::endl;
                    //    std::cout << "grid_vals: " << " (" << grid_vals[0] << ", " << grid_vals[2] << "), (" << grid_vals[1] << ", " << grid_vals[3] << ")" << std::endl;
                    //    std::cout << "grid_indx: " << " (" << grid_indx[0] << ", " << grid_indx[2] << "), (" << grid_indx[1] << ", " << grid_vals[3] << ")" << std::endl;
                    //    std::cout << "valid_neigh: " << " " << valid_neigh[0] << " " << valid_neigh[1] << " " << valid_neigh[2] << " " << valid_neigh[3] << std::endl;
                    //    std::cout << "valid_count: " << " " << valid_count << std::endl;
                    //    std::cout << "tx " << " " << tx << " " << ty << std::endl;
                    //    std::cout << "shiftx " << " " << shift_x00 << " " << shift_x10 << " " << shift_x01 << " " << shift_x11 << std::endl;
                    //    std::cout << "shifty " << " " << shift_y00 << " " << shift_y10 << " " << shift_y01 << " " << shift_y11 << std::endl;
                    //    std::cout << "interp " << interp_x << " " << interp_y << std::endl;
                    //    std::cout << "num_ss_x " << ssdata[i].num_ss_x << "num_ss_y " << ssdata[i].num_ss_y << std::endl;
                    //}


                }

                // get the deformed subset
                util::extract_ss(ss_def[i],ss_x, ss_y, conf.px_hori,
                                 conf.px_vert, img_def);

                // get the reformed subset
                util::extract_ss(ss_ref[i], ss_x-shift_x_prev, ss_y-shift_y_prev, conf.px_hori,
                                 conf.px_vert, img_ref);

                // calc mean
                double mean_def = 0.0;
                double mean_ref = 0.0;
                for (int px = 0; px < w*w; px++) {
                    mean_def += ss_def[i].vals[px];
                    mean_ref += ss_ref[i].vals[px];
                }
                mean_def /= (w * w);
                mean_ref /= (w * w);
                
                // calc std. dev.
                double std_def = 0.0;
                double std_ref = 0.0;
                for (int px = 0; px < w*w; px++) {
                    std_def += std::pow(ss_def[i].vals[px] - mean_def, 2);
                    std_ref += std::pow(ss_ref[i].vals[px] - mean_ref, 2);
                }
                std_def = std::sqrt(std_def / (w * w));
                std_ref = std::sqrt(std_ref / (w * w));

                // sub mean, div by std dev.
                for (int px = 0; px < w*w; px++) {
                    ss_def[i].vals[px] = (ss_def[i].vals[px] - mean_def) / std_def;
                    ss_ref[i].vals[px] = (ss_ref[i].vals[px] - mean_ref) / std_ref;
                }


                // perform fft
                fftw_execute(plans_def[i]);
                fftw_execute(plans_ref[i]);

                // convolution (index: (window), (pixel), (real/imag))
                // results stored in fft_def
                for (int px = 0; px < w * (w/2 + 1); px++) {
                    double def_re = fft_def[i][px][0];
                    double def_im = fft_def[i][px][1];
                    double ref_re = fft_ref[i][px][0];
                    double ref_im = fft_ref[i][px][1];
                    fft_def[i][px][0] = def_re * ref_re + def_im * ref_im;  // real part
                    fft_def[i][px][1] = def_im * ref_re - def_re * ref_im;  // imag part
                }

                // reverse fft
                fftw_execute(plans_inv[i]);

                // get max val:
                int peak_x = 0, peak_y = 0;
                double max_val = -1e9;
                for (int y = 0; y < w; ++y) {
                    for (int x = 0; x < w; ++x) {
                        double val = ifft_out[i][y * w + x];
                        if (val > max_val) {
                            max_val = val;
                            peak_x = x;
                            peak_y = y;
                        }
                    }
                }

                int peak_x_fftshift = fftshift(peak_x, w);
                int peak_y_fftshift = fftshift(peak_y, w);

                // update the shift
                if (i == 0){
                    shift_x[i][ss] = peak_x_fftshift;
                    shift_y[i][ss] = peak_y_fftshift;
                }
                else {
                    shift_x[i][ss] = shift_x_prev + peak_x_fftshift;
                    shift_y[i][ss] = shift_y_prev + peak_y_fftshift;
                }
                std::cout << ss_x << " " << ss_y << " " << peak_x_fftshift << " " << peak_y_fftshift << " " << shift_x_prev << " " << shift_y_prev << " " << shift_x[i][ss] << " " << shift_y[i][ss] << " " << max_val << std::endl;

            }
            std::cout << std::endl;

            // remove outliers
            remove_outliers_auto(shift_x[i], ssdata[i]);
            remove_outliers_auto(shift_y[i], ssdata[i]);
        }
    }
    
   
    void get_4nn(std::vector<int> &neighlist,
                const util::SubsetData ssdata,
                const util::SubsetData ssdata_prev) {

        const int num_subsets = ssdata.num;
        const int prev_num = ssdata_prev.num;
        const int size = ssdata.step;

        // For each subset, find 4 nearest neighbours in ssdata_prev
        for (int ss = 0; ss < num_subsets; ++ss) {

            const int ss_x = ssdata.coords[2 * ss];
            const int ss_y = ssdata.coords[2 * ss + 1];

            const double ss_x_centre = ss_x + size / 2.0;
            const double ss_y_centre = ss_y + size / 2.0;

            // Vector to store pairs of (distance, index)
            std::vector<std::pair<double, int>> dist_index_list;

            for (int nss = 0; nss < prev_num; ++nss) {
                int prev_x = ssdata_prev.coords[2 * nss];
                int prev_y = ssdata_prev.coords[2 * nss + 1];

                double dx = (prev_x + size / 2.0) - ss_x_centre;
                double dy = (prev_y + size / 2.0) - ss_y_centre;
                double dist_sq = dx * dx + dy * dy;

                dist_index_list.emplace_back(dist_sq, nss);
            }

            // Partial sort to get 4 nearest neighbours
            if (dist_index_list.size() > 4) {
                std::nth_element(dist_index_list.begin(), dist_index_list.begin() + 4, dist_index_list.end());
                dist_index_list.resize(4);
            }

            // Sort the 4 nearest neighbours by distance (optional, for consistent ordering)
            std::sort(dist_index_list.begin(), dist_index_list.end());

            // Store neighbours indices into neighlist
            for (int i = 0; i < 4; ++i) {
                neighlist[ss * 4 + i] = dist_index_list[i].second;
            }
        }
    }

    void get_neighlist(std::vector<int> &neighlist,
                       const util::SubsetData ssdata,
                       const util::SubsetData ssdata_prev) {


        // loop over the subsets for the window
        for (int ss = 0; ss < ssdata.num; ss++){

            const int size = ssdata.step;
            const int step = ssdata.size;
            const int ss_x = ssdata.coords[2*ss];
            const int ss_y = ssdata.coords[2*ss+1];
            
            // centre of 
            const double ss_x_centre = ss_x + ssdata.step / 2.0;
            const double ss_y_centre = ss_y + ssdata.step / 2.0;

            double min_dist = std::numeric_limits<double>::max();
            int best_nss = -1;
            bool collision_flag = false;

            for (int nss = 0; nss < ssdata_prev.num; nss++) {

                int prev_x = ssdata_prev.coords[2*nss];
                int prev_y = ssdata_prev.coords[2*nss+1];


                // AABB colision check
                if (prev_x < ss_x + step &&
                    prev_x + step > ss_x &&
                    prev_y < ss_y + step &&
                    prev_y + step > ss_y) {

                    // early exit when we get an overlap with prev window
                    neighlist[ss] = nss;
                    collision_flag = true;
                    break;
                }



                // distance to center
                double dx = (prev_x + size / 2.0) - ss_x_centre;
                double dy = (prev_y + size / 2.0) - ss_y_centre;
                double dist_sq = dx * dx + dy * dy;

                if (dist_sq < min_dist) {
                    min_dist = dist_sq;
                    best_nss = nss;
                }
            }

            // if overlap found fall use nearest neighbour.
            if (collision_flag == false) {
                neighlist[ss] = best_nss;
            }

        }
    }

    void cleanup(std::vector<util::SubsetData> ssdata){
        std::cout << "cleanup" << std::endl;

        for (size_t i = 0; i < ssdata.size(); i++){
            fftw_destroy_plan(plans_def[i]);
            fftw_destroy_plan(plans_ref[i]);
            fftw_free(fft_def[i]);
            fftw_free(fft_ref[i]);
        }

        ssdata.clear();
        ss_def.clear();
        ss_ref.clear();
        plans_def.clear();
        plans_ref.clear();
        fft_def.clear();
        fft_ref.clear();
    }

    inline int fftshift(int peak, int w){
        return (peak < w / 2) ? peak: peak - w;
    }
}
