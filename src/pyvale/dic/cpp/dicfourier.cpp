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

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"
#include "./dicfourier.hpp"

namespace fourier {

    std::vector<util::SubsetData> ssdata;
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


    void init(bool *img_roi, util::Config conf, int *windows, int n_windows){

        for (int i = 0; i < n_windows; i++) {

            int w = windows[i];

            // generate subset information
            ssdata.push_back(util::gen_ss_list(img_roi, w/2, w,
                                          conf.px_hori, conf.px_vert));

            ss_def.push_back(util::Subset(w));
            ss_ref.push_back(util::Subset(w));
            fft_def.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));
            fft_ref.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));

            //std::cout << "ss_def size: " << ss_def[i].size << std::endl;
            //std::cout << "ss_def vals size: " << ss_def[i].vals.size() << std::endl;

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
                neighlist.push_back(std::vector<int>(ssdata[i].num, 0));
                get_neighlist(neighlist[i-1], ssdata[i], ssdata[i-1]);
            }


        }

        std::cout << "Finished FFT initialisation" << std::endl;
    }



    void mgwd(double *img_def, double *img_ref,
              int *windows, int n_windows, 
              util::Config conf){

    

        // Loop over window size
        for (int i = 0; i < n_windows; i++){

            int w = windows[i];

            int ss_x, ss_y;
            int shift_x_prev = 0;
            int shift_y_prev = 0;

            // loop over subsets for each window size
            for (int ss = 0; ss < ssdata[i].num; ss++){

                ss_x = ssdata[i].coords[2*ss];
                ss_y = ssdata[i].coords[2*ss+1];

                // window has to always be decreasing in size
                if (i > 0) {

                    // assuming always 50% overlap
                    shift_x_prev = shift_x[i-1][neighlist[i-1][ss]];
                    shift_y_prev = shift_y[i-1][neighlist[i-1][ss]];
                    
                    // debugging
                    //std::cout << ss_x << " " << ss_y << " " << " " << neighlist[i][ss] << " " << shift_x_prev << " " << shift_y_prev << std::endl;

                    // TODO: Bilinear Interpolation
                    //int prev_step = windows[i-1]/2;
                    //double sx[4], sy[4], tx, ty;
                    //interp_x = (1.0-tx) * (1-ty) * sx[0] + tx * (1-ty) * sx[1] + (1-tx) * ty * sx[2] + tx * ty * sx[3];
                    //interp_y = (1.0-tx) * (1-ty) * sy[0] + tx * (1-ty) * sy[1] + (1-tx) * ty * sy[2] + tx * ty * sy[3];


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

                if (((ss_x == 1536) && (ss_y == 2560) && (i==1)) || ((ss_x ==  7680) && (ss_y == 2048) && (i==1))){
                    std::cout << "MEAN " << mean_ref << " " << mean_def << " " << std_ref << " " << std_def << std::endl;
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
                        // debugging
                        if (((ss_x == 1536) && (ss_y == 2560) && (i==1)) || ((ss_x ==  7680) && (ss_y == 2048) && (i==1)))
                            std::cout << x << " " << y << " " << ss_ref[i].vals[y*w + x] << " " << ss_def[i].vals[y*w + x] << " " << val << std::endl;
                        if (val > max_val) {
                            max_val = val;
                            peak_x = x;
                            peak_y = y;
                        }
                    }
                }
                if (((ss_x == 1536) && (ss_y == 2560) && (i==1)) || ((ss_x ==  7680) && (ss_y == 2048) && (i==1)))
                    std::cout << std::endl;

                if ((ss_x == 1536) && (ss_y == 2560) && (i==1))
                    exit(0);

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
                std::cout << ss_x << " " << ss_y << " " << peak_x_fftshift << " " << peak_y_fftshift << " " << shift_x_prev << " " << shift_y_prev << " " << shift_x[i][ss] << " " << shift_y[i][ss] << std::endl;


                // update the shift GRID
                //int idx = (ss_y/(w/2)) * ssdata[i].num_ss_x + ss_x/(w/2);
                //if (i == 0){
                //    shift_x[i][idx] = peak_x;
                //    shift_y[i][idx] = peak_y;
                //}
                //else {
                //    shift_x[i][idx] = interp_x + peak_x;
                //    shift_y[i][idx] = interp_y + peak_y;
                //}
                //std::cout << ss_x << " " << ss_y << " " << peak_x << " " << peak_y << " " << shift_x[i][idx] << " " << shift_y[i][idx] << std::endl;


                // DEBUGGING
                //if ((ss_x == 368) && (ss_y == 208)){
                //    for (int y = 0; y < w; ++y) {
                //        for (int x = 0; x < w; ++x) {
                //         std::cout << x << " " << y << " " << ss_def[i].vals[y*w+x] << " " << ss_ref[i].vals[y*w+x] << " " << ifft_out[i][y*w+x] << " ";
                //         std::cout << " " << peak_x << " " << peak_y << " " << interp_x << " " << interp_y << std::endl;
                //        }
                //    }
                //    exit(0);
                //}

            }
            std::cout << std::endl;
        }
        //exit(0);
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
                
                //if ((ss_x == 4608) && (ss_y == 7424)){
                //    std::cout << ss_x << " " << ss_y << " " << prev_x << " " << prev_y << " " << nss << " " << best_nss << std::endl;
                //}

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
                //if ((ss_x == 4608) && (ss_y == 7424)){
                //    std::cout << ssdata.num << " " << ssdata_prev.num << std::endl;
                //    std::cout << ss_x << " " << ss_y << " " << prev_x << " " << prev_y << " " << dx << " " << dy << " " << dist_sq << " " << min_dist << " " << nss << " " << best_nss << std::endl;
                //}
                if (dist_sq < min_dist) {
                    min_dist = dist_sq;
                    best_nss = nss;
                }
            }

            // if overlap found fall use nearest neighbour.
            if (collision_flag == false) {
                neighlist[ss] = best_nss;
            }

            // debugging
            //int idx = neighlist[ss];
            //std::cout << ss_x << " " << ss_y << " " << neighlist[ss] << " " << ssdata_prev.coords[2*idx] << " " << ssdata_prev.coords[2*idx+1] << std::endl;
        }
    }

    // TODO: Linear interpolation of the previous shifts to populate new values.
    // void get_prev_shift_vals(double *sx, double *sy,
    //                          double &tx, double &ty,
    //                          const int ss_x, const int ss_y,
    //                          const int i, const int prev_step,
    //                          const int num_ss_x, const std::vector<bool> &mask) {
    //
    //     // Compute corners of the grid square
    //     int grid_vals[4];
    //     grid_vals[0] = (ss_x / prev_step) * prev_step;    // x0
    //     grid_vals[1] = (ss_y / prev_step) * prev_step;    // y0
    //     grid_vals[2] = grid_vals[0] + prev_step;     // x1
    //     grid_vals[3] = grid_vals[1] + prev_step;     // y1
    //
    //     tx = (ss_x - static_cast<double>(grid_vals[0])) / prev_step;
    //     ty = (ss_y - static_cast<double>(grid_vals[1])) / prev_step;
    //
    //     int i0 = grid_vals[0] / prev_step;
    //     int i1 = grid_vals[1] / prev_step;
    //     int j0 = grid_vals[2] / prev_step;
    //     int j1 = grid_vals[3] / prev_step;
    //
    //     int grid_indx[4];
    //     grid_indx[0] = j0 * num_ss_x + i0; // idx00
    //     grid_indx[1] = j0 * num_ss_x + i1; // idx10
    //     grid_indx[2] = j1 * num_ss_x + i0; // idx01
    //     grid_indx[3] = j1 * num_ss_x + i1; // idx11
    //
    //     int valid[4];
    //     valid[0] = mask[grid_indx[0]];
    //     valid[1] = mask[grid_indx[1]];
    //     valid[2] = mask[grid_indx[2]];
    //     valid[3] = mask[grid_indx[3]];
    //
    //     sx[0] = valid[0] ? shift_x[i-1][grid_indx[0]] : 0.0;
    //     sx[1] = valid[1] ? shift_x[i-1][grid_indx[1]] : 0.0;
    //     sx[2] = valid[2] ? shift_x[i-1][grid_indx[2]] : 0.0;
    //     sx[3] = valid[3] ? shift_x[i-1][grid_indx[3]] : 0.0;
    //
    //     sy[0] = valid[0] ? shift_y[i-1][grid_indx[0]] : 0.0;
    //     sy[1] = valid[1] ? shift_y[i-1][grid_indx[1]] : 0.0;
    //     sy[2] = valid[2] ? shift_y[i-1][grid_indx[2]] : 0.0;
    //     sy[3] = valid[3] ? shift_y[i-1][grid_indx[3]] : 0.0;
    //
    //     int valid_count = valid[0] + valid[1] + valid[2] + valid[3];
    //
    //     while (valid_count == 0){
    //
    //          if (ss_x % prev_step == 0){
    //              grid_vals[0] += -prev_step/2;
    //              grid_vals[2] += -prev_step/2;
    //          }
    //          if (ss_y % prev_step == 0){
    //              grid_vals[1] += -prev_step/2;
    //              grid_vals[3] += -prev_step/2;
    //          }
    //
    //          tx = (ss_x - static_cast<double>(grid_vals[0])) / prev_step;
    //          ty = (ss_y - static_cast<double>(grid_vals[1])) / prev_step;
    //
    //          i0 = grid_vals[0] / prev_step;
    //          i1 = grid_vals[1] / prev_step;
    //          j0 = grid_vals[2] / prev_step;
    //          j1 = grid_vals[3] / prev_step;
    //
    //          grid_indx[0] = j0 * num_ss_x + i0; // idx00
    //          grid_indx[1] = j0 * num_ss_x + i1; // idx10
    //          grid_indx[2] = j1 * num_ss_x + i0; // idx01
    //          grid_indx[3] = j1 * num_ss_x + i1; // idx11
    //
    //          valid[0] = mask[grid_indx[0]];
    //          valid[1] = mask[grid_indx[1]];
    //          valid[2] = mask[grid_indx[2]];
    //          valid[3] = mask[grid_indx[3]];
    //
    //          sx[0] = valid[0] ? shift_x[i-1][grid_indx[0]] : 0.0;
    //          sx[1] = valid[1] ? shift_x[i-1][grid_indx[1]] : 0.0;
    //          sx[2] = valid[2] ? shift_x[i-1][grid_indx[2]] : 0.0;
    //          sx[3] = valid[3] ? shift_x[i-1][grid_indx[3]] : 0.0;
    //
    //          sy[0] = valid[0] ? shift_y[i-1][grid_indx[0]] : 0.0;
    //          sy[1] = valid[1] ? shift_y[i-1][grid_indx[1]] : 0.0;
    //          sy[2] = valid[2] ? shift_y[i-1][grid_indx[2]] : 0.0;
    //          sy[3] = valid[3] ? shift_y[i-1][grid_indx[3]] : 0.0;
    //
    //          valid_count = valid[0] + valid[1] + valid[2] + valid[3];
    //     }
    //
    //
    //     if (valid_count == 1) {
    //         for (int j = 0; j < 4; ++j) {
    //             if (valid[j]) {
    //                 sx[0] = sx[1] = sx[2] = sx[3] = sx[j];
    //                 sy[0] = sy[1] = sy[2] = sy[3] = sy[j];
    //                 break;
    //             }
    //         }
    //     }
    //     else if (valid_count == 2 || valid_count == 3) {
    //         double x_sum = 0.0, y_sum = 0.0;
    //         for (int j = 0; j < 4; ++j) {
    //             if (valid[j]) {
    //                 x_sum += sx[j];
    //                 y_sum += sy[j];
    //             }
    //         }
    //         double x_avg = x_sum / valid_count;
    //         double y_avg = y_sum / valid_count;
    //         for (int j = 0; j < 4; ++j) {
    //             if (!valid[j]) {
    //                 sx[j] = x_avg;
    //                 sy[j] = y_avg;
    //             }
    //         }
    //     }
    //
    //
    //     if ((ss_x == 768) && (ss_y == 224)){
    //         std::cout << "subset: " << " (" << ss_x << ", " << ss_y << ")" << std::endl;
    //         std::cout << "grid_vals: " << " (" << grid_vals[0] << ", " << grid_vals[1] << "), (" << grid_vals[2] << ", " << grid_vals[3] << ")" << std::endl;
    //         std::cout << "grid_indx: " << " (" << grid_indx[0] << ", " << grid_indx[1] << "), (" << grid_indx[2] << ", " << grid_vals[3] << ")" << std::endl;
    //         std::cout << "valid: " << " " << valid[0] << " " << valid[1] << " " << valid[2] << " " << valid[3] << std::endl;
    //         std::cout << "valid_count: " << " " << valid_count << std::endl;
    //         std::cout << "tx " << " " << tx << " " << ty << std::endl;
    //         std::cout << "sx " << " " << sx[0] << " " << sx[1] << " " << sx[2] << " " << sx[3] << std::endl;
    //         std::cout << "sy " << " " << sy[0] << " " << sy[1] << " " << sy[2] << " " << sy[3] << std::endl;
    //     }
    //
    // }

    void cleanup(){
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
