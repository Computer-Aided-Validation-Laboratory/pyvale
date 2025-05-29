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

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"
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

    std::vector<int> neigh_x;
    std::vector<int> neigh_y;
    std::vector<int> neigh_valid;
    std::vector<int> max_x;
    std::vector<int> max_y;

    // values from the bilinear interpolation of the neighouring
    // points from previous window
    double interp_x, interp_y;

    void init(bool *img_roi, util::Config conf, int *windows, int n_windows){

        neigh_x.resize(4);
        neigh_y.resize(4);
        neigh_valid.resize(4);
        max_x.resize(n_windows);
        max_y.resize(n_windows);

        for (int i = 0; i < n_windows; i++) {

            int w = windows[i];

            // generate subset information
            ssdata.push_back(util::gen_ss_list(img_roi, w/2, w,
                                          conf.px_hori, conf.px_vert));

            ss_def.push_back(util::Subset(w));
            ss_ref.push_back(util::Subset(w));
            fft_def.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));
            fft_ref.push_back((fftw_complex*) fftw_malloc(sizeof(fftw_complex) * w * (w/2+1)));

            std::cout << "ss_def size: " << ss_def[i].size << std::endl;
            std::cout << "ss_def vals size: " << ss_def[i].vals.size() << std::endl;

            // forward fftw plans
            plans_def.push_back(fftw_plan_dft_r2c_2d(w, w, &ss_def[i].vals[0], fft_def[i], FFTW_ESTIMATE));
            plans_ref.push_back(fftw_plan_dft_r2c_2d(w, w, &ss_ref[i].vals[0], fft_ref[i], FFTW_ESTIMATE));
            
            // inverse fftw plans
            ifft_out.push_back(std::vector<double>(w*w));
            plans_inv.push_back(fftw_plan_dft_c2r_2d(w, w, fft_def[i], ifft_out[i].data(), FFTW_ESTIMATE));


            // shifts for each window size
            shift_x.push_back(std::vector<double>(ssdata[i].num_ss_x*ssdata[i].num_ss_y, 0.0));
            shift_y.push_back(std::vector<double>(ssdata[i].num_ss_x*ssdata[i].num_ss_y, 0.0));



            // maximum values for each subset window
            max_x[i] = std::numeric_limits<int>::min();
            max_y[i] = std::numeric_limits<int>::min();

            for (size_t j = 0; j < ssdata[i].coords.size(); j += 2) {
                int x = ssdata[i].coords[j];
                int y = ssdata[i].coords[j + 1];
                if (x > max_x[i]) max_x[i] = x;
                if (y > max_y[i]) max_y[i] = y;
            }
            std::cout << "Maxval: " << i << " " << max_x[i] << " " << max_y[i] << std::endl;

        }
        std::cout << "Finished FFT initialisation" << std::endl;
    }



    void mgwd(double *img_def, double *img_ref,
              int *windows, int n_windows, 
              util::Config conf){



        // Loop over window size
        for (int i = 0; i < n_windows; i++){

            int w = windows[i];

            std::cout << "window: " << windows[i] << std::endl;


            // loop over subsets for each window size
            for (int ss = 0; ss < ssdata[i].num; ss++){

                int ss_x = ssdata[i].coords[2*ss];
                int ss_y = ssdata[i].coords[2*ss+1];

                // window has to always be decreasing in size
                if (i > 0) {

                    // assuming always 50% overlap
                    int spacing = windows[i-1]/2;

                    // get the index of the subset in the previous window
                    int x0 = (int)(ss_x / spacing) * spacing;
                    int y0 = (int)(ss_y / spacing) * spacing;
                    int x1 = x0 + spacing;
                    int y1 = y0 + spacing;

                    // if on a boundary and no valid neighbours, look left
                    if (ss_x >= max_x[i-1]) {
                        x1 = max_x[i-1];
                        x0 = x1 - spacing;
                    } else {
                        x0 = (ss_x / spacing) * spacing;
                        x1 = x0 + spacing;
                    }
                    if (ss_y >= max_y[i-1]) {
                        y1 = max_y[i-1];
                        y0 = y1 - spacing;
                    } else {
                        y0 = (ss_y / spacing) * spacing;
                        y1 = y0 + spacing;
                    }

                    int i0 = (x0 / spacing);
                    int i1 = (x1 / spacing);
                    int j0 = (y0 / spacing);
                    int j1 = (y1 / spacing);

                    int idx00 = j0 * ssdata[i-1].num_ss_x + i0; // top-left
                    int idx10 = j0 * ssdata[i-1].num_ss_x + i1; // top-right
                    int idx01 = j1 * ssdata[i-1].num_ss_x + i0; // bottom-left
                    int idx11 = j1 * ssdata[i-1].num_ss_x + i1; // bottom-right
                    
                    bool v00 = ssdata[i-1].mask[idx00];
                    bool v10 = ssdata[i-1].mask[idx10];
                    bool v01 = ssdata[i-1].mask[idx01];
                    bool v11 = ssdata[i-1].mask[idx11];

                    int valid_count = v00 + v10 + v01 + v11;
                    

                    double tx = (ss_x - static_cast<double>(x0)) / spacing;
                    double ty = (ss_y - static_cast<double>(y0)) / spacing;

                    
                    // testing with dummy values
                    // shift_x[i-1][idx00] = 10;
                    // shift_x[i-1][idx10] = 20;
                    // shift_x[i-1][idx01] = 30;
                    // shift_x[i-1][idx11] = 40;
                    
                    double shift_x00 = v00 ? shift_x[i-1][idx00] : 0.0;
                    double shift_x10 = v10 ? shift_x[i-1][idx10] : 0.0;
                    double shift_x01 = v01 ? shift_x[i-1][idx01] : 0.0;
                    double shift_x11 = v11 ? shift_x[i-1][idx11] : 0.0;
                    double shift_y00 = v00 ? shift_y[i-1][idx00] : 0.0;
                    double shift_y10 = v10 ? shift_y[i-1][idx10] : 0.0;
                    double shift_y01 = v01 ? shift_y[i-1][idx01] : 0.0;
                    double shift_y11 = v11 ? shift_y[i-1][idx11] : 0.0;

                    if ((ss_x == 368) && (ss_y == 208)){
                         std::cout << "ss " << " " << ss_x << " " << ss_y << std::endl;
                         std::cout << "x0 " << " " << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;
                         std::cout << "i0 " << " " << i0 << " " << j0 << " " << i1 << " " << j1 << std::endl;
                         std::cout << "id " << " " << idx00 << " " << idx10 << " " << idx01 << " " << idx11 << std::endl;
                         std::cout << "v0 " << " " << v00 << " " << v10 << " " << v01 << " " << v11 << std::endl;
                         std::cout << "tx " << " " << tx << " " << ty << std::endl;
                         std::cout << "va " << " " << valid_count << std::endl;
                         std::cout << "shiftx " << " " << shift_x00 << " " << shift_x10 << " " << shift_x01 << " " << shift_x11 << std::endl;
                         std::cout << "shifty " << " " << shift_y00 << " " << shift_y10 << " " << shift_y01 << " " << shift_y11 << std::endl;
                    }

                    if (valid_count == 4) {
                        interp_x = (1.0-tx) * (1-ty) * shift_x00 + tx * (1-ty) * shift_x10 + (1-tx) * ty * shift_x01 + tx * ty * shift_x11;
                        interp_y = (1.0-tx) * (1-ty) * shift_y00 + tx * (1-ty) * shift_y10 + (1-tx) * ty * shift_y01 + tx * ty * shift_y11;
                    }
                    else if (valid_count >= 2) {
                        // Fallback: do linear interpolation or average available
                        double res_x = 0.0;
                        double res_y = 0.0;
                        double weight = 0.0;
                        if (v00) { 
                            res_x += (1 - tx) * (1 - ty) * shift_x00; 
                            res_y += (1 - tx) * (1 - ty) * shift_y00; 
                            weight += (1 - tx) * (1 - ty); 
                        }
                        if (v10) {
                        res_x += tx * (1 - ty) * shift_x10; 
                        res_y += tx * (1 - ty) * shift_y10; 
                        weight += tx * (1 - ty); }
                        if (v01) {
                            res_x += (1 - tx) * ty * shift_x01; 
                            res_y += (1 - tx) * ty * shift_y01; 
                            weight += (1 - tx) * ty; 
                        }
                        if (v11) {
                            res_x += tx * ty * shift_x11; 
                            res_y += tx * ty * shift_y11; 
                            weight += tx * ty; 
                        }
                        interp_x = weight > 0 ? res_x / weight : 0.0;
                        interp_y = weight > 0 ? res_y / weight : 0.0;
                    }
                    else if (valid_count == 1) {
                        if (v00) { interp_x = shift_x00; interp_y = shift_y00;}
                        if (v10) { interp_x = shift_x10; interp_y = shift_y10;}
                        if (v01) { interp_x = shift_x01; interp_y = shift_y01;}
                        if (v11) { interp_x = shift_x11; interp_y = shift_y11;}
                    }
                    else {
                        std::cerr << "NO NEIGHBOURS FOR POINT " << ss_x << " " << ss_y << std::endl;
                        std::cout << "ss " << " " << ss_x << " " << ss_y << std::endl;
                        std::cout << "x0 " << " " << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;
                        std::cout << "i0 " << " " << i0 << " " << j0 << " " << i1 << " " << j1 << std::endl;
                        std::cout << "id " << " " << idx00 << " " << idx10 << " " << idx01 << " " << idx11 << std::endl;
                        std::cout << "v0 " << " " << v00 << " " << v10 << " " << v01 << " " << v11 << std::endl;
                        std::cout << "tx " << " " << tx << " " << ty << std::endl;
                        std::cout << "va " << " " << valid_count << std::endl;
                        std::cout << "shiftx " << " " << shift_x00 << " " << shift_x10 << " " << shift_x01 << " " << shift_x11 << std::endl;
                        std::cout << "shifty " << " " << shift_y00 << " " << shift_y10 << " " << shift_y01 << " " << shift_y11 << std::endl;
                        std::cout << "num_ss_x " << ssdata[i].num_ss_x << "num_ss_y " << ssdata[i].num_ss_y << std::endl;
                        exit(0);
                    }
                    // bilinear interpolation result
                    //std::cout << interp_x << " " << interp_y << std::endl;
                }


                //if ((i > 0) && (ss >= 3)){
                //    exit(0);
                //}

                // get the deformed subset
                util::extract_ss(ss_def[i],ss_x, ss_y, conf.px_hori,
                                 conf.px_vert, img_def);

                // get the reformed subset
                util::extract_ss(ss_ref[i], ss_x-interp_x, ss_y-interp_y, conf.px_hori,
                                 conf.px_vert, img_ref);

                // calculate mean for each subset
                double mean_def = 0.0;
                double mean_ref = 0.0;
                for (int px = 0; px < w*w; px++){
                    mean_def += ss_def[i].vals[px];
                    mean_ref += ss_ref[i].vals[px];
                }
                mean_def /= w*w;
                mean_ref /= w*w;


                // subtract mean from each subset
                for (int px = 0; px < w*w; px++){
                    ss_def[i].vals[px] -= mean_def;
                    ss_ref[i].vals[px] -= mean_ref;
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

                // get max val
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


                // update the shift
                int idx = (ss_y/(w/2)) * ssdata[i].num_ss_x + ss_x/(w/2);
                if (i == 0){
                    shift_x[i][idx] = peak_x;
                    shift_y[i][idx] = peak_y;
                }
                else {
                    shift_x[i][idx] = shift_x[i-1][idx] + peak_x;
                    shift_y[i][idx] = shift_y[i-1][idx] + peak_y;
                }

                if ((ss_x == 368) && (ss_y == 208)){
                    for (int y = 0; y < w; ++y) {
                        for (int x = 0; x < w; ++x) {
                         std::cout << x << " " << y << " " << ss_def[i].vals[y*w+x] << " " << ss_ref[i].vals[y*w+x] << " " << ifft_out[i][y*w+x] << " ";
                         std::cout << " " << peak_x << " " << peak_y << " " << interp_x << " " << interp_y << std::endl;
                        }
                    }
                    exit(0);
                }
                
                std::cout << ss_x << " " << ss_y << " " << peak_x << " " << peak_y << std::endl;
            }
            std::cout << std::endl;
        }
    }


    void cleanup(){
        std::cout << "cleanup" << std::endl;

        for (int i = 0; i < ssdata.size(); i++){
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


}
