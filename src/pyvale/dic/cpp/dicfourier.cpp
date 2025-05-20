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

    std::vector<std::vector<double>> offset_x;
    std::vector<std::vector<double>> offset_y;


    // inverse fft
    std::vector<fftw_plan> plans_inv;
    std::vector<std::vector<double>> ifft_out;

    void init(bool *img_roi, util::Config conf, int *windows, int n_windows){

        std::cout << "init" << std::endl;

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
        }

        std::cout << "FINISHED INIT!!!" << std::endl;
    }



    void mgwd(double *img_def, double *img_ref,
              int *windows, int n_windows, 
              util::Config conf){


        std::cout << "mgwd" << std::endl;

        // Loop over window size
        for (int i = 0; i < n_windows; i++){

            int w = windows[i];

        std::cout << "window: " << windows[i] << std::endl;


            // loop over subsets for each window size
            for (int ss = 0; ss < ssdata[i].num; ss++){

                int ss_x = ssdata[i].coords[2*ss];
                int ss_y = ssdata[i].coords[2*ss+1];
            
                // going to assume that subset windows always decrease in
                // factors of 2
                if (i > 0) {
                    offset_x
                }

                // get the deformed subset
                util::extract_ss(ss_def[i],ss_x, ss_y, conf.px_hori,
                                 conf.px_vert, img_def);

                // get the deformed subset
                util::extract_ss(ss_ref[i], ss_x, ss_y, conf.px_hori,
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
                    fft_def[i][px][1] = def_re * ref_im - def_im * ref_re;  // imag part
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

                int dx = peak_x - w;
                int dy = peak_y - w;

                // update the offset
                offset_x[i][ss] = dx;
                offset_y[i][ss] = dy;


            }
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
