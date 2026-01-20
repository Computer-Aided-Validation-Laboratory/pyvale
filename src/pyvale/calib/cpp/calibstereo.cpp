// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>

// Eigen header files
#include <Eigen/Core>

// pyvale header files
#include "./calibopt.hpp"
#include "./calibstereo.hpp"


void stereo_calibration(const std::vector<double> &init_params,
                        const std::vector<double> &dots_cam0,
                        const std::vector<double> &dots_cam1,
                        const std::vector<double> &grid,
                        const std::vector<int> &lengths,
                        const int px_hori, const int px_vert, const int num_img){

    int num_params = 4*2 + 5*2 + 3 + 3 + 6*num_img;
    optimization::Parameters opt(init_params.size(), 100, 0.001, px_hori, px_vert);

    // assign initial guess for parameter values
    for (int i = 0; i < init_params.size(); i++){
        opt.p[i] = init_params[i];
    }

    // run optimization routine
    optimization::Output output = optimization::bundle_adjustment(opt, dots_cam0, dots_cam1, grid, num_img, lengths);

    // calculate the error for each image based on the final residuals
    std::vector<double> err0(num_img,0.0);
    std::vector<double> err1(num_img,0.0);

    std::string formulation = "MatchID";

    int img_start = 0;
    for (int img = 0; img < num_img; img++){
        for (int d = 0; d < lengths[img]; d++){

            const int idx_x = img_start+2*d+0;
            const int idx_y = img_start+2*d+1;


            // length diff for cam0
            const double dx0 = output.proj0[idx_x] - dots_cam0[idx_x];
            const double dy0 = output.proj0[idx_y] - dots_cam0[idx_y];

            // length diff for cam1
            const double dx1 = output.proj1[idx_x] - dots_cam1[idx_x];
            const double dy1 = output.proj1[idx_y] - dots_cam1[idx_y];

            if (formulation=="MatchID"){
                err0[img] += (dx0*dx0 + dy0*dy0)/(2.0*lengths[img]);
                err1[img] += (dx1*dx1 + dy1*dy1)/(2.0*lengths[img]);
            }
            else if (formulation=="RMS"){
                err0[img] += (dx0*dx0 + dy0*dy0)/(lengths[img]);
                err1[img] += (dx1*dx1 + dy1*dy1)/(lengths[img]);
            }
            else if (formulation=="mean"){
                err0[img] += std::sqrt(dx0*dx0 + dy0*dy0)/(lengths[img]);
                err1[img] += std::sqrt(dx1*dx1 + dy1*dy1)/(lengths[img]);
            }
            else {
                std::cout << "Unknown Reprojection Error formulation: '" << formulation << "'." << std::endl;
                std::cout << "Allowed options: 'MatchID', 'RMS', 'mean'." << std::endl;
            }



        }

        if (formulation=="RMS"){
            err0[img] = std::sqrt(err0[img]);
            err1[img] = std::sqrt(err1[img]);
        }

        img_start += lengths[img];
        std::cout << "error image " << img << ": " << err0[img] << " (L) " << err1[img] << " (R) " << std::endl;
    }
}





