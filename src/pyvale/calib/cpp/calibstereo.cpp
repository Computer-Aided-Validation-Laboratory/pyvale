// ================================================================================)
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#define _USE_MATH_DEFINES
#include <iostream>
#include <cstring>
#include <omp.h>
#include <vector>
#include <math.h>
#include <iomanip>
#include <stdexcept>

// Eigen header files
#include <Eigen/Core>

// pyvale header files
#include "./calibopt.hpp"
#include "./calibstereo.hpp"

#include "../../common_cpp/util.hpp"


StereoCalibResult calibrate_stereo(const std::vector<double> &init_params,
                                   const std::vector<double> &dots_cam0, // 2d
                                   const std::vector<double> &dots_cam1, // 2d
                                   const std::vector<double> &grid, // 3d
                                   const std::vector<int> &lengths,
                                   const int px_hori, const int px_vert,
                                   const int num_img,
                                   const bool optimize_distortion,
                                   const double precision,
                                   const int max_iter,
                                   const ReprojError reproj_error){

    int num_params = 5*2 + 5*2 + 3 + 3 + 6*num_img;
    
    if (init_params.size() != num_params){
        throw std::invalid_argument("Unexpected number of calibration parameters");
    }

    optimization::Parameters opt(init_params.size(), max_iter, precision);

    // assign initial guess for parameter values
    for (int i = 0; i < init_params.size(); i++){
        opt.p[i] = init_params[i];
    }

    if (!optimize_distortion) {
        for (int i = 5; i < 10; ++i) {
            opt.p[i] = 0.0;
            opt.vary[i] = false;
        }
        for (int i = 15; i < 20; ++i) {
            opt.p[i] = 0.0;
            opt.vary[i] = false;
        }
    }

    // run optimization routine
    optimization::Output output = optimization::bundle_adjustment(opt, dots_cam0, dots_cam1, grid, num_img, lengths);
    common_util::info_out("Optimization finished after " + std::to_string(output.iter+1) + " iterations.", "");

    if ((output.iter+1) >= max_iter) {
        common_util::info_out("WARN: Maximum number of iterations reached.", "");
        common_util::info_out("WARN: Optimization may not have converged.", "");
    }


    common_util::info_out("Calculating final reprojection errors for each image...", "");

    // calculate the error for each image based on the final residuals
    std::vector<double> err0(num_img,0.0);
    std::vector<double> err1(num_img,0.0);

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

            if (reproj_error==ReprojError::MSE){
                err0[img] += (dx0*dx0 + dy0*dy0)/(2.0*lengths[img]);
                err1[img] += (dx1*dx1 + dy1*dy1)/(2.0*lengths[img]);
            }
            else if (reproj_error==ReprojError::RMSE){
                err0[img] += (dx0*dx0 + dy0*dy0)/(lengths[img]);
                err1[img] += (dx1*dx1 + dy1*dy1)/(lengths[img]);
            }
            else if (reproj_error==ReprojError::MEAN){
                err0[img] += std::sqrt(dx0*dx0 + dy0*dy0)/(lengths[img]);
                err1[img] += std::sqrt(dx1*dx1 + dy1*dy1)/(lengths[img]);
            }
            else {
                throw std::invalid_argument(
                    "Unknown reprojection error formulation: '"
                    "'. Allowed options are: 'MSE', 'RMS', 'mean'."
                );
            }



        }

        if (reproj_error==ReprojError::RMSE){
            err0[img] = std::sqrt(err0[img]);
            err1[img] = std::sqrt(err1[img]);
        }

        img_start += 2*lengths[img];
        //std::cout << "error image " << img << ": " << err0[img] << " (L) " << err1[img] << " (R) " << std::endl;




         int start_cam0 = 26;

         // rotation vector
         Eigen::Vector3d rvec0(opt.p[start_cam0 + img*6 + 0],
                                 opt.p[start_cam0 + img*6 + 1],
                                 opt.p[start_cam0 + img*6 + 2]);

         // translation vector
         Eigen::Vector3d tvec0(opt.p[start_cam0 + img*6 + 3],
                                 opt.p[start_cam0 + img*6 + 4],
                                 opt.p[start_cam0 + img*6 + 5]);

         // rotation matrix
         Eigen::Matrix3d R0 = optimization::rodrigues_to_matrix(rvec0);
         Eigen::Vector3d rvec_stereo(opt.p[20], opt.p[21], opt.p[22]);
         Eigen::Vector3d tvec_stereo(opt.p[23], opt.p[24], opt.p[25]);
         Eigen::Matrix3d R_stereo = optimization::rodrigues_to_matrix(rvec_stereo);


         // Cam1 pose. From cam0 + stereo
         Eigen::Matrix3d R1 = R_stereo * R0;
         Eigen::Vector3d rvec1 = optimization::matrix_to_rodrigues(R1);
         Eigen::Vector3d tvec1 = R_stereo * tvec0 + tvec_stereo;
    }

    Calib calib{
        .cam0 = {opt.p[0], opt.p[1], opt.p[2], opt.p[3], opt.p[4],
                {opt.p[5], opt.p[6], opt.p[7], opt.p[8], opt.p[9]}},
        .cam1 = {opt.p[10], opt.p[11], opt.p[12], opt.p[13], opt.p[14],
                {opt.p[15], opt.p[16], opt.p[17], opt.p[18], opt.p[19]}},
        .translation = {opt.p[23], opt.p[24], opt.p[25]},
        .rotation = {opt.p[20], opt.p[21], opt.p[22]},
    };


    common_util::info_out("Calibration Finished.", "");

    return {std::move(calib), std::move(err0), std::move(err1)};
}
