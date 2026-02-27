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

// Eigen header files
#include <Eigen/Core>

// pyvale header files
#include "./calibopt.hpp"
#include "./calibstereo.hpp"


void calibrate_stereo(const std::vector<double> &init_params,
                        const std::vector<double> &dots_cam0, // 2d
                        const std::vector<double> &dots_cam1, // 2d
                        const std::vector<double> &grid, // 3d
                        const std::vector<int> &lengths,
                        const int px_hori, const int px_vert, const int num_img){

    int num_params = 5*2 + 5*2 + 3 + 3 + 6*num_img;
    
    if (init_params.size() != num_params){
        std::cout << "ERROR: unexpected number of parameters" << std::endl;
        exit(0);
    }

    optimization::Parameters opt(init_params.size(), 100, 0.00001);

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


    std::cout << "num_img " << num_img << std::endl;

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

        img_start += 2*lengths[img];
        std::cout << "error image " << img << ": " << err0[img] << " (L) " << err1[img] << " (R) " << std::endl;




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

        // filename Fx Fy Fs K1 K2 K3 Cx Cy Theta Phi Psi Tx Ty Tz Error
        // print calibration parameters in matchid output format PER FILE
        // std::cout << "test" << std::setw(4) << std::setfill('0') << img << std::setw(1) << "_0 ";
        // std::cout << opt.p[0] << " " << opt.p[1] << " " << opt.p[2] << " " << " " << opt.p[5] << " " << opt.p[6] << " " << opt.p[9] << " " << opt.p[3] << " " << opt.p[4] << " ";
        // std::cout << rvec0[0] * (180.0 / M_PI) << " " << rvec0[1] * (180.0 / M_PI) << " " << rvec0[2] * (180.0 / M_PI) << " ";
        // std::cout << tvec0[0] << " " << tvec0[1] << " " << tvec0[2] << " ";
        // std::cout << err0[img] << std::endl;
        //
        // std::cout << "test" << std::setw(4) << std::setfill('0') << img << std::setw(1) << "_1 ";
        // std::cout << opt.p[10] << " " << opt.p[11] << " " << opt.p[12] << " " << " " << opt.p[15] << " " << opt.p[16] << " " << opt.p[19] << " " << opt.p[13] << " " << opt.p[14] << " ";
        // std::cout << rvec1[0] * (180.0 / M_PI) << " " << rvec1[1] * (180.0 / M_PI) << " " << rvec1[2] * (180.0 / M_PI) << " ";
        // std::cout << tvec1[0] << " " << tvec1[1] << " " << tvec1[2] << " ";
        // std::cout << err1[img] << std::endl;

    }

    std::cout << std::endl;
    std::cout << std::setprecision(8);
    std::cout << "Cam0_Fx [pixels]; " << opt.p[0] << std::endl;
    std::cout << "Cam0_Fy [pixels]; " << opt.p[1] << std::endl;
    std::cout << "Cam0_Fs [pixels]; " << opt.p[2] << std::endl;
    std::cout << "Cam0_Kappa 1; " << opt.p[5] << std::endl;
    std::cout << "Cam0_Kappa 2; " << opt.p[6] << std::endl;
    std::cout << "Cam0_Kappa 3; " << opt.p[9] << std::endl;
    std::cout << "Cam0_P1; " << opt.p[7] << std::endl;
    std::cout << "Cam0_P2; " << opt.p[8] << std::endl;
    std::cout << "Cam0_Cx [pixels]; " << opt.p[3] << std::endl;
    std::cout << "Cam0_Cy [pixels]; " << opt.p[4] << std::endl;
    std::cout << "Cam1_Fx [pixels]; " << opt.p[10] << std::endl;
    std::cout << "Cam1_Fy [pixels]; " << opt.p[11] << std::endl;
    std::cout << "Cam1_Fs [pixels]; " << opt.p[12] << std::endl;
    std::cout << "Cam1_Kappa 1; " << opt.p[15] << std::endl;
    std::cout << "Cam1_Kappa 2; " << opt.p[16] << std::endl;
    std::cout << "Cam1_Kappa 3; " << opt.p[19] << std::endl;
    std::cout << "Cam1_P1; " << opt.p[17] << std::endl;
    std::cout << "Cam1_P2; " << opt.p[18] << std::endl;
    std::cout << "Cam1_Cx [pixels]; " << opt.p[13] << std::endl;
    std::cout << "Cam1_Cy [pixels]; " << opt.p[14] << std::endl;
    std::cout << "Tx [mm]; " << opt.p[23] << std::endl;
    std::cout << "Ty [mm]; " << opt.p[24] << std::endl;
    std::cout << "Tz [mm]; " << opt.p[25] << std::endl;
    std::cout << "Theta [deg]; " << opt.p[20] * (180.0 / M_PI) << std::endl;
    std::cout << "Phi [deg]; " << opt.p[21] * (180.0 / M_PI) << std::endl;
    std::cout << "Psi [deg]; " << opt.p[22] * (180.0 / M_PI) << std::endl;
}





