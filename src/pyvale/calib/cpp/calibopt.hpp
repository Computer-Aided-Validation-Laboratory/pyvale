// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef CALIBOPT_H
#define CALIBOPT_H

// STD library Header files
#include <iostream>
#include <cstring>
#include <locale>
#include <omp.h>
#include <vector>

// common_cpp header filesfiles
#include <Eigen/Dense>


namespace optimization {

    struct Parameters {
        int num_img;
        int num_params;
        double lambda; // damping
        double costp; // cost function for current P values
        double costpdp; // cost function for P+deltaP values
        std::vector<double> p; // hard coded optimzation parameters
        std::vector<double> dp; // deltaP
        std::vector<double> pdp; // P + deltaP
        std::vector<bool> vary; // whether each parameter is optimized
        int max_iter;
        double precision;
        int px_vert;
        int px_hori;
        Eigen::MatrixXd jac;
        Eigen::MatrixXd H;


        // Constructor to initialize vectors and other parameters
        Parameters(int num_params_, int max_iter_, 
                double precision_)
            :
            num_params(num_params_),
            lambda(0.01),
            costp(0.0),
            costpdp(0.0),
            p(num_params, 0.0),
            dp(num_params, 0.0),
            pdp(num_params, 0.0),
            vary(num_params, true),
            max_iter(max_iter_),
            precision(precision_) {}
    };


    struct Output {
        Eigen::VectorXd residuals;
        std::vector<double> proj0;
        std::vector<double> proj1;
        int iter;
    };

    // master optimization routine
    optimization::Output bundle_adjustment(Parameters &opt, 
                                           const std::vector<double> &dots_cam0, 
                                           const std::vector<double> &dots_cam1,
                                           const std::vector<double> &grid, 
                                           const size_t num_img, 
                                           const std::vector<int> &lengths);

    // single iteration of optimization
    void iterate_cost(Parameters &opt, 
                      const std::vector<double> &dots_cam0, 
                      const std::vector<double> &dots_cam1, 
                      const std::vector<double> &grid, 
                      const size_t num_img, 
                      const std::vector<int> &lengths, 
                      const int iter);

    // calculate jacobian
    Eigen::MatrixXd calc_jac(std::vector<double> &p, 
                             const Eigen::VectorXd &r, 
                             const std::vector<double> &dots_cam0, 
                             const std::vector<double> &dots_cam1, 
                             const std::vector<double> &grid, 
                             const size_t num_img, 
                             const int iter,
                             const std::vector<int> &lengths);

    // calculate residuals
    optimization::Output calc_residuals(std::vector<double> &p, 
                                        const std::vector<double> &dots_cam0, 
                                        const std::vector<double> &dots_cam1, 
                                        const std::vector<double> &grid, 
                                        const size_t num_img,
                                        const std::vector<int> &lengths, 
                                        const int iter,
                                        const bool print_flag);


    // Function to convert Rodrigues rotation vector to rotation matrix using Eigen
    Eigen::Matrix3d rodrigues_to_matrix(const Eigen::Vector3d &rvec);
    Eigen::Vector3d matrix_to_rodrigues(const Eigen::Matrix3d &R);
}

#endif // CALIBOPT_H
