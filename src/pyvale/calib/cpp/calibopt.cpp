// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <iostream>
#include <cstring>
#include <omp.h>
#include <ostream>
#include <iomanip>
#include <vector>

// common_cpp Header files
#include "../../common_cpp/defines.hpp"

// Eigen Header Files
#include <Eigen/Dense>
#include <Eigen/Geometry>

// pyvale header files
#include "./calibopt.hpp"


namespace optimization {

    optimization::Output bundle_adjustment(Parameters &opt, const std::vector<double> &dots_cam0, const std::vector<double> &dots_cam1, 
                                           const std::vector<double> &grid, const size_t num_img, const std::vector<int> &lengths){

        int iter = 0;
        double ftol = 0;
        double xtol = 0;
        bool converged = false;
        opt.lambda = 0.001;

        while (iter < opt.max_iter) {

            // calculate updated parameters
            iterate_cost(opt, dots_cam0, dots_cam1, grid, num_img, lengths);

            ftol = std::abs(opt.costpdp - opt.costp);

            std::cout << iter << " " << opt.costp << " " << opt.costpdp << " " << ftol << std::endl;

            if ((xtol < opt.precision) && (ftol < opt.precision)) {
                converged=true;
                break;
            }
            iter++;
        }

        //return the residuals from the final iteration 
        opt.p = opt.pdp;
        optimization::Output final_results = calc_residuals(opt.p, dots_cam0, dots_cam1, grid, num_img, lengths, false);

        return final_results;
    }


    void iterate_cost(Parameters &opt, const std::vector<double> &dots_cam0, const std::vector<double> &dots_cam1, 
                    const std::vector<double> &grid, const size_t num_img, const std::vector<int> &lengths){


        // Compute residuals at current point. p get updated in this
        optimization::Output res = calc_residuals(opt.p, dots_cam0, dots_cam1, grid, num_img, lengths, false);

        // calculate jacobian
        Eigen::MatrixXd J = calc_jac(opt.p, res.residuals, dots_cam0, dots_cam1, grid, num_img, lengths);


        // calc gradient
        Eigen::VectorXd g = J.transpose() * res.residuals;

        // Hessian
        Eigen::MatrixXd H = J.transpose() * J;

        // (H + lambda*Identity)
        Eigen::MatrixXd A = H + opt.lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());

        // get change in parameters
        Eigen::VectorXd dp = A.ldlt().solve(-g);

        // Updated parameters
        for (int i = 0; i < opt.p.size(); i++) {
            opt.pdp[i] = opt.p[i] + dp(i);
        }


        // Evaluate new cost
        optimization::Output res_new = calc_residuals(opt.pdp, dots_cam0, dots_cam1, grid, num_img, lengths, false);
        opt.costp = 0.0;
        opt.costpdp = 0.0;
        for (int i = 0; i < res_new.residuals.size(); i++){
            opt.costp   += res.residuals(i) * res.residuals(i);
            opt.costpdp += res_new.residuals(i) * res_new.residuals(i);
        }

        //std::cout << opt.costp << " " << opt.costpdp << std::endl;
            
        // Accept or reject step
        if (opt.costpdp < opt.costp) {
            opt.p = opt.pdp;
            opt.lambda *= 0.1;
        } 
        else {
            opt.lambda *= 10.0;
        }

    }


    // Function to convert Rodrigues rotation vector to rotation matrix using Eigen
    Eigen::Matrix3d rodrigues_to_matrix(const Eigen::Vector3d &rvec) {

        double theta = rvec.norm();

        // handle tiny angles
        if (theta < 1e-10) {
            return Eigen::Matrix3d::Identity();
        }

        // Normalise
        Eigen::Vector3d axis = rvec / theta;

        // rotation
        Eigen::AngleAxisd rotation(theta, axis);
        return rotation.toRotationMatrix();
    }



    // Project 3D points to 2D with distortion (assuming radial + tangential distortion)
    std::vector<double> project_points(const std::vector<Eigen::Vector3d> &gridpoints_3d,
                                                const Eigen::Vector3d &rvec,
                                                const Eigen::Vector3d &tvec,
                                                const Eigen::Matrix3d &K,
                                                const Eigen::VectorXd &D) {


        Eigen::Matrix3d R = rodrigues_to_matrix(rvec);
        std::vector<double> projected(gridpoints_3d.size()*2);
        int count = 0;

        for (const auto& point : gridpoints_3d) {

            // Transform to camera coordinates
            Eigen::Vector3d p_cam = R * point + tvec;

            // Normalize
            const double x = p_cam(0) / p_cam(2);
            const double y = p_cam(1) / p_cam(2);

            // Apply distortion
            const double r2 = x*x + y*y;
            const double r4 = r2*r2;
            const double r6 = r4*r2;

            // Radial distortion
            const double radial = 1 + D(0)*r2 + D(1)*r4 + D(4)*r6;

            // Tangential distortion
            const double dx = 2*D(2)*x*y + D(3)*(r2 + 2*x*x);
            const double dy = D(2)*(r2 + 2*y*y) + 2*D(3)*x*y;

            const double x_distorted = x * radial + dx;
            const double y_distorted = y * radial + dy;

            // Project to image coordinates
            projected[2*count+0] = K(0,0) * x_distorted + K(0,2);
            projected[2*count+1] = K(1,1) * y_distorted + K(1,2);
            count++;
        }
        return projected;
    }




        optimization::Output calc_residuals(std::vector<double> &p, const std::vector<double> &dots_cam0,
                                            const std::vector<double> &dots_cam1, const std::vector<double> &grid, 
                                            const size_t num_img, const std::vector<int> &lengths,
                                            const bool print_flag){

        // ------------------------------------------------------
        // unpack parameters
        // ------------------------------------------------------


        // Camera matrices
        Eigen::Matrix3d K0, K1;
        K0 << p[0], 0,  p[2], 0,  p[1],  p[3], 0, 0, 1;
        K1 << p[9], 0, p[11], 0, p[10], p[12], 0, 0, 1;

        // cam 0 distortion parameters
        Eigen::VectorXd D0(5);
        for (int i = 0; i < 5; i++) D0(i) = p[4 + i];

        // cam1 distortion parameters
        Eigen::VectorXd D1(5);
        for (int i = 0; i < 5; i++) D1(i) = p[13 + i];


        // Stereo translation and rotation
        Eigen::Vector3d rvec_stereo(p[18], p[19], p[20]);
        Eigen::Vector3d tvec_stereo(p[21], p[22], p[23]);
        Eigen::Matrix3d R_stereo = rodrigues_to_matrix(rvec_stereo);

        //cam0 projections start at element 24
        int start_cam0 = 24;

        // init residuals
        Eigen::VectorXd residuals(2*dots_cam0.size());
        std::vector<double> proj0(2*dots_cam0.size());
        std::vector<double> proj1(2*dots_cam0.size());

        std::vector<int> img_offsets(num_img);
        int count = 0;
        for (size_t i = 0; i < num_img; i++) {
            img_offsets[i] = count;
            count += lengths[i];
        }


        // loop over all imgs in stereo_calibration
        #pragma omp parallel for
        for (int i = 0; i < num_img; i++){

            // rotation vector
            Eigen::Vector3d rvec0(p[start_cam0 + i*6],
                                  p[start_cam0 + i*6 + 1],
                                  p[start_cam0 + i*6 + 2]);

            // translation vector
            Eigen::Vector3d tvec0(p[start_cam0 + i*6 + 3],
                                  p[start_cam0 + i*6 + 4],
                                  p[start_cam0 + i*6 + 5]);

            // rotation matrix
            Eigen::Matrix3d R0 = rodrigues_to_matrix(rvec0);

            // Cam1 pose (derived from cam0 + stereo)
            Eigen::Matrix3d R1 = R_stereo * R0;
            Eigen::Vector3d T1 = R_stereo * tvec0 + tvec_stereo;

            // Convert R1 back to rotation vector for projection
            // Simple conversion (can be improved for numerical stability)
            Eigen::Vector3d rvec1;
            double trace = R1.trace();
            double angle = acos((trace - 1) / 2);
            if (angle < 1e-8) {
                rvec1.setZero();
            }
            else {
                Eigen::Vector3d axis;
                axis(0) = R1(2,1) - R1(1,2);
                axis(1) = R1(0,2) - R1(2,0);
                axis(2) = R1(1,0) - R1(0,1);
                axis.normalize();
                rvec1 = angle * axis;
            }

            //convert grid for this image to a vector of eigen 3d points
            std::vector<Eigen::Vector3d> grid_img_i(lengths[i]);
            int idx_start_3d = 0;
            int idx_start_2d = 0;

            //get start index of the grid for this image
            for (int j = 0; j < i; j++){
                idx_start_3d+=3*lengths[j];
                idx_start_2d+=2*lengths[j];
            }

            for (int j = 0; j < lengths[i]; j++){
                grid_img_i[j](0) = grid[idx_start_3d+j*3+0];
                grid_img_i[j](1) = grid[idx_start_3d+j*3+1];
                grid_img_i[j](2) = grid[idx_start_3d+j*3+2];
            }

            // Projection
            std::vector<double> proj0_i = project_points(grid_img_i, rvec0, tvec0, K0, D0);
            std::vector<double> proj1_i = project_points(grid_img_i, rvec1, T1, K1, D1);

            // offset in results arrays for local copy
            int local_offset = img_offsets[i];

            // residuals
            for (size_t j = 0; j < lengths[i]; j++) {

                //global index
                int global_idx = local_offset+j;

                if (print_flag) {
                    std::cout << std::setprecision(10) << dots_cam0[idx_start_2d+2*j+0] << " " << dots_cam0[idx_start_2d+2*j+1] << " ";
                    std::cout << std::setprecision(10) << proj0_i[2*j+0] << " " << proj0_i[2*j+1] << std::endl;
                }

                // residuals
                residuals[4*global_idx+0] = proj0_i[2*j+0] - dots_cam0[idx_start_2d+2*j+0];
                residuals[4*global_idx+1] = proj0_i[2*j+1] - dots_cam0[idx_start_2d+2*j+1];
                residuals[4*global_idx+2] = proj1_i[2*j+0] - dots_cam1[idx_start_2d+2*j+0];
                residuals[4*global_idx+3] = proj1_i[2*j+1] - dots_cam1[idx_start_2d+2*j+1];

                // populate master reprojection array for every image
                proj0[2*global_idx+0] = proj0_i[2*j+0];
                proj0[2*global_idx+1] = proj0_i[2*j+1];
                proj1[2*global_idx+0] = proj1_i[2*j+0];
                proj1[2*global_idx+1] = proj1_i[2*j+1];

                // increment count
                count++;
            }
            if (print_flag) std::cout << std::endl;
        }
        return {residuals, proj0, proj1};
    }


    Eigen::MatrixXd calc_jac(std::vector<double> &p, const Eigen::VectorXd &r, const std::vector<double> &dots_cam0, const std::vector<double> &dots_cam1,
                            const std::vector<double> &grid, const size_t num_img, const std::vector<int> &lengths){

        const int m = r.size();
        const int n = p.size();
        const double h = 1e-8;

        Eigen::MatrixXd jac(m,n);


        // perturb one parameter at a time
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            std::vector<double> p_prime = p;
            p_prime[j] += h;

            auto [r_prime, proj0, proj1] = calc_residuals(p_prime, dots_cam0, dots_cam1, grid, num_img, lengths, false);

            for (int i = 0; i < m; i++) {
                jac(i, j) = (r_prime[i] - r[i]) / h;
            }
        }
        return jac;


    }

}

