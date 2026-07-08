// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <cstring>
#include <omp.h>
#include <ostream>
#include <iomanip>
#include <vector>
#include <numeric>

// common_cpp Header files
#include "../../common_cpp/util.hpp"

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
        opt.lambda = 0.01;
        const double eps = 1e-10;

        while (iter < opt.max_iter) {

            // calculate updated parameters
            iterate_cost(opt, dots_cam0, dots_cam1, grid, num_img, lengths, iter);

            // relative change of all parameters
            const double dp_norm = std::sqrt(std::inner_product(opt.dp.begin(), opt.dp.end(), opt.dp.begin(), 0.0));
            const double p_norm  = std::sqrt(std::inner_product( opt.p.begin(), opt.p.end(),  opt.p.begin(), 0.0));
            xtol = dp_norm / (p_norm+eps);

            ftol = std::abs(opt.costpdp - opt.costp);

            //std::cout << iter << " " << opt.costp << " " << opt.costpdp << " " << ftol << std::endl;

            if ((xtol < opt.precision) && (ftol < opt.precision)) {
                converged=true;
                break;
            }
            iter++;
        }

        //return the residuals from the final iteration 
        optimization::Output final_results = calc_residuals(opt.p, dots_cam0, dots_cam1, grid, num_img, lengths, iter, false);
        final_results.iter = iter;

        return final_results;
    }


    void iterate_cost(Parameters &opt,
                      const std::vector<double> &dots_cam0,
                      const std::vector<double> &dots_cam1, 
                      const std::vector<double> &grid, 
                      const size_t num_img, 
                      const std::vector<int> &lengths, 
                      const int iter){


        // Compute residuals at current point. p get updated in this
        optimization::Output res = calc_residuals(opt.p, dots_cam0, dots_cam1, grid, num_img, lengths, iter, false);

        // calculate jacobian
        Eigen::MatrixXd J = calc_jac(opt.p, res.residuals, dots_cam0, dots_cam1, grid, num_img, iter, lengths);


        // calc gradient
        Eigen::VectorXd g = J.transpose() * res.residuals;

        // Hessian
        Eigen::MatrixXd H = J.transpose() * J;

        // Remove fixed parameters from the linear system while retaining the
        // coupled solution for every parameter that is allowed to vary.
        for (int i = 0; i < opt.num_params; ++i) {
            if (!opt.vary[i]) {
                H.row(i).setZero();
                H.col(i).setZero();
                H(i, i) = 1.0;
                g(i) = 0.0;
            }
        }

        // (H + lambda*diag(H))
        Eigen::VectorXd diagH = H.diagonal();
        Eigen::MatrixXd D = diagH.asDiagonal();
        Eigen::MatrixXd A = H + opt.lambda*D;
        //Eigen::MatrixXd A = H + opt.lambda * Eigen::MatrixXd::Identity(H.rows(), H.cols());

        // get change in parameters
        Eigen::VectorXd dp = A.ldlt().solve(-g);

        // Updated parameters
        for (int i = 0; i < opt.p.size(); i++) {
            opt.dp[i] = opt.vary[i] ? dp(i) : 0.0;
            opt.pdp[i] = opt.p[i] + opt.dp[i];
        }


        // Evaluate new cost
        optimization::Output res_new = calc_residuals(opt.pdp, dots_cam0, dots_cam1, grid, num_img, lengths, iter, false);
        opt.costp = 0.0;
        opt.costpdp = 0.0;
        for (int i = 0; i < res_new.residuals.size(); i++){
            opt.costp   += res.residuals(i) * res.residuals(i);
            opt.costpdp += res_new.residuals(i) * res_new.residuals(i);
        }


 
        double cost_cam0_p = 0.0, cost_cam1_p = 0.0;
        double cost_cam0_pdp = 0.0, cost_cam1_pdp = 0.0;
        for (int k = 0; k < res_new.residuals.size() / 4; ++k) {
            double r0x = res.residuals(4*k+0), r0y = res.residuals(4*k+1);
            double r1x = res.residuals(4*k+2), r1y = res.residuals(4*k+3);
            cost_cam0_p += r0x*r0x + r0y*r0y;
            cost_cam1_p += r1x*r1x + r1y*r1y;
            r0x = res_new.residuals(4*k+0);
            r0y = res_new.residuals(4*k+1);
            r1x = res_new.residuals(4*k+2);
            r1y = res_new.residuals(4*k+3);
            cost_cam0_pdp += r0x*r0x + r0y*r0y;
            cost_cam1_pdp += r1x*r1x + r1y*r1y;
        }
        
        double actual = opt.costp - opt.costpdp;
        double predicted = -g.dot(dp) - 0.5 * dp.dot(H * dp);
        double rho = actual / predicted;
        const bool step_accepted = rho > 0.0;

        
        // std::cout << std::scientific << std::setprecision(4
        //   << "[iter " << std::setw(3) << iter << "] "
        //   << "cost=" << opt.costp
        //   << " -> " << opt.costpdp
        //   << "  dcost=" << actual
        //   << "  rho=" << rho
        //   << "  lambda=" << opt.lambda
        //   << "  |dp|=" << dp.norm()
        //   << "  cam0=" << cost_cam0_p << "->" << cost_cam0_pdp
        //   << "  cam1=" << cost_cam1_p << "->" << cost_cam1_pdp
        //   << "  step=" << (step_accepted ? "accepted" : "rejected")
        //   << '\n';

        common_util::info_out(
            "Iter= " + std::to_string(iter),
            "cost=" + std::to_string(opt.costp) + "->" + std::to_string(opt.costpdp) +
            " cam0=" + std::to_string(cost_cam0_pdp) +
            " cam1=" + std::to_string(cost_cam1_pdp));

        if (rho > 0) {
            opt.p = opt.pdp;
            opt.lambda *= std::max(1.0/3.0, 1.0 - pow(2*rho - 1, 3));
        } else {
            opt.lambda *= 2.0;
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

    Eigen::Vector3d matrix_to_rodrigues(const Eigen::Matrix3d &R) {
        Eigen::Vector3d rvec;

        double trace = R.trace();
        double cos_theta = (trace - 1.0) * 0.5;

        // Clamp for numerical stability
        cos_theta = std::min(1.0, std::max(-1.0, cos_theta));

        double theta = std::acos(cos_theta);

        // v small angle
        if (theta < 1e-10)
        {
            return Eigen::Vector3d::Zero();
        }

        // small angle
        if (M_PI - theta < 1e-6)
        {
            Eigen::Vector3d axis;

            axis(0) = std::sqrt(std::max(0.0, (R(0,0) + 1.0) * 0.5));
            axis(1) = std::sqrt(std::max(0.0, (R(1,1) + 1.0) * 0.5));
            axis(2) = std::sqrt(std::max(0.0, (R(2,2) + 1.0) * 0.5));

            // Fix signs using off-diagonal elements
            if (R(0,1) < 0) axis(1) = -axis(1);
            if (R(0,2) < 0) axis(2) = -axis(2);

            return theta * axis.normalized();
        }

        // General
        Eigen::Vector3d axis;
        axis << R(2,1) - R(1,2),
                R(0,2) - R(2,0),
                R(1,0) - R(0,1);

        axis /= (2.0 * std::sin(theta));

        rvec = theta * axis;
        return rvec;
    }



    // Project 3D points to 2D with distortion (assuming radial + tangential distortion)
    std::vector<double> project_points(const std::vector<Eigen::Vector3d> &gridpoints_3d,
                                                const Eigen::Matrix3d &R,
                                                const Eigen::Vector3d &tvec,
                                                const Eigen::Matrix3d &K,
                                                const Eigen::VectorXd &D) {

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
            projected[2*count+0] = K(0,0) * x_distorted + K(0,1) * y_distorted + K(0,2);
            projected[2*count+1] = K(1,0) * x_distorted + K(1,1) * y_distorted + K(1,2);
            count++;
        }
        return projected;
    }




        optimization::Output calc_residuals(std::vector<double> &p, const std::vector<double> &dots_cam0,
                                            const std::vector<double> &dots_cam1, const std::vector<double> &grid, 
                                            const size_t num_img, const std::vector<int> &lengths, const int iter,
                                            const bool print_flag){

        // ------------------------------------------------------
        // unpack parameters
        // ------------------------------------------------------


        // Camera matrices
        Eigen::Matrix3d K0, K1;
        K0 << p[0],  p[2],  p[3],
                 0,   p[1],  p[4],
                 0,      0,     1.0;

        K1 << p[10], p[12], p[13],
                 0, p[11], p[14],
                 0,     0,     1.0;

        // cam 0 distortion parameters
        Eigen::VectorXd D0(5);
        for (int i = 0; i < 5; i++) D0(i) = p[5 + i];

        // cam1 distortion parameters
        Eigen::VectorXd D1(5);
        for (int i = 0; i < 5; i++) D1(i) = p[15 + i];


        // Stereo translation and rotation
        Eigen::Vector3d rvec_stereo(p[20], p[21], p[22]);
        Eigen::Vector3d tvec_stereo(p[23], p[24], p[25]);
        Eigen::Matrix3d R_stereo = rodrigues_to_matrix(rvec_stereo);

        //cam0 projections start at element 26
        int start_cam0 = 26;

        // init residuals
        Eigen::VectorXd residuals(2*dots_cam0.size());
        std::vector<double> proj0(dots_cam0.size());
        std::vector<double> proj1(dots_cam1.size());

        std::vector<int> point_offsets(num_img);
        std::vector<int> offset_2d(num_img);
        std::vector<int> offset_3d(num_img);

        int cumulative_points = 0;
        int cumulative_2d = 0;
        int cumulative_3d = 0;

        for (size_t i = 0; i < num_img; i++) {
            point_offsets[i] = cumulative_points;
            offset_2d[i]     = cumulative_2d;
            offset_3d[i]     = cumulative_3d;

            cumulative_points += lengths[i];
            cumulative_2d     += 2 * lengths[i];
            cumulative_3d     += 3 * lengths[i];
        }


        // loop over all imgs in stereo_calibration
        #pragma omp parallel for
        for (int i = 0; i < num_img; i++){

            // rotation vector
            Eigen::Vector3d rvec0(p[start_cam0 + i*6 + 0],
                                  p[start_cam0 + i*6 + 1],
                                  p[start_cam0 + i*6 + 2]);

            // translation vector
            Eigen::Vector3d tvec0(p[start_cam0 + i*6 + 3],
                                  p[start_cam0 + i*6 + 4],
                                  p[start_cam0 + i*6 + 5]);

            // rotation matrix
            Eigen::Matrix3d R0 = rodrigues_to_matrix(rvec0);

            // Cam1 pose. From cam0 + stereo)
            Eigen::Matrix3d R1 = R_stereo*R0;
            Eigen::Vector3d T1 = R_stereo*tvec0 + tvec_stereo;


            //convert grid for this image to a vector of eigen 3d points
            std::vector<Eigen::Vector3d> grid_img_i(lengths[i]);
            int idx_start_3d = offset_3d[i];
            int idx_start_2d = offset_2d[i];
            int local_offset = point_offsets[i];

            for (int j = 0; j < lengths[i]; j++){
                grid_img_i[j](0) = grid[idx_start_3d+j*3+0];
                grid_img_i[j](1) = grid[idx_start_3d+j*3+1];
                grid_img_i[j](2) = grid[idx_start_3d+j*3+2];
            }

            // Projection
            std::vector<double> proj0_i = project_points(grid_img_i, R0, tvec0, K0, D0);
            std::vector<double> proj1_i = project_points(grid_img_i, R1, T1, K1, D1);

            // residuals
            for (size_t pt = 0; pt < lengths[i]; pt++) {

                //global index
                int global_idx = local_offset+pt;

                // residuals
                residuals[4*global_idx+0] = proj0_i[2*pt+0] - dots_cam0[idx_start_2d+2*pt+0];
                residuals[4*global_idx+1] = proj0_i[2*pt+1] - dots_cam0[idx_start_2d+2*pt+1];
                residuals[4*global_idx+2] = proj1_i[2*pt+0] - dots_cam1[idx_start_2d+2*pt+0];
                residuals[4*global_idx+3] = proj1_i[2*pt+1] - dots_cam1[idx_start_2d+2*pt+1];

                // populate master reprojection array for every image
                proj0[2*global_idx+0] = proj0_i[2*pt+0]; // cam0 x 
                proj0[2*global_idx+1] = proj0_i[2*pt+1]; // cam0 y
                proj1[2*global_idx+0] = proj1_i[2*pt+0]; // cam1 x
                proj1[2*global_idx+1] = proj1_i[2*pt+1]; // cam1 y

                if (print_flag) {
                    std::cout << std::setprecision(10) << dots_cam0[idx_start_2d+2*pt+0] << " " << dots_cam0[idx_start_2d+2*pt+1] << " ";
                    std::cout << std::setprecision(10) << proj0_i[2*pt+0] << " " << proj0_i[2*pt+1] << " ";
                    std::cout << std::setprecision(10) << dots_cam1[idx_start_2d+2*pt+0] << " " << dots_cam1[idx_start_2d+2*pt+1] << " ";
                    std::cout << std::setprecision(10) << proj1_i[2*pt+0] << " " << proj1_i[2*pt+1] << std::endl;
                }
            }
            if (print_flag) std::cout << std::endl;
        }
        return {residuals, proj0, proj1, iter};
    }


    Eigen::MatrixXd calc_jac(std::vector<double> &p, const Eigen::VectorXd &r, const std::vector<double> &dots_cam0, const std::vector<double> &dots_cam1,
                            const std::vector<double> &grid, const size_t num_img, const int iter, const std::vector<int> &lengths){

        const int m = r.size();
        const int n = p.size();
        const double eps_fd = 1e-6;

        Eigen::MatrixXd jac(m,n);


        // perturb one parameter at a time
        for (int j = 0; j < n; j++) {
            std::vector<double> p_prime = p;
            double h_j = eps_fd; // * std::max(1.0, std::abs(p[j]));
            p_prime[j] += h_j;

            auto [r_prime, proj0, proj1, _] = calc_residuals(p_prime, dots_cam0, dots_cam1, grid, num_img, lengths, iter, false);

            for (int i = 0; i < m; i++) {
                jac(i, j) = (r_prime[i] - r[i]) / h_j;
            }
        }
        return jac;


    }

}

