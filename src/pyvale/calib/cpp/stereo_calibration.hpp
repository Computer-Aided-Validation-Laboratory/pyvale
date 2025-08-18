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

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// eigen header filesfiles
#include "./Eigen/Dense"
#include "./Eigen/Geometry"

namespace py = pybind11;

void stereo_calibration(const py::array_t<double> &grid,
                        const py::array_t<double> &dots_cam0,
                        const py::array_t<double> &dots_cam1,
                        const std::vector<int> *lengths,
                        const int px_hori, const int px_vert, const int num_img_pairs);

//
// void optimizsation(){
//
//     int max_iter = 100;
//     int iter = 0;
//     double ftol = 0;
//     double xtol = 0;
//     double lambda = 0.001;
//     uint8_t converged = false;
//     double cost;
//     double costpdp;
//     double precision = 1e-5;
//
//     while (iter < max_iter) {
//
//         // perform the optimization
//         optimize_cost();
//         update_lambda();
//
//         // variation on correlation coefficient
//         ftol = std::abs(costpdp - costp);
//
//         if (ftol < precision) {
//             converged=true; 
//             break;
//         }
//         iter++;
//     }
//
// }
//
//
//
// void optimize_cost(std::vector<double> &p, const size_t num_img){
//
//
//     // Compute residuals at current point. p get updated in this
//     Eigen::VectorXd residuals = calc_residuals(p, grid, dots_cam0, dots_cam1, num_img);
//
//     // calculate jacobian
//     Eigen::MatrixXd J = calc_jac(p, grid, dots_cam0, dots_cam1, num_img);
//
//     // calc gradient
//     Eigen::VectorXd gradient = J.transpose() * r;
//
//     // Hessian
//     Eigen::MatrixXd H = J.transpose() * J;
//
//     // (H + lambda*Identity)
//     Eigen::MatrixXd A = H + lambda * MatrixXd::Identity(H.rows(), H.cols());
//
//     // get change in parameters
//     Eigen::VectorXd dp = A.ldlt().solve(-gradient);
//
//     // Updated parameters
//     std::vector<double> new_p(p.size());
//     for (int i = 0; i < p.size(); i++) {
//         pdp[i] = p[i] + dp(i);
//     }
//
//     // Evaluate new cost
//     std::vector<double> new_residuals = calc_residuals(pdp, grid, dots_cam0, dots_cam1, num_img);
//     double new_cost = 0.0;
//     for (double r : new_residuals) {
//         costpdp += r * r;
//     }
//     costpdp *= 0.5;
//
//     // Accept or reject step
//     if (new_cost < cost) {
//         p = new_p;
//         lambda /= 10.0;
//         cost = cost;
//     } 
//     else {
//         lambda *= 10.0;
//     }
// }
//
//
// // Function to convert Rodrigues rotation vector to rotation matrix using Eigen
// Eigen::Matrix3d rodrigues_to_matrix(const Eigen::Vector3d& rvec) {
//
//     double theta = rvec.norm();
//
//     // handle tiny angles
//     if (theta < 1e-10) {
//         return Eigen::Matrix3d::Identity();
//     }
//
//     // Normalise
//     Eigen::Vector3d axis = rvec / theta;
//
//     // rotation
//     Eigen::AngleAxisd rotation(theta, axis);
//     return rotation.toRotationMatrix();
// }
//
//
//
// // Project 3D points to 2D with distortion (assuming radial + tangential distortion)
// std::vector<Vector2d> projectPoints(const std::vector<Vector3d>& points3d,
//                                    const Vector3d& rvec,
//                                    const Vector3d& tvec,
//                                    const Matrix3d& K,
//                                    const VectorXd& D) {
//     Matrix3d R = rodrigues(rvec);
//     std::vector<Vector2d> projected;
//     projected.reserve(points3d.size());
//
//     for (const auto& point : points3d) {
//         // Transform to camera coordinates
//         Vector3d p_cam = R * point + tvec;
//
//         // Normalize
//         double x = p_cam(0) / p_cam(2);
//         double y = p_cam(1) / p_cam(2);
//
//         // Apply distortion (assuming 5 distortion coefficients: k1, k2, p1, p2, k3)
//         double r2 = x*x + y*y;
//         double r4 = r2*r2;
//         double r6 = r4*r2;
//
//         // Radial distortion
//         double radial = 1 + D(0)*r2 + D(1)*r4 + D(4)*r6;
//
//         // Tangential distortion
//         double dx = 2*D(2)*x*y + D(3)*(r2 + 2*x*x);
//         double dy = D(2)*(r2 + 2*y*y) + 2*D(3)*x*y;
//
//         double x_distorted = x * radial + dx;
//         double y_distorted = y * radial + dy;
//
//         // Project to image coordinates
//         Vector2d p_img;
//         p_img(0) = K(0,0) * x_distorted + K(0,2);
//         p_img(1) = K(1,1) * y_distorted + K(1,2);
//
//         projected.push_back(p_img);
//     }
//
//     return projected;
// }
//
//
//
//
// void calc_residuals(std::vector<double> &p, const size_t num_img){
//
//     Eigen::Matrix3d R = rodrigues_to_matrix(rvec);
//     Eigen::Matrix3d cam0;
//     Eigen::Matrix3d cam1;
//
//     // ------------------------------------------------------
//     // unpack parameters
//     // ------------------------------------------------------
//
//     // Camera matrices
//     Eigen::Matrix3d K0, K1;
//     K0 << p[0], 0,  p[2], 0,  p[1],  p[3], 0, 0, 1;
//     K1 << p[9], 0, p[11], 0, p[10], p[12], 0, 0, 1;
//
//     // cam 0 distortion parameters
//     Eigen::VectorXd D0(5);
//     for (int i = 0; i < 5; i++) D0(i) = p[4 + i];
//
//     // cam1 distortion parameters
//     Eigen::VectorXd D1(5);
//     for (int i = 0; i < 5; i++) D1(i) = p[13 + i];
//
//
//     // Stereo translation and rotation
//     Eigen::Vector3d rvec_stereo(p[18], p[19], p[20]);
//     Eigen::Vector3d tvec_stereo(p[21], p[22], p[23]);
//     Eigen::Matrix3d R_stereo = rodrigues(rvec_stereo);
//
//     //cam0 projections start at element 24
//     int start_cam0 = 24;
//     std::vector<double> residuals(4*p.size(), 0.0);
//
//     // loop over all imgs in stereo_calibration
//     for (size_t i = 0; i < num_img; i++){
//
//         Eigen::Vector3d rvec0(p[start_cam0 + i*6],
//                        p[start_cam0 + i*6 + 1],
//                        p[start_cam0 + i*6 + 2]);
//         Eigen::Vector3d tvec0(p[start_cam0 + i*6 + 3],
//                        p[start_cam0 + i*6 + 4],
//                        p[start_cam0 + i*6 + 5]);
//
//         Eigen::Matrix3d R0 = rodrigues(rvec0);
//
//         // Cam1 pose (derived from cam0 + stereo)
//         Eigen::Matrix3d R1 = R_stereo * R0;
//         Eigen::Vector3d T1 = R_stereo * tvec0 + tvec_stereo;
//
//         // Convert R1 back to rotation vector for projection
//         // Simple conversion (can be improved for numerical stability)
//         Eigen::Vector3d rvec1;
//         double trace = R1.trace();
//         double angle = acos((trace - 1) / 2);
//         if (angle < 1e-8) {
//             rvec1.setZero();
//         } else {
//             Eigen::Vector3d axis;
//             axis(0) = R1(2,1) - R1(1,2);
//             axis(1) = R1(0,2) - R1(2,0);
//             axis(2) = R1(1,0) - R1(0,1);
//             axis.normalize();
//             rvec1 = angle * axis;
//         }
//
//
//         // Projection
//         std::vector<Eigen::Vector2d> proj0 = projectPoints(grid[i], rvec0, tvec0, K0, D0);
//         std::vector<Eigen::Vector2d> proj1 = projectPoints(grid[i], rvec1, T1, K1, D1);
//
//         // residuals
//         for (size_t j = 0; j < proj0.size(); j++) {
//             Vector2d res0 = proj0[j] - dots_cam0[i][j];
//             Vector2d res1 = proj1[j] - dots_cam1[i][j];
//
//             residuals[4*j+0] = res0(0);
//             residuals[4*j+1] = res0(1);
//             residuals[4*j+2] = res1(0);
//             residuals[4*j+3] = res1(1);
//         }
//     }
//
//     return residuals;
// }
//
//
// Eigen::matrixXd calc_jac(const std::vector<double>& p,
//                         const std::vector<std::vector<Vector3d>>& grid,
//                         const std::vector<std::vector<Vector2d>>& dots_cam0,
//                         const std::vector<std::vector<Vector2d>>& dots_cam1,
//                         const size_t num_img){
//
//     const int n = p.size();
//     const int m = n*4;
//
//     Eigen::MatrixXd jac(m,n);
//
//     for (int j = 0; j < n; j++) {
//         std::vector<double> p_new = bundle_adjustment_error(p, grid, dots_cam0, dots_cam1, num_img);
//
//         for (int i = 0; i < m; i++) {
//             jacobian(i, j) = (p_new[i] - p[i]) / h;
//         }
//     }
//
//     // Gauss-Newton approximation: H ≈ J^T * J
//     return jacobian;
//
//
// }
//
//
//
//
// PYBIND11_MODULE(dic2dcpp, m) {
//
//
//     m.def("stereo_calibration", &stereo_calibration, "stereo_calibration");
// }
//

