// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <vector>

// common_cpp Header files
#include <Eigen/Dense>

// Strain Header files
#include "./smooth.hpp"


namespace smooth {

    // bilinear lagrange polynomials
    Eigen::VectorXd q4(std::vector<int> &x, std::vector<int> &y, 
                       std::vector<double>& disp_vals){

        Eigen::MatrixXd A(disp_vals.size(), 4);
        Eigen::VectorXd b(disp_vals.size());
        
        for (size_t i = 0; i < x.size(); ++i) {
            double ss_x = x[i];
            double ss_y = y[i];
            A(i, 0) = 1.0;
            A(i, 1) = ss_x;
            A(i, 2) = ss_y;
            A(i, 3) = ss_x * ss_y;
            b(i) = disp_vals[i];
        }

        // solve linear system A * coeffs = b
        Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        return coeffs;
    }

    // biquadratic lagrange polynomials
    Eigen::VectorXd q9(std::vector<int>& x, std::vector<int>& y,
                       std::vector<double>& disp_vals) {

        Eigen::MatrixXd A(disp_vals.size(), 9);
        Eigen::VectorXd b(disp_vals.size());

        for (size_t i = 0; i < x.size(); ++i) {
            double ss_x = x[i];
            double ss_y = y[i];
            A(i, 0) = 1.0;
            A(i, 1) = ss_x;
            A(i, 2) = ss_y;
            A(i, 3) = ss_x * ss_y;
            A(i, 4) = ss_x * ss_x;
            A(i, 5) = ss_y * ss_y;
            A(i, 6) = ss_x * ss_x * ss_y;
            A(i, 7) = ss_x * ss_y * ss_y;
            A(i, 8) = ss_x * ss_x * ss_y * ss_y;

            b(i) = disp_vals[i];
        }

        Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        return coeffs;
    }


    // 2D Gaussian kernel with a given radius and sigma
    std::vector<std::vector<double>> make_gaussian_kernel(int radius, double sigma) {
        std::vector<std::vector<double>> kernel(2 * radius + 1, std::vector<double>(2 * radius + 1));
        double sum = 0.0;

        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                double value = std::exp(-(x * x + y * y) / (2.0 * sigma * sigma));
                kernel[y + radius][x + radius] = value;
                sum += value;
            }
        }

        // Normalize the kernel
        for (int y = 0; y < 2 * radius + 1; ++y) {
            for (int x = 0; x < 2 * radius + 1; ++x) {
                kernel[y][x] /= sum;
            }
        }

        return kernel;
    }

    // Reflect index at boundaries (mirrored edges)
    int mirror_index(int idx, int max) {
        if (idx < 0) return -idx;
        if (idx >= max) return 2 * max - idx - 2;
        return idx;
    }

   void gaussian_2d(std::vector<double>& data,
                               const std::vector<int>& mask,
                               int width, int height, double sigma) {

        int radius = static_cast<int>(std::ceil(3.0 * sigma));
        std::vector<std::vector<double>> kernel = make_gaussian_kernel(radius, sigma);

        std::vector<double> result(data.size(), 0.0);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {

                int idx = y * width + x;
                if (mask[idx]==-1) continue;

                double weighted_sum = 0.0;
                double weight_total = 0.0;

                for (int ky = -radius; ky <= radius; ++ky) {
                    for (int kx = -radius; kx <= radius; ++kx) {
                        int nx = mirror_index(x + kx, width);
                        int ny = mirror_index(y + ky, height);
                        int nidx = ny * width + nx;

                        if (mask[nidx]!=-1) {
                            double weight = kernel[ky + radius][kx + radius];
                            weighted_sum += data[nidx] * weight;
                            weight_total += weight;
                        }
                    }
                }

                if (weight_total > 0.0) {
                    result[idx] = weighted_sum / weight_total;
                }
            }
        }

        data = result;
    }

}
