// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Program Header files
#include "./dicsmooth.hpp"
#include "./defines.hpp"

namespace smooth {

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

} // namespace smooth
