// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSMOOTH_H
#define DICSMOOTH_H

// STD library Header files
#include <vector>
#include <Eigen/Dense>

// Program Header files




namespace smooth {

    Eigen::VectorXd q4(std::vector<int> &x, std::vector<int> &y, std::vector<double>& disp_vals);
    Eigen::VectorXd q9(std::vector<int> &x, std::vector<int> &y, std::vector<double>& disp_vals);
    void gaussian_2d(std::vector<double>& data, const std::vector<int>& mask, int width, int height, double sigma);

}
#endif // DICSMOOTH_H
