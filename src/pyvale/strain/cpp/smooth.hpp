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

    /**
     * @brief 
     * 
     * @param[in] x displacement x-coordinates within strain window
     * @param[in] y displacement y-coordinates within strain window
     * @param[in] disp_vals 
     * @return Eigen::VectorXd A vector of coefficients for a bilinear fit inside strain window
     */
    Eigen::VectorXd q4(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double>& disp_vals);

    /**
     * @brief 
     * 
     * @param[in] x displacement x-coordinates within strain window
     * @param[in] y displacement y-coordinates within strain window
     * @param[in] disp_vals 
     * @return Eigen::VectorXd  A vector of coefficients for a bilinear fit inside strain window
     */
    Eigen::VectorXd q9(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double>& disp_vals);

    /**
     * @brief 
     * 
     * @param data 
     * @param mask 
     * @param width 
     * @param height 
     * @param sigma 
     */
    void gaussian_2d(std::vector<double>& data, const std::vector<int>& mask, int width, int height, double sigma);

}
#endif // DICSMOOTH_H
