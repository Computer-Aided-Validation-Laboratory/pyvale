// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSTRAIN_H
#define DICSTRAIN_H

// STD library Header files
#include <vector>
#include <Eigen/Dense>

// Program Header files




namespace strain {


    struct Window {
        std::vector<int> x;
        std::vector<int> y;
        std::vector<double> u;
        std::vector<double> v;
    };
    

    void engine(int *ss_x, int *ss_y, double *u, double *v, int num_ss_x, int num_ss_y, int sw_size, int q, std::string& formulation);
    bool fill_strain_window(int *ss_x,int *ss_y,
                            double *u, double *v, 
                            Window &window,
                            int num_ss_x, int num_ss_y,
                            int x0_idx, int y0_idx, int swr);
        
    // strain formulations
    inline Eigen::Matrix2d green(Eigen::Matrix2d F, Eigen::Matrix2d I);
    inline Eigen::Matrix2d hencky(Eigen::Matrix2d F, Eigen::Matrix2d I);
    inline Eigen::Matrix2d almansi(Eigen::Matrix2d F, Eigen::Matrix2d I);
    inline Eigen::Matrix2d biot_euler(Eigen::Matrix2d F, Eigen::Matrix2d I);
    inline Eigen::Matrix2d biot_lagrange(Eigen::Matrix2d F, Eigen::Matrix2d I);

}


#endif // DICSTRAIN_H
