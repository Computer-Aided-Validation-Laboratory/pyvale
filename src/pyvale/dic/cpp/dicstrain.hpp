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

    /**
     * @brief Struct for the strain window. Contains subset coordinates (x,y) 
     * and the corresponding displacement vectors (u,v).
     * 
     */
    struct Window {
        std::vector<int> x;
        std::vector<int> y;
        std::vector<double> u;
        std::vector<double> v;
    };
    

    void engine(int *ss_x, int *ss_y, double *u, double *v, int num_ss_x, int num_ss_y, int sw_size, int q, std::string& formulation);

    /**
     * @brief Fills the strain window with the subset coordinates 
     * and displacement vectors based on the input parameters
     * 
     * @param ss_x subset x-coordinates
     * @param ss_y subset y-coordinates
     * @param u horizontal displacement
     * @param v vertical displacement
     * @param window strain window struct. 
     * @param num_ss_x number of subsets along the x-axis
     * @param num_ss_y number of subsets along the y-axis
     * @param x0_idx index of the x-coordinate in the subset mask
     * @param y0_idx index of the x-coordinate in the subset mask
     * @param swr strain window radius (strain window / 2)
     * @return true if the strain window is filled successfully
     * @return false if the strain window is out of bounds
     */
    bool fill_strain_window(int *ss_x,int *ss_y,
                            double *u, double *v, 
                            Window &window,
                            int num_ss_x, int num_ss_y,
                            int x0_idx, int y0_idx, int swr);
        
    // strain formulations
    /**
     * @brief Calculates Green strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Green Strain
     */
    inline Eigen::Matrix2d green(Eigen::Matrix2d F, Eigen::Matrix2d I);

    /**
     * @brief Calculates Hencky strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Hencky Strain
     */
    inline Eigen::Matrix2d hencky(Eigen::Matrix2d F, Eigen::Matrix2d I);

    /**
     * @brief Calculates Almansi strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d almansi(Eigen::Matrix2d F, Eigen::Matrix2d I);

    /**
     * @brief Calculates Biot strain in the euler coordiate system for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d biot_euler(Eigen::Matrix2d F, Eigen::Matrix2d I);

    /**
     * @brief Calculates Biot strain in the lagrange coordiate system for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d biot_lagrange(Eigen::Matrix2d F, Eigen::Matrix2d I);

}


#endif // DICSTRAIN_H
