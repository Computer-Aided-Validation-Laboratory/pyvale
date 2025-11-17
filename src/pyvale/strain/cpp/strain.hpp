// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICSTRAIN_H
#define DICSTRAIN_H

// STD library Header files
#include <vector>
#include <cmath>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// common_cpp Header files
#include "../../common_cpp/util.hpp"
#include <Eigen/Dense>

// Program Header files


namespace py = pybind11;


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

        Window(int sw_size) 
            : x(sw_size * sw_size, 0.0),
              y(sw_size * sw_size, 0.0),
              u(sw_size * sw_size, 0.0),
              v(sw_size * sw_size, 0.0)
        {}
    };


    struct Results {
        std::vector<int> x;
        std::vector<int> y;
        std::vector<double> def_grad;
        std::vector<double> strain;
        std::vector<bool> valid_window;

        Results(int nwindows) 
            : x(nwindows, 0),
              y(nwindows, 0),
              def_grad(nwindows*4, std::nan("")),
              strain(nwindows*4, std::nan("")),
              valid_window(nwindows, false)
        {}
    };



    void engine(const py::array_t<int> &ss_x_arr,
                const py::array_t<int> &ss_y_arr,
                const py::array_t<double> &u_arr,
                const py::array_t<double> &v_arr,
                const int nss_x, const int nss_y, 
                const int nimg, const int sw_size, 
                const int q, const std::string &form,
                const std::vector<std::string> &filenames,
                const common_util::SaveConfig &strain_save_conf);

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
    bool fill_window(int *ss_x, int *ss_y, double *u, double *v,
                            int img, int sw, Window &window,
                            int nss_x, int nss_y, int sw_size);



    Eigen::Matrix2d compute_def_grad(const int q, const Eigen::VectorXd &uc, 
                                     const Eigen::VectorXd& vc, const double x0, 
                                     const double y0);


    Eigen::Matrix2d compute_strain(const std::string& form, 
                                   const Eigen::Matrix2d& deform_grad);


    /**
     */
    void append_results(int sw, strain::Results &results,
                        const bool save_at_end, const int x0, const int y0, 
                        const Eigen::Matrix2d &deform_grad, 
                        const Eigen::Matrix2d &eps, 
                        const int nwindows, const int img);


    /**
     */
    void save_to_disk(int img, const strain::Results &results, 
                      const common_util::SaveConfig &strain_save_conf, 
                      const int nwindows, const int nimg,
                      const std::vector<std::string> filenames);



    // strain formulations
    /**
     * @brief Calculates Green strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Green Strain
     */
    inline Eigen::Matrix2d green(Eigen::Matrix2d F);

    /**
     * @brief Calculates Hencky strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Hencky Strain
     */
    inline Eigen::Matrix2d hencky(Eigen::Matrix2d F);

    /**
     * @brief Calculates Almansi strain for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d almansi(Eigen::Matrix2d F);

    /**
     * @brief Calculates Biot strain in the euler coordiate system for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d biot_euler(Eigen::Matrix2d F);

    /**
     * @brief Calculates Biot strain in the lagrange coordiate system for a given deformation gradient F and identity matrix I.
     * 
     * @param F deformation gradient
     * @param I Identity Matrix
     * @return Eigen::Matrix2d Almansi Strain
     */
    inline Eigen::Matrix2d biot_lagrange(Eigen::Matrix2d F);

}


#endif // DICSTRAIN_H
