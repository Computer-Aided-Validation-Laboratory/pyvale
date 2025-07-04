// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

// Program Header files
#include "./dicsmooth.hpp"
#include "./dicstrain.hpp"
#include "./defines.hpp"
#include "dicutil.hpp"

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace strain {

    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

    void engine(const py::array_t<int> &ss_x_arr,
                const py::array_t<int> &ss_y_arr,
                const py::array_t<double> &u_arr,
                const py::array_t<double> &v_arr,
                int nss_x, int nss_y, int nimg,
                int sw_size, int q, std::string &form,
                util::SaveConfig &strain_save_conf){


        // get raw pointers for numpy arrays
        int* ss_x = static_cast<int*>(ss_x_arr.request().ptr);
        int* ss_y = static_cast<int*>(ss_y_arr.request().ptr);
        double* u = static_cast<double*>(u_arr.request().ptr);
        double* v = static_cast<double*>(v_arr.request().ptr);

        // function wrapper
        std::function<Eigen::VectorXd(std::vector<int>&, std::vector<int>&, std::vector<double>&)> smooth_window = (q == 4) ? smooth::q4 : smooth::q9;


        strain::Window window(sw_size);
        if (strain_save_conf.at_end) strain::Results results(nimg*nss_x*nss_y);
        else strain::Results results(nss_x*nss_y);


        // loop over the displacement images
        for (int img = 0; img < nimg; img++) {

            // loop over strain windows within the image
            for (int sw = 0; sw < nss_x*nss_y; sw++){

                int x0 = ss_x[sw];
                int y0 = ss_y[sw];

                bool valid_window = fill_window(ss_x, ss_y, u, v, img,
                                                sw, window, nss_x,
                                                nss_y, sw_size);

                // element coefficients
                Eigen::VectorXd uc;
                Eigen::VectorXd vc;

                // 2D deformation gradient matrix and identity matrix
                Eigen::Matrix2d deform_grad = Eigen::Matrix2d::Zero();
                Eigen::Matrix2d eps = Eigen::Matrix2d::Zero();


                if (valid_window){
                    uc = smooth_window(window.x, window.y, window.u);
                    vc = smooth_window(window.x, window.y, window.v);
                    deform_grad = compute_deformation_gradient(q, uc, vc, x0, y0);
                    eps = compute_strain(form, deform_grad);
                }
            }
        }
    }



    bool fill_window(int *ss_x, int *ss_y, double *u, double *v,
                            int img, int sw, Window &window,
                            int nss_x, int nss_y, int sw_size){

        const int swr = sw_size / 2;
        const int x0_idx = sw % nss_x;
        const int y0_idx = sw / nss_x;
        const int xmin = x0_idx - swr;
        const int xmax = x0_idx + swr;
        const int ymin = y0_idx - swr;
        const int ymax = y0_idx + swr;

        // check centre of strain window is within mask bounds
        if ((xmin < 0) || (xmax >= nss_x) || (ymin < 0) || (ymax >= nss_y)) return false;
        
        int widx = 0;
        for (int j = ymin; j <= ymax; j++){
            for (int i = xmin; i <= xmax; i++){

                // index in 3d results array
                int idx_2d = nss_x*j + i;
                int idx_3d = nss_x*nss_y*img + idx_2d;

                // check if all subsets in the strain window are not nan
                if (std::isnan(u[idx_3d]) || std::isnan(v[idx_3d])) return false;

                // populate subset window
                window.x[widx] = ss_x[idx_2d];
                window.y[widx] = ss_y[idx_2d];
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];

                //std::cout << window.x[idx] << " " << window.y[idx] << " " << window.u[idx] << " " << window.v[idx] << std::endl;
                widx++;
            }
        }
        return true;
    }


    Eigen::Matrix2d compute_deformation_gradient(const int q, const Eigen::VectorXd &uc, const Eigen::VectorXd& vc, const double x0, const double y0) {

        Eigen::Matrix2d grad;

        if (q == 4) {
            grad(0,0) = 1.0 + uc[1] + uc[3]*y0;
            grad(0,1) = uc[2] + uc[3]*x0;
            grad(1,0) = vc[1] + vc[3]*y0;
            grad(1,1) = 1.0 + vc[2] + vc[3]*x0;
        }
        else if (q == 9) {
            grad(0,0) = 1.0 + uc[1] + uc[3]*y0 + 2.0*uc[4]*x0 + 2.0*uc[6]*x0*y0 + uc[7]*y0*y0 + 2.0*uc[8]*x0*y0*y0;
            grad(0,1) = uc[2] + uc[3]*x0 + 2.0*uc[5]*y0 + uc[6]*x0*x0 + 2.0*uc[7]*x0*y0 + 2.0*uc[8]*x0*x0*y0;
            grad(1,0) = vc[1] + vc[3]*y0 + 2.0*vc[4]*x0 + 2.0*vc[6]*x0*y0 + vc[7]*y0*y0 + 2.0*vc[8]*x0*y0*y0;
            grad(1,1) = 1.0 + vc[2] + vc[3]*x0 + 2.0*vc[5]*y0 + vc[6]*x0*x0 + 2.0*vc[7]*x0*y0 + 2.0*vc[8]*x0*x0*y0;
        }

        return grad;
    }

    Eigen::Matrix2d compute_strain(const std::string& form, const Eigen::Matrix2d& deform_grad) {
        if (form == "GREEN")        return green(deform_grad);
        else if (form == "ALMANSI") return almansi(deform_grad);
        else if (form == "HENCKY")  return hencky(deform_grad);
        else if (form == "BIOT_EULER") return biot_euler(deform_grad);
        else if (form == "BIOT_LAGRANGE") return biot_lagrange(deform_grad);

        std::cerr << "Unknown Strain formulation: '" << form << "'." << std::endl;
        return Eigen::Matrix2d::Zero();
    }


    inline Eigen::Matrix2d green(Eigen::Matrix2d F){
        return 0.5 * (F.transpose() * F - I);
    }





    inline Eigen::Matrix2d hencky(Eigen::Matrix2d F){
        Eigen::Matrix2d C = F.transpose() * F;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // Get eigenvectors and sqrt-eigenvalues
        const Eigen::Matrix2d Q = solver.eigenvectors();
        const Eigen::Vector2d eigvals = solver.eigenvalues();

        return Q * (0.5 * eigvals.array().log().matrix().asDiagonal()) * Q.transpose();
    }




    inline Eigen::Matrix2d almansi(Eigen::Matrix2d F){
        Eigen::Matrix2d B = F * F.transpose();
        Eigen::Matrix2d B_inv = B.inverse();
        return 0.5 * (I - B_inv); 
    }





    inline Eigen::Matrix2d biot_euler(Eigen::Matrix2d F){

        Eigen::Matrix2d C = F * F.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // U = sqrt(C) = Q * sqrt(D) * Q^T
        Eigen::Matrix2d D_sqrt = solver.eigenvalues().cwiseSqrt().asDiagonal();
        Eigen::Matrix2d U = solver.eigenvectors() * D_sqrt * solver.eigenvectors().transpose();

        return U - I;

    }




    inline Eigen::Matrix2d biot_lagrange(Eigen::Matrix2d F){

        Eigen::Matrix2d C = F.transpose() * F;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // U = sqrt(C) = Q * sqrt(D) * Q^T
        Eigen::Matrix2d D_sqrt = solver.eigenvalues().cwiseSqrt().asDiagonal();
        Eigen::Matrix2d U = solver.eigenvectors() * D_sqrt * solver.eigenvectors().transpose();

        return U - I;

    }

} // namespace strain
