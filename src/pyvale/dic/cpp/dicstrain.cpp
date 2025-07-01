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

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace strain {

    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

    void engine(const py::array_t<int> ss_x_arr,
                const py::array_t<int> ss_y_arr,
                const py::array_t<double> u_arr,
                const py::array_t<double> v_arr,
                int nss_x, int nss_y, int nimg,
                int sw_size, int q, std::string &form){


        // get raw pointers for numpy arrays
        int* ss_x = static_cast<int*>(ss_x_arr.request().ptr);
        int* ss_y = static_cast<int*>(ss_y_arr.request().ptr);
        double* u = static_cast<double*>(u_arr.request().ptr);
        double* v = static_cast<double*>(v_arr.request().ptr);


        strain::Window window;
        window.x.resize(sw_size*sw_size,0.0);
        window.y.resize(sw_size*sw_size,0.0);
        window.u.resize(sw_size*sw_size,0.0);
        window.v.resize(sw_size*sw_size,0.0);


        // loop over the displacement images
        for (int img = 0; img < nimg; img++) {

            std::cout << img << std::endl;

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

                    // calculate derivatives
                    if (q==4){

                        // get polynomial coeffieicnts from smoothing
                        uc = strainsmooth::q4(window.x ,window.y, window.u);
                        vc = strainsmooth::q4(window.x ,window.y, window.v);

                        // populate 2D deformation gradient matrix
                        deform_grad(0,0) = 1.0 + uc[1] + uc[3]*y0;
                        deform_grad(0,1) = uc[2] + uc[3]*x0;
                        deform_grad(1,0) = vc[1] + vc[3]*y0;
                        deform_grad(1,1) = 1.0 + vc[2] + vc[3]*x0;

                        // calculate strain
                        if (form=="GREEN"){
                            eps = green(deform_grad);
                        }
                        else if (form=="ALMANSI"){
                            eps = almansi(deform_grad);
                        }
                        else if (form=="HENCKY"){
                            eps = hencky(deform_grad);
                        }
                        else if (form=="BIOT_EULER"){
                            eps = biot_euler(deform_grad);
                        }
                        else if (form=="BIOT_LAGRANGE"){
                            eps = biot_lagrange(deform_grad);
                        }
                        else {
                            std::cerr << "Unknown Strain formulation: '" << form << "'." << std::endl;
                        }

                        //std::cout << x0 << " " << y0 << " " << deform_grad(0,0) << " " << deform_grad(0,1) << " " << deform_grad(1,0) << " " << deform_grad(1,1) << std::endl; 
                        std::cout << x0 << " " << y0 << " " << eps(0,0) << " " << eps(0,1) << " " << eps(1,0) << " " << eps(1,1) << std::endl; 

                    }
                     if (q==9){

                        // get polynomial coeffieicnts from smoothing
                        uc = strainsmooth::q9(window.x ,window.y, window.u);
                        vc = strainsmooth::q9(window.x ,window.y, window.v);

                        // populate 2D deformation gradient matrix
                        deform_grad(0,0) = 1.0 + uc[1] + uc[3]*y0 + 2.0*uc[4]*x0 + 2.0*uc[6]*x0*y0 + uc[7]*y0*y0 + 2.0*uc[8]*x0*y0*y0;
                        deform_grad(0,1) = uc[2] + uc[3]*x0 + 2.0*uc[5]*y0 + uc[6]*x0*x0 + 2.0*uc[7]*x0*y0 + 2.0*uc[8]*x0*x0*y0;
                        deform_grad(1,0) = vc[1] + vc[3]*y0 + 2.0*vc[4]*x0 + 2.0*vc[6]*x0*y0 + vc[7]*y0*y0 + 2.0*vc[8]*x0*y0*y0;
                        deform_grad(1,1) = 1.0 + vc[2] + vc[3]*x0 + 2.0*vc[5]*y0 + vc[6]*x0*x0 + 2.0*vc[7]*x0*y0 + 2.0*vc[8]*x0*x0*y0;

                        // calculate strain
                        if (form=="GREEN"){
                            eps = green(deform_grad);
                        }
                        else if (form=="ALMANSI"){
                            eps = almansi(deform_grad);
                        }
                        else if (form=="HENCKY"){
                            eps = hencky(deform_grad);
                        }
                        else if (form=="BIOT_EULER"){
                            eps = biot_euler(deform_grad);
                        }
                        else if (form=="BIOT_LAGRANGE"){
                            eps = biot_lagrange(deform_grad);
                        }

                        //std::cout << x0 << " " << y0 << " " << deform_grad(0,0) << " " << deform_grad(0,1) << " " << deform_grad(1,0) << " " << deform_grad(1,1) << std::endl; 
                        std::cout << x0 << " " << y0 << " " << eps(0,0) << " " << eps(0,1) << " " << eps(1,0) << " " << eps(1,1) << std::endl; 

                    }
                }
            }
            std::cout << std::endl;
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

        // check if the centre of strain window is within mask bounds
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
