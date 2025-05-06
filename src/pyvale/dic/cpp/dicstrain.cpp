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

namespace strain {

    
    void engine(int *ss_x, int *ss_y, double *u, double *v, 
                int num_ss_x, int num_ss_y, int sw_size, int q, std::string &formulation){

        int sw_radius = sw_size / 2;

        strain::Window window;
        window.x.resize(sw_size*sw_size,0.0);
        window.y.resize(sw_size*sw_size,0.0);
        window.u.resize(sw_size*sw_size,0.0);
        window.v.resize(sw_size*sw_size,0.0);
        
        int num_def_images = 1;

        // loop over the displacement images
        for (int img = 0; img < num_def_images; img++) {
    
            // loop over strain windows within the image
            for (int sw = 0; sw < num_ss_x*num_ss_y; sw++){

                int x0 = ss_x[sw];
                int y0 = ss_y[sw];
                
                int x0_idx = sw % num_ss_x;
                int y0_idx = sw / num_ss_x;
                
                bool valid_window = fill_strain_window(ss_x, ss_y, u, v, window, num_ss_x, num_ss_y, x0_idx, y0_idx, sw_radius);
                
                // element coefficients
                Eigen::VectorXd uc;
                Eigen::VectorXd vc;

                // 2D deformation gradient matrix and identity matrix
                Eigen::Matrix2d deform_grad;
                Eigen::Matrix2d eps;
                Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

                if (valid_window){

                    // calculate derivatives
                    if (q==4){

                        // get polynomial coeffieicnts from smoothing
                        uc = smooth::q4(window.x ,window.y, window.u);
                        vc = smooth::q4(window.x ,window.y, window.v);

                        // populate 2D deformation gradient matrix
                        deform_grad(0,0) = 1.0 + uc[1] + uc[3]*y0;
                        deform_grad(0,1) = uc[2] + uc[3]*x0;
                        deform_grad(1,0) = vc[1] + vc[3]*y0;
                        deform_grad(1,1) = 1.0 + vc[2] + vc[3]*x0;

                        // calculate strain
                        if (formulation=="GREEN"){
                            eps = green(deform_grad, I);
                        }
                        else if (formulation=="ALMANSI"){
                            eps = almansi(deform_grad, I);
                        }
                        else if (formulation=="HENCKY"){
                            eps = hencky(deform_grad, I);
                        }
                        else if (formulation=="BIOT_EULER"){
                            eps = biot_euler(deform_grad, I);
                        }
                        else if (formulation=="BIOT_LAGRANGE"){
                            eps = biot_lagrange(deform_grad, I);
                        }

                        //std::cout << x0 << " " << y0 << " " << deform_grad(0,0) << " " << deform_grad(0,1) << " " << deform_grad(1,0) << " " << deform_grad(1,1) << std::endl; 
                        std::cout << x0 << " " << y0 << " " << x0_idx << " " << y0_idx << " " << eps(0,0) << " " << eps(0,1) << " " << eps(1,0) << " " << eps(1,1) << std::endl; 

                    }
                     if (q==9){

                        // get polynomial coeffieicnts from smoothing
                        uc = smooth::q9(window.x ,window.y, window.u);
                        vc = smooth::q9(window.x ,window.y, window.v);

                        // populate 2D deformation gradient matrix
                        deform_grad(0,0) = 1.0 + uc[1] + uc[3]*y0 + 2.0*uc[4]*x0 + 2.0*uc[6]*x0*y0 + uc[7]*y0*y0 + 2.0*uc[8]*x0*y0*y0;
                        deform_grad(0,1) = uc[2] + uc[3]*x0 + 2.0*uc[5]*y0 + uc[6]*x0*x0 + 2.0*uc[7]*x0*y0 + 2.0*uc[8]*x0*x0*y0;
                        deform_grad(1,0) = vc[1] + vc[3]*y0 + 2.0*vc[4]*x0 + 2.0*vc[6]*x0*y0 + vc[7]*y0*y0 + 2.0*vc[8]*x0*y0*y0;
                        deform_grad(1,1) = 1.0 + vc[2] + vc[3]*x0 + 2.0*vc[5]*y0 + vc[6]*x0*x0 + 2.0*vc[7]*x0*y0 + 2.0*vc[8]*x0*x0*y0;

                        // calculate strain
                        if (formulation=="GREEN"){
                            eps = green(deform_grad, I);
                        }
                        else if (formulation=="ALMANSI"){
                            eps = almansi(deform_grad, I);
                        }
                        else if (formulation=="HENCKY"){
                            eps = hencky(deform_grad, I);
                        }
                        else if (formulation=="BIOT_EULER"){
                            eps = biot_euler(deform_grad, I);
                        }
                        else if (formulation=="BIOT_LAGRANGE"){
                            eps = biot_lagrange(deform_grad, I);
                        }

                        //std::cout << x0 << " " << y0 << " " << deform_grad(0,0) << " " << deform_grad(0,1) << " " << deform_grad(1,0) << " " << deform_grad(1,1) << std::endl; 
                        std::cout << x0 << " " << y0 << " " << x0_idx << " " << y0_idx << " " << eps(0,0) << " " << eps(0,1) << " " << eps(1,0) << " " << eps(1,1) << std::endl; 

                    }
                }
            }
        }
    }



    bool fill_strain_window(int *ss_x, int *ss_y, 
                            double *u, double *v,
                            Window &window,
                            int num_ss_x, int num_ss_y, 
                            int x0_idx, int y0_idx, int swr){
        

        int xmin = x0_idx - swr;
        int xmax = x0_idx + swr;
        int ymin = y0_idx - swr;
        int ymax = y0_idx + swr;

        // check if the centre of strain window is within bounds
        if ((xmin < 0) || (xmax >= num_ss_x) || (ymin < 0) || (ymax >= num_ss_y)) return false;
        
        int idx = 0;
        for (int j = ymin; j <= ymax; j++){
            for (int i = xmin; i <= xmax; i++){

                // check if all subsets in the strain window are not nan
                if (std::isnan(u[num_ss_x*j + i]) || std::isnan(v[num_ss_x*j + i])) return false;

                window.x[idx] = ss_x[num_ss_x*j + i];
                window.y[idx] = ss_y[num_ss_x*j + i];
                window.u[idx] = u[num_ss_x*j + i];
                window.v[idx] = v[num_ss_x*j + i];

                //std::cout << window.x[idx] << " " << window.y[idx] << " " << window.u[idx] << " " << window.v[idx] << std::endl;
                idx++;
            }
        }
        //std::cout << std::endl;
        
        return true;
    }


    inline Eigen::Matrix2d green(Eigen::Matrix2d F, Eigen::Matrix2d I){
        return 0.5 * (F.transpose() * F - I);
    }





    inline Eigen::Matrix2d hencky(Eigen::Matrix2d F, Eigen::Matrix2d I){
        Eigen::Matrix2d C = F.transpose() * F;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // Get eigenvectors and sqrt-eigenvalues
        const Eigen::Matrix2d Q = solver.eigenvectors();
        const Eigen::Vector2d eigvals = solver.eigenvalues();

        return Q * (0.5 * eigvals.array().log().matrix().asDiagonal()) * Q.transpose();
    }




    inline Eigen::Matrix2d almansi(Eigen::Matrix2d F, Eigen::Matrix2d I){
        Eigen::Matrix2d B = F * F.transpose();
        Eigen::Matrix2d B_inv = B.inverse();
        return 0.5 * (I - B_inv); 
    }





    inline Eigen::Matrix2d biot_euler(Eigen::Matrix2d F, Eigen::Matrix2d I){
        
        Eigen::Matrix2d C = F * F.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // U = sqrt(C) = Q * sqrt(D) * Q^T
        Eigen::Matrix2d D_sqrt = solver.eigenvalues().cwiseSqrt().asDiagonal();
        Eigen::Matrix2d U = solver.eigenvectors() * D_sqrt * solver.eigenvectors().transpose();

        return U - I;

    }




    inline Eigen::Matrix2d biot_lagrange(Eigen::Matrix2d F, Eigen::Matrix2d I){
        
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
