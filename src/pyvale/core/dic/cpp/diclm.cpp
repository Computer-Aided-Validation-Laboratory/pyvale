// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <chrono>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>
#include <Eigen/Dense>

// Program Header files
#include "./diclm.hpp"
#include "./dicinterpolator.hpp"


namespace lm {

    
    // values pulled from the interpolated reference image
    std::vector<double> p(6,0);
    std::vector<double> dp(6,0.0);
    std::vector<double> pdp(6,0.0);
    std::vector<double> p_plus_deltap(6,0);
    std::vector<double> subset_ref;
    std::vector<double> subset_ref_x;
    std::vector<double> subset_ref_y;
    std::vector<double> gradient(6, 0.0);
    std::vector<double> hessian(36,0.0);
    std::vector<double> dfdp(6,0.0);
    double lambda = 0.1;

    double inv_sum_squared_def;
    double inv_sum_squared_ref;
    double mean_def;
    double mean_ref;
    double costfunc_p;
    double costfunc_pdp;



    void init(int n){
        subset_ref.resize(n, 0.0);
        subset_ref_x.resize(n, 0.0);
        subset_ref_y.resize(n, 0.0);
    }


    void loop(std::vector<double> &subset_def,
                 std::vector<double> &subset_def_x,
                 std::vector<double> &subset_def_y,
                 gsl_spline2d *spline,
                 gsl_interp_accel* xacc,
                 gsl_interp_accel* yacc,
                 int n){

            // std::cout << __FILE__ << " " << __LINE__ << std::endl;
            std::cout << "==============================================================" << std::endl;

            // initialise p values
            p[0] = -0.0;
            p[1] = 0.0;
            // p[2] = 0.0;
            // p[3] = 0.0;
            // p[4] = 0.0;
            // p[5] = 0.0;

            std::cout << "p:      " << std::endl;
            for (int i = 0; i < 6; i++){
                std::cout << p[i] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;

            // std::cout << __FILE__ << " " << __LINE__ << std::endl;
            int iter = 0;
            // some kind of loop
            for (int l = 0; l < 5000; l++){
            // std::cout << __FILE__ << " " << __LINE__ << std::endl;

            // get parameters associated with the deformed subset

            subset_def_params(subset_def, n);
            // std::cout << __FILE__ << " " << __LINE__ << std::endl;

            // get parameters assosciated with the reference subset.
            // includes subset values and coords based on interpolation
            subset_ref_params(p, subset_def_x, subset_def_y, spline, xacc, yacc, n);

            //checking def and ref subset values
            // for (int i = 0; i < n; i++){
            //     std::cout << subset_def[i] << " " << subset_ref[i] << std::endl;
            // }
            // std::cout << std::endl;


            // std::cout << __FILE__ << " " << __LINE__ << std::endl;
            // calculate the gradient 
            calculate_gradient(subset_def, spline, xacc, yacc, n);


            std::cout << "gradient: " << std::endl;
            for (int i = 0; i < 6; i++){
                std::cout << gradient[i] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;


            // std::cout << __FILE__ << " " << __LINE__ << std::endl;
            // calculate the hessian 
            calculate_hessian();

            int count = 0;
            std::cout << "hessian: " << std::endl;
            for (int i = 0; i < 6; i++){
                for (int j = 0; j < 6; j++){
                    std::cout << hessian[count] << " ";
                    count++;
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;


            // calculate delta p
            calculate_deltap();
            std::cout << "dp:     ";
            for (int i = 0; i < 6; i++){
                std::cout << dp[i] << " ";
            }
            std::cout << std::endl;
            std::cout << std::endl;


            costfunc_p = 0.0;
            calculate_costfunc_p(subset_ref, subset_def, n);

            std::cout << "costfunc_p:" << std::endl;
            std::cout << costfunc_p << std::endl;
            std::cout << std::endl;

            costfunc_pdp = 0.0;
            calculate_costfunc_pdp(subset_def, subset_def_x, subset_def_y, spline, xacc, yacc, n);
            std::cout << "costfunc_pdp:" << std::endl;
            std::cout << costfunc_pdp << std::endl;
            std::cout << std::endl;

            std::cout << "lambda:" << std::endl;
            std::cout << lambda << std::endl;
            std::cout << std::endl;

            if (costfunc_p < costfunc_pdp){
                lambda *= 5.0;
                iter++;
            }
            else{
                lambda *= 0.01;
                iter++;
                // initialise p values
                for (int i = 0; i < 6; i++){
                    p[i] = pdp[i];
                }
            }


            // std::cout << p[0] << " " << p[1] << std::endl;
            std::cout << iter << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
            std::cout << "==============================================================" << std::endl;

            
        }





    }
    


    void subset_def_params(std::vector<double> &subset_def, int n){

        mean_def = 0.0;
        
        //get the mean values
        for (int i = 0; i < n; ++i) {
            mean_def += subset_def[i];
        }

        // normalise the mean values
        mean_def /= n;
        std::cout << "mean_def:" << std::endl;
        std::cout << mean_def << std::endl;
        std::cout << std::endl;

        // (f(x,y) - f_mean)**2
        double sum_squared_def = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_def += (subset_def[i] -  mean_def) * (subset_def[i] -  mean_def);
        }

        // 1.0 / (d(x,y) - d_mean)**2
        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);

        std::cout << "inv_sum_squared_def:" << std::endl;
        std::cout << inv_sum_squared_def << std::endl;       
        std::cout << std::endl;

    }
    



    void subset_ref_params(std::vector<double> &p,
                           std::vector<double> &subset_def_x, 
                           std::vector<double> &subset_def_y, 
                           gsl_spline2d* spline,
                           gsl_interp_accel* xacc,
                           gsl_interp_accel* yacc,
                           int n){


        mean_ref = 0.0;

        //get the mean values
        for (int i = 0; i < n; ++i) {

            double x = subset_def_x[i];
            double y = subset_def_y[i];

            subset_ref_x[i] = p[0] + (1 + p[2]) * x + p[3] * y;
            subset_ref_y[i] = p[1] + (1 + p[5]) * y + p[4] * x;

            // subset_ref_x[i] = p[0] + x;
            // subset_ref_y[i] = p[1] + y;


            // std::cout << x << " " << y << " " << subset_ref_x[i] << " " << subset_ref_y[i] << std::endl;

            subset_ref[i] = gsl_spline2d_eval(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            mean_ref += subset_ref[i];
        }

        // normalise the mean values
        mean_ref /= n;

        std::cout << "mean_ref:" << std::endl;
        std::cout << mean_ref << std::endl;
        std::cout << std::endl;

        // (f(x,y) - f_mean)**2
        double sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_ref += (subset_ref[i] - mean_ref) * (subset_ref[i] - mean_ref);
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);
        std::cout << "inv_sum_squared_ref:" << std::endl;
        std::cout << inv_sum_squared_ref << std::endl;
        std::cout << std::endl;

    }



    void calculate_gradient(
                 std::vector<double> &subset_def,
                 gsl_spline2d* spline,
                 gsl_interp_accel* xacc,
                 gsl_interp_accel* yacc,
                 int n){

        // reset gradient values
        gradient[0] = 0.0;
        gradient[1] = 0.0;
        gradient[2] = 0.0;
        gradient[3] = 0.0;
        gradient[4] = 0.0;
        gradient[5] = 0.0;
        dfdp[0] = 0.0;
        dfdp[1] = 0.0;
        dfdp[2] = 0.0;
        dfdp[3] = 0.0;
        dfdp[4] = 0.0;
        dfdp[5] = 0.0;

        // loop over the subset values
        for (int i = 0; i < n; i++){

            double prefactor  = -2.0 * inv_sum_squared_ref * ((subset_def[i] - mean_def) * inv_sum_squared_def - (subset_ref[i] - mean_ref) * inv_sum_squared_ref);
            // std::cout << i << " " << prefactor << std::endl;

            double df_dx = gsl_spline2d_eval_deriv_x(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            double df_dy = gsl_spline2d_eval_deriv_y(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            
            dfdp[0] += df_dx;
            dfdp[1] += df_dy;
            dfdp[2] += df_dx * subset_ref_x[i];
            dfdp[3] += df_dx * subset_ref_y[i];
            dfdp[4] += df_dy * subset_ref_x[i];
            dfdp[5] += df_dy * subset_ref_y[i];

            gradient[0] += prefactor * df_dx;
            gradient[1] += prefactor * df_dy;
            gradient[2] += prefactor * df_dx * subset_ref_x[i];
            gradient[3] += prefactor * df_dx * subset_ref_y[i];
            gradient[4] += prefactor * df_dy * subset_ref_x[i];
            gradient[5] += prefactor * df_dy * subset_ref_y[i];

        }
    }

    void calculate_hessian(){

        double const_term = 2.0 * inv_sum_squared_ref * inv_sum_squared_ref;

        // calculate the hessian
        int count = 0;
        for (int row = 0; row < 6; row++){
            for (int col = 0; col < 6; col++){

                hessian[count] =  const_term * dfdp[row] * dfdp[col];
                
                if (row == col){
                    hessian[count] += lambda * hessian[count];
                }

                count++;
            }
        }
    } 

    void calculate_costfunc_p(std::vector<double> &subset_ref, std::vector<double> &subset_def, int n){

        for (int i = 0; i < n; i++){

            double term_def = (subset_def[i] - mean_def) * inv_sum_squared_def;
            double term_ref = (subset_ref[i] - mean_ref) * inv_sum_squared_ref;
            costfunc_p += (term_def - term_ref) * (term_def - term_ref);

        }
    }

    void calculate_costfunc_pdp(std::vector<double> &subset_def, std::vector<double> &subset_def_x, std::vector<double> &subset_def_y,
                                gsl_spline2d *spline, gsl_interp_accel* xacc, gsl_interp_accel* yacc, int n){

        
        // mean values of reference and deformed subset
        mean_ref = 0.0;

        //get the mean values
        for (int i = 0; i < n; ++i) {

            double x = subset_def_x[i];
            double y = subset_def_y[i];

            // affine
            subset_ref_x[i] = pdp[0] + (1 + pdp[2]) * x + pdp[3] * y;
            subset_ref_y[i] = pdp[1] + (1 + pdp[5]) * y + pdp[4] * x;

            // rigid
            // subset_ref_x[i] = pdp[0] + x;
            // subset_ref_y[i] = pdp[1] + y;

            // std::cout << x << " " << y << " " << subset_ref_x[i] << " " << subset_ref_y[i] << std::endl;

            if (subset_ref_x[i] < 0 || subset_ref_x[i] > 199 || subset_ref_y[i] < 0 || subset_ref_y[i] > 199) {
                    costfunc_pdp = 1.0e7;
                    return;
            }


            subset_ref[i] = gsl_spline2d_eval(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            mean_ref += subset_ref[i];
        }

        // normalise the mean values
        mean_ref /= n;

        // (f(x,y) - f_mean)**2
        double sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_ref += (subset_ref[i] - mean_ref) * (subset_ref[i] - mean_ref);
        }

         inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (int i = 0; i < n; i++){

            double term_def = (subset_def[i] - mean_def) * inv_sum_squared_def;
            double term_ref = (subset_ref[i] - mean_ref) * inv_sum_squared_ref;

            costfunc_pdp += (term_def - term_ref) * (term_def - term_ref);

        }
    }

    void calculate_deltap() {

        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> mat(hessian.data());

        // Compute inverse
        Eigen::Matrix<double, 6, 6> invMat = mat.inverse();

        dp[0] = 0.0;
        dp[1] = 0.0;
        dp[2] = 0.0;
        dp[3] = 0.0;
        dp[4] = 0.0;
        dp[5] = 0.0;

        // multiply inverse with gradient
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                dp[i] +=  -1.0 * invMat(i,j) * gradient[j];
            }
        }

        // add p to delta p
        for (int i = 0; i < 6; ++i) {
            pdp[i] = p[i] + dp[i];
        }

    }


}