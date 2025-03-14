// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <iomanip>
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
    std::vector<double> q(6, 0.0);
    std::vector<std::vector<double>> H(6, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> invH(6, std::vector<double>(6, 0.0));
    std::vector<double> dfdp(6,0.0);
    double lambda = 0.01;
    double tol = 0.001;
    double dp_mag;

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
        // std::cout << "==============================================================" << std::endl;

        // initialise p values
        // p[0] = 0.0;
        // p[1] = 0.0;
        // p[2] = 0.0;
        // p[3] = 0.0;
        // p[4] = 0.0;
        // p[5] = 0.0;

        // std::cout << "p:      " << std::endl;
        // for (int i = 0; i < 6; i++){
        //     std::cout << p[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        int iter = 0;

        // some kind of loop
        for (int l = 0; l < 20; l++){

            calculate_q(subset_def, subset_def_x, subset_def_y,spline, xacc, yacc, n);
            calculate_hessian();
            calculate_deltap();
            calculate_costfunc_p(subset_ref, subset_def, n);
            calculate_costfunc_pdp(subset_def, subset_def_x, subset_def_y, spline, xacc, yacc, n);
            


            if (costfunc_p < costfunc_pdp){
                lambda *= 10.0;
            }
            else{
                lambda *= 0.1;
                // initialise p values
                for (int i = 0; i < 6; i++){
                    p[i] = pdp[i];
                }
            }

            iter++;
            // std::cout << iter << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
            
            // get magnitude of deltap
            dp_mag = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2] + dp[3]*dp[3] + dp[4]*dp[4] + dp[5]*dp[5]);
            // std::cout << dp_mag << std::endl;
            if (dp_mag < tol) {
                std::cout << iter << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << "\n";
                break;
            }
            
        }





    }

    void calculate_q(
                 std::vector<double> &subset_def,
                    std::vector<double> &subset_coords_x, 
                    std::vector<double> &subset_coords_y, 
                 gsl_spline2d* spline,
                 gsl_interp_accel* xacc,
                 gsl_interp_accel* yacc,
                 int n){

        // reset gradient values
        std::fill(q.begin(), q.end(), 0.0);
        std::fill(dfdp.begin(), dfdp.end(), 0.0);

        double x, y, df_dx, df_dy;

        // loop over the subset values
        for (int i = 0; i < n; i++){

            x = subset_coords_x[i];
            y = subset_coords_y[i];

            subset_ref_x[i] = p[0] + (1 + p[2]) * x + p[3] * y;
            subset_ref_y[i] = p[1] + (1 + p[5]) * y + p[4] * x;
            subset_ref[i] = gsl_spline2d_eval(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);

            df_dx = gsl_spline2d_eval_deriv_x(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            df_dy = gsl_spline2d_eval_deriv_y(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            
            dfdp[0] = df_dx;
            dfdp[1] = df_dy;
            dfdp[2] = df_dx * subset_ref_x[i];
            dfdp[3] = df_dx * subset_ref_y[i];
            dfdp[4] = df_dy * subset_ref_x[i];
            dfdp[5] = df_dy * subset_ref_y[i];

            // Update Hessian matrix - utilize symmetry
            for (int row = 0; row < 6; row++) {
                for (int col = row; col < 6; col++) {
                    H[row][col] += dfdp[row] * dfdp[col];
                    if (row != col) {
                        H[col][row] = H[row][col];
                    }
                }
            }


            double residual = (subset_ref[i] - subset_def[i]);
            q[0] += residual * df_dx;
            q[1] += residual * df_dy;
            q[2] += residual * df_dx * subset_ref_x[i];
            q[3] += residual * df_dx * subset_ref_y[i];
            q[4] += residual * df_dy * subset_ref_x[i];
            q[5] += residual * df_dy * subset_ref_y[i];
        }

        // Update Hessian matrix - utilize symmetry
        for (int diag = 0; diag < 6; diag++) {
            H[diag][diag] += lambda * H[diag][diag];
        }
    }

    void calculate_hessian(){
        // for (int row = 0; row < 6; row++){
        //     for (int col = 0; col < 6; col++){
        //         H[row][col] = dfdp[row] * dfdp[col];
        //     }
        // }
    }

    void calculate_costfunc_p(std::vector<double> &subset_ref, std::vector<double> &subset_def, int n){
        costfunc_p = 0.0;
        for (int i = 0; i < n; i++){

            double term_def = (subset_def[i] - mean_def) * inv_sum_squared_def;
            double term_ref = (subset_ref[i] - mean_ref) * inv_sum_squared_ref;
            // costfunc_p += (term_def - term_ref) * (term_def - term_ref);
            costfunc_p += (subset_ref[i] - subset_def[i]) * (subset_ref[i] - subset_def[i]);

        }
    }

    void calculate_costfunc_pdp(std::vector<double> &subset_def, std::vector<double> &subset_def_x, std::vector<double> &subset_def_y,
                                gsl_spline2d *spline, gsl_interp_accel* xacc, gsl_interp_accel* yacc, int n){

        
        mean_ref = 0.0;
        costfunc_pdp = 0.0;

        //get the mean values
        for (int i = 0; i < n; ++i) {

            double x = subset_def_x[i];
            double y = subset_def_y[i];

            // affine
            subset_ref_x[i] = pdp[0] + (1 + pdp[2]) * x + pdp[3] * y;
            subset_ref_y[i] = pdp[1] + (1 + pdp[5]) * y + pdp[4] * x;

            if (subset_ref_x[i] < 0 || subset_ref_x[i] > 1040 || subset_ref_y[i] < 0 || subset_ref_y[i] > 1540) {
                    costfunc_pdp = 1.0e9;
                    return;
            }


            subset_ref[i] = gsl_spline2d_eval(spline, subset_ref_x[i], subset_ref_y[i], xacc, yacc);
            mean_ref += subset_ref[i];

            costfunc_pdp += (subset_ref[i] - subset_def[i]) * (subset_ref[i] - subset_def[i]);

        }
    }

    void calculate_deltap() {

        // Eigen::Matrix<double, 6, 6> mat;
        // for (int i = 0; i < 6; i++) {
        //     for (int j = 0; j < 6; j++) {
        //         mat(i, j) = H[i][j];
        //     }
        // }
        // Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> mat(hessian.data());


        // Compute inverse
        // Eigen::Matrix<double, 6, 6> invH = mat.inverse();

        invertMatrix(H, invH);

        // int count = 0;
        // std::cout << "inverse H: " << std::endl;
        // for (int i = 0; i < 6; i++){
        //     for (int j = 0; j < 6; j++){
        //         std::cout << invH(i,j) << " ";
        //         // std::cout << inverse[i][j] << " ";
        //         count++;
        //     }
        //     std::cout << std::endl;
        // }
        // std::cout << std::endl;

        dp[0] = 0.0;
        dp[1] = 0.0;
        dp[2] = 0.0;
        dp[3] = 0.0;
        dp[4] = 0.0;
        dp[5] = 0.0;

        // multiply inverse with gradient
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                // dp[i] +=  1.0 * invH(i,j) * q[j];
                dp[i] +=  1.0 * invH[i][j] * q[j];
            }
        }

        // add p to delta p
        for (int i = 0; i < 6; ++i) {
            pdp[i] = p[i] - dp[i];
        }

    }



        // Function to perform matrix inversion using Gaussian elimination
    bool invertMatrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& inverse) {
        int n = 6;

        // Create an augmented matrix: [matrix | identity matrix]
        std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n));
        
        // Initialize augmented matrix with matrix on the left and identity on the right
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Perform Gaussian elimination
        for (int i = 0; i < n; ++i) {
            // Search for maximum in this column (pivoting)
            double maxEl = abs(augmented[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (abs(augmented[k][i]) > maxEl) {
                    maxEl = abs(augmented[k][i]);
                    maxRow = k;
                }
            }

            // Swap maximum row with current row
            if (i != maxRow) {
                swap(augmented[i], augmented[maxRow]);
            }

            // Make the pivot element 1 by dividing the row by the pivot element
            double pivot = augmented[i][i];
            if (pivot == 0) {
                return false; // Singular matrix, can't invert
            }

            for (int j = 0; j < 2 * n; ++j) {
                augmented[i][j] /= pivot;
            }

            // Make the elements below the pivot 0
            for (int k = i + 1; k < n; ++k) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k][j] -= augmented[i][j] * factor;
                }
            }
        }

        // Perform back substitution to eliminate entries above the pivot
        for (int i = n - 1; i >= 0; --i) {
            for (int k = i - 1; k >= 0; --k) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k][j] -= augmented[i][j] * factor;
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        return true;
    }


}