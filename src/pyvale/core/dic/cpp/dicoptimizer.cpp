// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <cmath>


// Program Header files
#include "./dicoptimizer.hpp"
#include "./dicinterpolator.hpp"


namespace optimizer {

    
    // values pulled from the interpolated reference image
    std::vector<double> p(6,0);
    std::vector<double> dp(6,0.0);
    std::vector<double> pdp(6,0.0);
    std::vector<double> p_plus_deltap(6,0);
    std::vector<double> subset_ref;
    std::vector<double> subset_ref_x;
    std::vector<double> subset_ref_y;
    std::vector<double> g(6, 0.0);
    std::vector<std::vector<double>> H(6, std::vector<double>(6, 0.0));
    std::vector<std::vector<double>> invH(6, std::vector<double>(6, 0.0));
    std::vector<double> dfdp(6,0.0);
    double lambda = 0.01;
    double costfunc_p;
    double costfunc_pdp;

    // interpolation data struct
    interpolator::Data interp_data;


    // function pointer for correlation criteria
    void (*optimize_costfunc)(std::vector<double> &, std::vector<double> &, std::vector<double> &, int n);
    void (*shape_function)(double &, double &, double , double , std::vector<double> &);
    void (*dshape_dp)(std::vector<double>&, double, double, double, double, int);

    void init(std::string &corr_crit, std::string &shape_func, int subset_size){


        // function pointer for cost function
        if (corr_crit == "SSD") optimize_costfunc=ssd;
        else if (corr_crit == "NSSD") optimize_costfunc=nssd;
        else if (corr_crit == "ZNSSD") optimize_costfunc=znssd;
        else {            
            std::cerr << "Unexpected Correlation Criteria: \'" << corr_crit << "\'" << std::endl;
            std::cerr << "Allowed Values: \'SSD\', \'NSSD\', \'ZNSSD\'. " << std::endl;
            exit(EXIT_FAILURE);
        }


        // function pointer for shape function
        if (shape_func == "rigid") {
            optimizer::shape_function=rigid;
            optimizer::dshape_dp=drigid_dp;
        }
        else if (shape_func == "affine") {
            optimizer::shape_function=affine;
            optimizer::dshape_dp=daffine_dp;
        }
        else {            
            std::cerr << "Unexpected Shape Function: \'" << shape_func << "\'" << std::endl;
            std::cerr << "Allowed Values: \'rigid\', \'affine\' " << std::endl;
            exit(EXIT_FAILURE);
        }

        

        // resize the reference subset values and coordinate vectors
        subset_ref.resize(subset_size*subset_size, 0.0);
        subset_ref_x.resize(subset_size*subset_size, 0.0);
        subset_ref_y.resize(subset_size*subset_size, 0.0);

    }


    void solve(std::vector<double> &subset_def,
                 std::vector<double> &subset_def_x,
                 std::vector<double> &subset_def_y,
                 int n,
                 double tol,
                 int max_iter){

        int iter = 0;

        double dp_mag;

        for (int l = 0; l < max_iter; l++){


            // optimize
            optimize_costfunc(subset_def, subset_def_x, subset_def_y, n);
            check_tolerance(costfunc_p, costfunc_pdp, p, pdp, lambda);


            // get magnitude of deltap
            dp_mag = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2] + dp[3]*dp[3] + dp[4]*dp[4] + dp[5]*dp[5]);
            
            //debugging
            // std::cout << iter << " " << dp_mag << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << "\n";
            // exit(0);

            if (dp_mag < tol) {
                std::cout << iter << " " << dp_mag << " " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << "\n";
                break;
            }
            
            iter++;
        }

        if (iter == max_iter) {
            std::cout << iter << " " << dp_mag << " " << nan << " " << nan << " " << nan << " " << nan << " " << nan << " " << nan << "\n";
            exit(0);
        }
    }

    void ssd(std::vector<double> &subset_def,
             std::vector<double> &subset_def_x, 
             std::vector<double> &subset_def_y, 
             int n){

        // reset derivative and hessian values
        std::fill(g.begin(), g.end(), 0.0);
        std::fill(dfdp.begin(), dfdp.end(), 0.0);
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);

        double dfdx;
        double dfdy;

        // loop over the subset values
        for (int i = 0; i < n; i++){

            // get subset coordinates based on shape function parameters
            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], p);

            double ref_x = subset_ref_x[i];
            double ref_y = subset_ref_y[i];
            
            // get the subset value and derivitives
            interp_data = interpolator::eval_bicubic_and_derivs(ref_x, ref_y);
            subset_ref[i] = interp_data.interp_value;
            dfdx = interp_data.interp_dx;
            dfdy = interp_data.interp_dy;

            // derivative of shape function with repsect to parameters
            dshape_dp(dfdp, ref_x, ref_y, dfdx, dfdy, n);
            
            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = dfdp[row];
                for (int col = row; col < 6; col++) {
                    H[row][col] += dfdp_row * dfdp[col];
                }
            }

            double dshape_df = - (subset_def[i] - subset_ref[i]);
            g[0] += dshape_df * dfdx;
            g[1] += dshape_df * dfdy;
            g[2] += dshape_df * dfdx * subset_ref_x[i];
            g[3] += dshape_df * dfdx * subset_ref_y[i];
            g[4] += dshape_df * dfdy * subset_ref_x[i];
            g[5] += dshape_df * dfdy * subset_ref_y[i];

        }

        // Populate lower triangle of hessian matrix
        for (int row = 0; row < 6; row++) {
            for (int col = row + 1; col < 6; col++) {
                H[col][row] = H[row][col];
            }
            H[row][row] += lambda * H[row][row]; // diagonal
        }

         
        // calculate cost function for current and updated parameter values 
        costfunc_p = 0.0;
        for (int i = 0; i < n; i++){
            costfunc_p += (subset_def[i] - subset_ref[i]) * (subset_def[i] - subset_ref[i]);
        }


        invertMatrix(H, invH);
        new_shape_func_params(pdp, invH, g);

        // Some useful debugging print statements
        //std::cout << df_dx << " " << df_dy << std::endl;
        // std::cout << "dfdp  " << dfdp[0] << " " << dfdp[1] << " " << dfdp[2] << " " << dfdp[3] << " " << dfdp[4] << " " << dfdp[5] << std::endl;
        // std::cout << "g  " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[4] << " " << g[5] << std::endl;
        // std::cout << "H  " << H[0][0] << " " << H[1][0] << " " << H[2][0] << " " << H[3][0] << " " << H[4][0] << " " << H[5][0] << std::endl;
        // std::cout << "Hi " << invH[0][0] << " " << invH[1][0] << " " << invH[2][0] << " " << invH[3][0] << " " << invH[4][0] << " " << invH[5][0] << std::endl;
        // std::cout << "p  " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
        // std::cout << "pd " << pdp[0] << " " << pdp[1] << " " << pdp[2] << " " << pdp[3] << " " << pdp[4] << " " << pdp[5] << std::endl;


        // calculate cost function for updated parameter values
        costfunc_pdp = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], pdp);
            subset_ref[i] = interpolator::eval_bicubic(subset_ref_x[i], subset_ref_y[i]);
            costfunc_pdp += (subset_def[i] - subset_ref[i]) * (subset_def[i] - subset_ref[i]);
        }
    }


    void nssd(std::vector<double> &subset_def,
             std::vector<double> &subset_def_x, 
             std::vector<double> &subset_def_y, 
             int n){


        // reset derivative and hessian values
        std::fill(g.begin(), g.end(), 0.0);
        std::fill(dfdp.begin(), dfdp.end(), 0.0);
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);

        std::vector<double> dfdx(n);
        std::vector<double> dfdy(n);

        double sum_squared_def = 0.0;
        double sum_squared_ref = 0.0; 
        double inv_sum_squared_def;
        double inv_sum_squared_ref;
        
        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < n; ++i) {

            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], p);
            interp_data = interpolator::eval_bicubic_and_derivs(subset_ref_x[i], subset_ref_y[i]);
            subset_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;
            sum_squared_def += subset_def[i] * subset_def[i];
            sum_squared_ref += subset_ref[i] * subset_ref[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(dfdp, subset_ref_x[i], subset_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * (subset_def[i] * inv_sum_squared_def - subset_ref[i] * inv_sum_squared_ref);
            g[0] += dshape_df * dfdx[i];
            g[1] += dshape_df * dfdy[i];
            g[2] += dshape_df * dfdx[i] * subset_ref_x[i];
            g[3] += dshape_df * dfdx[i] * subset_ref_y[i];
            g[4] += dshape_df * dfdy[i] * subset_ref_x[i];
            g[5] += dshape_df * dfdy[i] * subset_ref_y[i];

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = dfdp[row];
                for (int col = row; col < 6; col++) {
                    H[row][col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * dfdp[col];
                }
            }
        }

        // Populate lower triangle of hessian matrix
        for (int row = 0; row < 6; row++) {
            for (int col = row + 1; col < 6; col++) {
                H[col][row] = H[row][col];
            }
            H[row][row] += lambda * H[row][row]; // diagonal
        }


        invertMatrix(H, invH);
        new_shape_func_params(pdp, invH, g);


        //std::cout << df_dx << " " << df_dy << std::endl;
        // std::cout << "dfdp  " << dfdp[0] << " " << dfdp[1] << " " << dfdp[2] << " " << dfdp[3] << " " << dfdp[4] << " " << dfdp[5] << std::endl;
        // std::cout << "g  " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[4] << " " << g[5] << std::endl;
        // std::cout << "H  " << H[0][0] << " " << H[1][0] << " " << H[2][0] << " " << H[3][0] << " " << H[4][0] << " " << H[5][0] << std::endl;
        // std::cout << "Hi " << invH[0][0] << " " << invH[1][0] << " " << invH[2][0] << " " << invH[3][0] << " " << invH[4][0] << " " << invH[5][0] << std::endl;
        // std::cout << "p  " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
        // std::cout << "pd " << pdp[0] << " " << pdp[1] << " " << pdp[2] << " " << pdp[3] << " " << pdp[4] << " " << pdp[5] << std::endl;


        // calculate cost function for current parameter values
        costfunc_p = 0.0;
        for (int i = 0; i < n; i++){
            double def_norm = subset_def[i] * inv_sum_squared_def;
            double ref_norm = subset_ref[i] * inv_sum_squared_ref;
            costfunc_p += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        costfunc_pdp = 0.0;
        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], pdp);
            subset_ref[i] = interpolator::eval_bicubic(subset_ref_x[i], subset_ref_y[i]);
            sum_squared_ref += subset_ref[i] * subset_ref[i];
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (int i = 0; i < n; ++i) {
            double def_norm = subset_def[i] * inv_sum_squared_def;
            double ref_norm = subset_ref[i] * inv_sum_squared_ref;
            costfunc_pdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }


    void znssd(std::vector<double> &subset_def,
             std::vector<double> &subset_def_x, 
             std::vector<double> &subset_def_y, 
             int n){


        // reset derivative and hessian values
        std::fill(g.begin(), g.end(), 0.0);
        std::fill(dfdp.begin(), dfdp.end(), 0.0);
        for (auto& row : H) std::fill(row.begin(), row.end(), 0.0);

        std::vector<double> dfdx(n);
        std::vector<double> dfdy(n);

        double mean_ref = 0.0;
        double mean_def = 0.0;
        
        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < n; ++i) {

            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], p);

            interp_data = interpolator::eval_bicubic_and_derivs(subset_ref_x[i], subset_ref_y[i]);
            subset_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;

            mean_ref += subset_ref[i];
            mean_def += subset_def[i];

        }

        mean_def /= n;
        mean_ref /= n;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_def += (subset_def[i] - mean_def) * (subset_def[i] - mean_def);
            sum_squared_ref += (subset_ref[i] - mean_ref) * (subset_ref[i] - mean_ref);
        }

        double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(dfdp, subset_ref_x[i], subset_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * ((subset_def[i] - mean_def) * inv_sum_squared_def - (subset_ref[i] - mean_ref) * inv_sum_squared_ref);
            g[0] += dshape_df * dfdx[i];
            g[1] += dshape_df * dfdy[i];
            g[2] += dshape_df * dfdx[i] * subset_ref_x[i];
            g[3] += dshape_df * dfdx[i] * subset_ref_y[i];
            g[4] += dshape_df * dfdy[i] * subset_ref_x[i];
            g[5] += dshape_df * dfdy[i] * subset_ref_y[i];

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = dfdp[row];
                for (int col = row; col < 6; col++) {
                    H[row][col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * dfdp[col];
                }
            }
        }

        // Populate lower triangle of hessian matrix and lead diagonal
        for (int row = 0; row < 6; row++) {
            for (int col = row + 1; col < 6; col++) {
                H[col][row] = H[row][col];
            }
            H[row][row] += lambda * H[row][row]; // diagonal
        }


        invertMatrix(H, invH);
        new_shape_func_params(pdp, invH, g);


        //std::cout << df_dx << " " << df_dy << std::endl;
        // std::cout << "dfdp  " << dfdp[0] << " " << dfdp[1] << " " << dfdp[2] << " " << dfdp[3] << " " << dfdp[4] << " " << dfdp[5] << std::endl;
        // std::cout << "g  " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[4] << " " << g[5] << std::endl;
        // std::cout << "H  " << H[0][0] << " " << H[1][0] << " " << H[2][0] << " " << H[3][0] << " " << H[4][0] << " " << H[5][0] << std::endl;
        // std::cout << "Hi " << invH[0][0] << " " << invH[1][0] << " " << invH[2][0] << " " << invH[3][0] << " " << invH[4][0] << " " << invH[5][0] << std::endl;
        // std::cout << "p  " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
        // std::cout << "pd " << pdp[0] << " " << pdp[1] << " " << pdp[2] << " " << pdp[3] << " " << pdp[4] << " " << pdp[5] << std::endl;


        // calculate cost function for current parameter values
        costfunc_p = 0.0;
        for (int i = 0; i < n; i++){
            double def_norm = (subset_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (subset_ref[i] - mean_ref) * inv_sum_squared_ref;
            costfunc_p += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        mean_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(subset_ref_x[i], subset_ref_y[i], subset_def_x[i], subset_def_y[i], pdp);
            subset_ref[i] = interpolator::eval_bicubic(subset_ref_x[i], subset_ref_y[i]);
            mean_ref += subset_ref[i];
        }

        mean_ref /= n;

        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_ref += (subset_ref[i] - mean_ref) * (subset_ref[i] - mean_ref);
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        costfunc_pdp = 0.0;
        for (int i = 0; i < n; ++i) {
            double def_norm = (subset_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (subset_ref[i] - mean_ref) * inv_sum_squared_ref;
            costfunc_pdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }

    // Inv matrix using Gauss Elim.
    bool invertMatrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& inverse) {
        int n = 6;

        // Aug matrix with I matrix.
        std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n));        
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gauss elim
        for (int i = 0; i < n; ++i) {
            // Search for max in col
            double maxEl = abs(augmented[i][i]);
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (abs(augmented[k][i]) > maxEl) {
                    maxEl = abs(augmented[k][i]);
                    maxRow = k;
                }
            }

            // Swap max row wi col.
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

    
    void check_tolerance(double costfunc_p, double costfunc_pdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda){

        if (costfunc_p < costfunc_pdp){
            lambda *= 10.0;
        }
        else{
            lambda *= 0.1;
            for (int i = 0; i < 6; i++){
                p[i] = pdp[i];
            }
        }
        
    }

    void new_shape_func_params(std::vector<double> &pdp, std::vector<std::vector<double>> &invH, std::vector<double> &g){

        // multiply inverse with gradient
        for (int i = 0; i < 6; ++i) {
            dp[i] = 0.0;
            for (int j = 0; j < 6; ++j) {
                dp[i] +=  1.0 * invH[i][j] * g[j];
            }
        }

        // add p to delta p
        for (int i = 0; i < 6; ++i) {
            pdp[i] = p[i] - dp[i];
        }
    }



    void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p){
        x_new = p[0] + (1 + p[2]) * x + p[3] * y;
        y_new = p[1] + (1 + p[5]) * y + p[4] * x;
    }

    void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p){
        x_new = p[0] + y;
        y_new = p[1] + x;
    }

    void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p){

    }

    void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n){

        dfdp[0] = dfdx;
        dfdp[1] = dfdy;
        dfdp[2] = dfdx * x;
        dfdp[3] = dfdx * y;
        dfdp[4] = dfdy * x;
        dfdp[5] = dfdy * y;


    }

    void drigid_dp(std::vector<double> &dfdp, double x, double y,  double dfdx, double dfdy, int n){

            dfdp[0] = dfdx;
            dfdp[1] = dfdy;
    }

    void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p){

    }




}