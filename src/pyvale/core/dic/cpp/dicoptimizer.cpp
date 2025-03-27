// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>


// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicutil.hpp"


namespace optimizer {

    
    // parameters
    std::vector<double> p(6,0.0); // hard coded affine parameters
    std::vector<double> dp(6,0.0); // deltaP
    std::vector<double> pdp(6,0.0); // P + deltaP

    // reference subset values to be pulled from interpolator
    std::vector<double> ss_ref;
    std::vector<double> ss_ref_x;
    std::vector<double> ss_ref_y;

    // Optimization variables
    int iter; // number of iterations for each subset optimization
    double ftol; // tolerance for termination by the change of the cost function
    double xtol; // tolerance for termination by the change of the independent variables
    double lambda = 0.01; // damping
    double costfunc_p = 0.0; // cost function for current P values
    double costfunc_pdp = 0.0; // cost function for P+deltaP values
    std::vector<double> g(6, 0.0); // gradient
    std::vector<double> dfdp(6,0.0); // derivative of shape function with repsect to parameters
    std::vector<std::vector<double>> H(6, std::vector<double>(6, 0.0)); // Hessian. Also becomes (H + lambda * diag(H))
    std::vector<std::vector<double>> invH(6, std::vector<double>(6, 0.0)); // inverse of H + lambda * diag(H)


    // interpolation data struct
    interpolator::Data interp_data;


    // function pointer for correlation criteria
    void (*optimize_costfunc)(std::vector<double> &, std::vector<double> &, std::vector<double> &, int n);
    void (*shape_function)(double &, double &, double , double , std::vector<double> &);
    void (*dshape_dp)(std::vector<double>&, double, double, double, double, int);





    void init(std::string &corr_crit, std::string &shape_func, int ss_size){
        setCostFunction(corr_crit);
        setShapeFunction(shape_func);
        util::resize_ss(ss_ref, ss_ref_x, ss_ref_y, ss_size);
    }



    void solve(std::vector<double>& ss_def,
                    std::vector<double>& ss_def_x,
                    std::vector<double>& ss_def_y,
                    int n,
                    double tol,
                    int max_iter) {
        


        iter = 0;
        std::fill(p.begin(), p.end(), 0.0);

        while (iter < max_iter) {

            // perform the optimization
            optimize_costfunc(ss_def, ss_def_x, ss_def_y, n);
            update_lambda(costfunc_p, costfunc_pdp, p, pdp, lambda);



            ftol = std::abs(costfunc_pdp - costfunc_p) / std::abs(costfunc_p);
            xtol = std::sqrt(std::inner_product(dp.begin(), dp.end(), dp.begin(), 0.0));

            // if (xtol < tol || ftol < tol){
            if (xtol < tol){
                debugPrint(iter, ftol, xtol, p);
                break;
            }
            iter++;
        }

        // if its a bad subset and cant reach a minimum in the max number of iterations reset p values to 0.0
        if (iter == max_iter) {
            debugPrint(iter, ftol, xtol, p);
            // std::fill(p.begin(), p.end(), 0.0);
        }
    }




    void ssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
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
            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], p);

            double ref_x = ss_ref_x[i];
            double ref_y = ss_ref_y[i];
            
            // get the subset value and derivitives
            interp_data = interpolator::eval_bicubic_and_derivs(ref_x, ref_y);
            ss_ref[i] = interp_data.interp_value;
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

            double dshape_df = - (ss_def[i] - ss_ref[i]);
            g[0] += dshape_df * dfdx;
            g[1] += dshape_df * dfdy;
            g[2] += dshape_df * dfdx * ss_ref_x[i];
            g[3] += dshape_df * dfdx * ss_ref_y[i];
            g[4] += dshape_df * dfdy * ss_ref_x[i];
            g[5] += dshape_df * dfdy * ss_ref_y[i];

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
            costfunc_p += (ss_def[i] - ss_ref[i]) * (ss_def[i] - ss_ref[i]);
        }


        invertMatrix(H, invH);
        update_shapefunc_parameters(pdp, invH, g);

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
            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], pdp);
            ss_ref[i] = interpolator::eval_bicubic(ss_ref_x[i], ss_ref_y[i]);
            costfunc_pdp += (ss_def[i] - ss_ref[i]) * (ss_def[i] - ss_ref[i]);
        }
    }


    void nssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
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

            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], p);
            interp_data = interpolator::eval_bicubic_and_derivs(ss_ref_x[i], ss_ref_y[i]);
            ss_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;
            sum_squared_def += ss_def[i] * ss_def[i];
            sum_squared_ref += ss_ref[i] * ss_ref[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(dfdp, ss_ref_x[i], ss_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * (ss_def[i] * inv_sum_squared_def - ss_ref[i] * inv_sum_squared_ref);
            g[0] += dshape_df * dfdx[i];
            g[1] += dshape_df * dfdy[i];
            g[2] += dshape_df * dfdx[i] * ss_ref_x[i];
            g[3] += dshape_df * dfdx[i] * ss_ref_y[i];
            g[4] += dshape_df * dfdy[i] * ss_ref_x[i];
            g[5] += dshape_df * dfdy[i] * ss_ref_y[i];

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
        update_shapefunc_parameters(pdp, invH, g);


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
            double def_norm = ss_def[i] * inv_sum_squared_def;
            double ref_norm = ss_ref[i] * inv_sum_squared_ref;
            costfunc_p += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        costfunc_pdp = 0.0;
        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], pdp);
            ss_ref[i] = interpolator::eval_bicubic(ss_ref_x[i], ss_ref_y[i]);
            sum_squared_ref += ss_ref[i] * ss_ref[i];
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (int i = 0; i < n; ++i) {
            double def_norm = ss_def[i] * inv_sum_squared_def;
            double ref_norm = ss_ref[i] * inv_sum_squared_ref;
            costfunc_pdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }


    void znssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
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

            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], p);

            interp_data = interpolator::eval_bicubic_and_derivs(ss_ref_x[i], ss_ref_y[i]);
            ss_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;

            mean_ref += ss_ref[i];
            mean_def += ss_def[i];

        }

        mean_def /= n;
        mean_ref /= n;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_def += (ss_def[i] - mean_def) * (ss_def[i] - mean_def);
            sum_squared_ref += (ss_ref[i] - mean_ref) * (ss_ref[i] - mean_ref);
        }

        double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(dfdp, ss_ref_x[i], ss_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * ((ss_def[i] - mean_def) * inv_sum_squared_def - (ss_ref[i] - mean_ref) * inv_sum_squared_ref);
            g[0] += dshape_df * dfdx[i];
            g[1] += dshape_df * dfdy[i];
            g[2] += dshape_df * dfdx[i] * ss_ref_x[i];
            g[3] += dshape_df * dfdx[i] * ss_ref_y[i];
            g[4] += dshape_df * dfdy[i] * ss_ref_x[i];
            g[5] += dshape_df * dfdy[i] * ss_ref_y[i];

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
        update_shapefunc_parameters(pdp, invH, g);


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
            double def_norm = (ss_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (ss_ref[i] - mean_ref) * inv_sum_squared_ref;
            costfunc_p += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        mean_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(ss_ref_x[i], ss_ref_y[i], ss_def_x[i], ss_def_y[i], pdp);
            ss_ref[i] = interpolator::eval_bicubic(ss_ref_x[i], ss_ref_y[i]);
            mean_ref += ss_ref[i];
        }

        mean_ref /= n;

        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_ref += (ss_ref[i] - mean_ref) * (ss_ref[i] - mean_ref);
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        costfunc_pdp = 0.0;
        for (int i = 0; i < n; ++i) {
            double def_norm = (ss_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (ss_ref[i] - mean_ref) * inv_sum_squared_ref;
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

    
    void update_lambda(double costfunc_p, double costfunc_pdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda){

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

    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<std::vector<double>> &invH, std::vector<double> &g){

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

    void setCostFunction(const std::string& corr_crit) {
        if (corr_crit == "SSD") optimize_costfunc = ssd;
        else if (corr_crit == "NSSD") optimize_costfunc = nssd;
        else if (corr_crit == "ZNSSD") optimize_costfunc = znssd;
        else {
            std::cerr << "Unexpected Correlation Criteria: '" << corr_crit << "'" << std::endl;
            std::cerr << "Allowed Values: 'SSD', 'NSSD', 'ZNSSD'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void setShapeFunction(const std::string& shape_func) {
        if (shape_func == "rigid") {
            shape_function = rigid;
            dshape_dp = drigid_dp;
        } else if (shape_func == "affine") {
            shape_function = affine;
            dshape_dp = daffine_dp;
        } else {
            std::cerr << "Unexpected Shape Function: '" << shape_func << "'" << std::endl;
            std::cerr << "Allowed Values: 'rigid', 'affine'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void debugPrint(int iter, double ftol, double xtol, const std::vector<double>& p) {
        std::cout << iter << " " << ftol << " " << xtol << " ";
        std::cout << p[0] << " ";
        std::cout << p[1] << " ";
        std::cout << p[2] << " ";
        std::cout << p[3] << " ";
        std::cout << p[4] << " ";
        std::cout << p[5] << " ";
        std::cout << "\n";
    }


}