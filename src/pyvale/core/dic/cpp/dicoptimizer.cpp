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




    // function pointer for correlation criteria
    void (*optimize_cost)(std::vector<double> &, std::vector<double> &, std::vector<double> &, int n, Parameters *params);
    void (*shape_function)(double &, double &, double , double , std::vector<double> &);
    void (*dshape_dp)(std::vector<double>&, double, double, double, double, int);





    void init(std::string &corr_crit, std::string &shape_func){
        setCostFunction(corr_crit);
        setShapeFunction(shape_func);
    }



    Results solve(double ss_x, double ss_y, std::vector<double>& ss_def,
                    std::vector<double>& ss_def_x,
                    std::vector<double>& ss_def_y,
                    int n,
                    double tol,
                    int max_iter,
                    Parameters *params) {

        params->iter = 0;

        while (params->iter < max_iter) {

            // perform the optimization
            optimize_cost(ss_def, ss_def_x, ss_def_y, n, params);
            update_lambda(params->costp, params->costpdp, params->p, params->pdp, params->lambda);



            params->ftol = std::abs(params->costpdp - params->costp) / std::abs(params->costp);
            params->xtol = std::sqrt(std::inner_product(params->dp.begin(), params->dp.end(), params->dp.begin(), 0.0));

            if (params->xtol < tol || params->ftol < tol){
                // debugPrint(params->iter, params->ftol, params->xtol, params->p);
                break;
            }
            params->iter++;
        }

        // if its a bad subset and hits max_iterations then reset p values to 0.0 to prevent bad seeding.
        if (params->iter == max_iter) {
            debugPrint(params->iter, params->ftol, params->xtol, params->p);
            std::fill(params->p.begin(), params->p.end(), 0.0);
        }

        Results results;
        affine_parameters_to_displacement(&results, ss_x, ss_y, params->p);
        results.iter = params->iter;
        results.ftol = params->ftol;
        results.xtol = params->xtol;
        results.p = params->p;

        return results;
    }




    void ssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
             int n,
             Parameters *params){

        double dfdx;
        double dfdy;

        // interpolation data struct
        interpolator::Data interp_data;

        // reset derivative and hessian values
        std::fill(params->g.begin(), params->g.end(), 0.0);
        std::fill(params->H.begin(), params->H.end(), 0.0);

        // loop over the subset values
        for (int i = 0; i < n; i++){

            // get subset coordinates based on shape function parameters
            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->p);

            double ref_x = params->ss_ref_x[i];
            double ref_y = params->ss_ref_y[i];
            
            // get the subset value and derivitives
            interp_data = interpolator::eval_bicubic_and_derivs(ref_x, ref_y);
            params->ss_ref[i] = interp_data.interp_value;
            double ref = params->ss_ref[i];

            dfdx = interp_data.interp_dx;
            dfdy = interp_data.interp_dy;

            // derivative of shape function with repsect to parameters
            dshape_dp(params->dfdp, ref_x, ref_y, dfdx, dfdy, n);
            
            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = params->dfdp[row];
                for (int col = row; col < 6; col++) {
                    params->H[row * 6 + col] += dfdp_row * params->dfdp[col];
                }
            }

            double dshape_df = - (ss_def[i] - ref);
            params->g[0] += dshape_df * dfdx;
            params->g[1] += dshape_df * dfdy;
            params->g[2] += dshape_df * dfdx * ref_x;
            params->g[3] += dshape_df * dfdx * ref_y;
            params->g[4] += dshape_df * dfdy * ref_x;
            params->g[5] += dshape_df * dfdy * ref_y;

        }

        populate_hessian_lower_tri(params->H, params->lambda);
        invertMatrix(params->H, params->invH);
        update_shapefunc_parameters(params->pdp, params->p, params->dp, params->invH, params->g);

        // calculate cost function for current and updated parameter values 
        params->costp = 0.0;
        for (int i = 0; i < n; i++){
            params->costp += (ss_def[i] - params->ss_ref[i]) * (ss_def[i] - params->ss_ref[i]);
        }


        // calculate cost function for updated parameter values
        params->costpdp = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->pdp);
            params->ss_ref[i] = interpolator::eval_bicubic(params->ss_ref_x[i], params->ss_ref_y[i]);
            params->costpdp += (ss_def[i] - params->ss_ref[i]) * (ss_def[i] - params->ss_ref[i]);
        }
    }


    void nssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
             int n,
            Parameters *params){


        // reset derivative and hessian values
        std::fill(params->g.begin(), params->g.end(), 0.0);
        std::fill(params->H.begin(), params->H.end(), 0.0);

        std::vector<double> dfdx(n);
        std::vector<double> dfdy(n);

        double sum_squared_def = 0.0;
        double sum_squared_ref = 0.0; 
        double inv_sum_squared_def;
        double inv_sum_squared_ref;

        // interpolation data struct
        interpolator::Data interp_data;

        
        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < n; ++i) {

            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->p);
            interp_data = interpolator::eval_bicubic_and_derivs(params->ss_ref_x[i], params->ss_ref_y[i]);
            params->ss_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;
            sum_squared_def += ss_def[i] * ss_def[i];
            sum_squared_ref += params->ss_ref[i] * params->ss_ref[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(params->dfdp, params->ss_ref_x[i], params->ss_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * (ss_def[i] * inv_sum_squared_def - params->ss_ref[i] * inv_sum_squared_ref);
            params->g[0] += dshape_df * dfdx[i];
            params->g[1] += dshape_df * dfdy[i];
            params->g[2] += dshape_df * dfdx[i] * params->ss_ref_x[i];
            params->g[3] += dshape_df * dfdx[i] * params->ss_ref_y[i];
            params->g[4] += dshape_df * dfdy[i] * params->ss_ref_x[i];
            params->g[5] += dshape_df * dfdy[i] * params->ss_ref_y[i];

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = params->dfdp[row];
                for (int col = row; col < 6; col++) {
                    params->H[row * 6 + col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * params->dfdp[col];
                }
            }
        }

        populate_hessian_lower_tri(params->H, params->lambda);
        invertMatrix(params->H, params->invH);
        update_shapefunc_parameters(params->pdp, params->p, params->dp, params->invH, params->g);


        //std::cout << df_dx << " " << df_dy << std::endl;
        // std::cout << "dfdp  " << dfdp[0] << " " << dfdp[1] << " " << dfdp[2] << " " << dfdp[3] << " " << dfdp[4] << " " << dfdp[5] << std::endl;
        // std::cout << "g  " << g[0] << " " << g[1] << " " << g[2] << " " << g[3] << " " << g[4] << " " << g[5] << std::endl;
        // std::cout << "H  " << H[0][0] << " " << H[1][0] << " " << H[2][0] << " " << H[3][0] << " " << H[4][0] << " " << H[5][0] << std::endl;
        // std::cout << "Hi " << invH[0][0] << " " << invH[1][0] << " " << invH[2][0] << " " << invH[3][0] << " " << invH[4][0] << " " << invH[5][0] << std::endl;
        // std::cout << "p  " << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[5] << std::endl;
        // std::cout << "pd " << pdp[0] << " " << pdp[1] << " " << pdp[2] << " " << pdp[3] << " " << pdp[4] << " " << pdp[5] << std::endl;


        // calculate cost function for current parameter values
        params->costp = 0.0;
        for (int i = 0; i < n; i++){
            double def_norm = ss_def[i] * inv_sum_squared_def;
            double ref_norm = params->ss_ref[i] * inv_sum_squared_ref;
            params->costp += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        params->costpdp = 0.0;
        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->pdp);
            params->ss_ref[i] = interpolator::eval_bicubic(params->ss_ref_x[i], params->ss_ref_y[i]);
            sum_squared_ref += params->ss_ref[i] * params->ss_ref[i];
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (int i = 0; i < n; ++i) {
            double def_norm = ss_def[i] * inv_sum_squared_def;
            double ref_norm = params->ss_ref[i] * inv_sum_squared_ref;
            params->costpdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }


    void znssd(std::vector<double> &ss_def,
             std::vector<double> &ss_def_x, 
             std::vector<double> &ss_def_y, 
             int n,
             Parameters *params){


        // reset derivative and hessian values
        std::fill(params->g.begin(), params->g.end(), 0.0);
        std::fill(params->H.begin(), params->H.end(), 0.0);

        std::vector<double> dfdx(n);
        std::vector<double> dfdy(n);

        double mean_ref = 0.0;
        double mean_def = 0.0;
        
        // interpolation data struct
        interpolator::Data interp_data;

        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < n; ++i) {

            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->p);

            interp_data = interpolator::eval_bicubic_and_derivs(params->ss_ref_x[i], params->ss_ref_y[i]);
            params->ss_ref[i] = interp_data.interp_value;
            dfdx[i] = interp_data.interp_dx;
            dfdy[i] = interp_data.interp_dy;

            mean_ref += params->ss_ref[i];
            mean_def += ss_def[i];

        }

        mean_def /= n;
        mean_ref /= n;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_def += (ss_def[i] - mean_def) * (ss_def[i] - mean_def);
            sum_squared_ref += (params->ss_ref[i] - mean_ref) * (params->ss_ref[i] - mean_ref);
        }

        double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        // loop over the subset values
        for (int i = 0; i < n; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(params->dfdp, params->ss_ref_x[i], params->ss_ref_y[i], dfdx[i], dfdy[i], n);

            double dshape_df = - inv_sum_squared_ref * ((ss_def[i] - mean_def) * inv_sum_squared_def - (params->ss_ref[i] - mean_ref) * inv_sum_squared_ref);
            params->g[0] += dshape_df * dfdx[i];
            params->g[1] += dshape_df * dfdy[i];
            params->g[2] += dshape_df * dfdx[i] * params->ss_ref_x[i];
            params->g[3] += dshape_df * dfdx[i] * params->ss_ref_y[i];
            params->g[4] += dshape_df * dfdy[i] * params->ss_ref_x[i];
            params->g[5] += dshape_df * dfdy[i] * params->ss_ref_y[i];

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < 6; row++) {
                double dfdp_row = params->dfdp[row];
                for (int col = row; col < 6; col++) {
                    params->H[row * 6 + col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * params->dfdp[col];
                }
            }
        }



        populate_hessian_lower_tri(params->H, params->lambda);
        invertMatrix(params->H, params->invH);
        update_shapefunc_parameters(params->pdp, params->p, params->dp, params->invH, params->g);

        // #pragma omp critical
        // {
        //     if (ss_def[0] == 17){
        //         std::cout << "invs " << inv_sum_squared_def << " " << inv_sum_squared_ref << std::endl;
        //         std::cout << "dfdx " << dfdx[0] << " " << dfdy[0] << std::endl;
        //         std::cout << "dfdp " << params->dfdp[0] << " " << params->dfdp[1] << " " << params->dfdp[2] << " " << params->dfdp[3] << " " <<params->dfdp[4] << " " << params->dfdp[5] << std::endl;
        //         std::cout << "g   " << params->g[0] << " " << params->g[1] << " " << params->g[2] << " " << params->g[3] << " " << params->g[4] << " " << params->g[5] << std::endl;
        //         std::cout << "H   " << params->H[0] << " " << params->H[1] << " " << params->H[2] << " " << params->H[3] << " " << params->H[4] << " " << params->H[5] << std::endl;
        //         std::cout << "Hi  " << params->invH[0] << " " << params->invH[1] << " " << params->invH[2] << " " << params->invH[3] << " " << params->invH[4] << " " << params->invH[5] << std::endl;
        //         std::cout << "p   " << params->p[0] << " " << params->p[1] << " " << params->p[2] << " " << params->p[3] << " " << params->p[4] << " " << params->p[5] << std::endl;
        //         std::cout << "pdp " << params->pdp[0] << " " << params->pdp[1] << " " << params->pdp[2] << " " << params->pdp[3] << " " << params->pdp[4] << " " << params->pdp[5] << std::endl;
        //     }
        // }

        // calculate cost function for current parameter values
        params->costp = 0.0;
        for (int i = 0; i < n; i++){
            double def_norm = (ss_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (params->ss_ref[i] - mean_ref) * inv_sum_squared_ref;
            params->costp += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        mean_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            shape_function(params->ss_ref_x[i], params->ss_ref_y[i], ss_def_x[i], ss_def_y[i], params->pdp);
            params->ss_ref[i] = interpolator::eval_bicubic(params->ss_ref_x[i], params->ss_ref_y[i]);
            mean_ref += params->ss_ref[i];
        }

        mean_ref /= n;

        sum_squared_ref = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_squared_ref += (params->ss_ref[i] - mean_ref) * (params->ss_ref[i] - mean_ref);
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        params->costpdp = 0.0;
        for (int i = 0; i < n; ++i) {
            double def_norm = (ss_def[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (params->ss_ref[i] - mean_ref) * inv_sum_squared_ref;
            params->costpdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }

    // Inv matrix using Gauss Elim.
    bool invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse) {
            int n = 6;

        // Augmented matrix stored as 1D vector
        std::vector<double> augmented(n * 2 * n, 0.0);

        // Initialize augmented matrix with input matrix and identity matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                augmented[i * (2 * n) + j] = matrix[i * n + j];
                augmented[i * (2 * n) + (j + n)] = (i == j) ? 1.0 : 0.0;
            }
        }

        // Gauss-Jordan Elimination
        for (int i = 0; i < n; ++i) {
            // Search for max element in column
            double maxEl = std::abs(augmented[i * (2 * n) + i]);
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(augmented[k * (2 * n) + i]) > maxEl) {
                    maxEl = std::abs(augmented[k * (2 * n) + i]);
                    maxRow = k;
                }
            }

            // Swap maximum row with current row
            if (i != maxRow) {
                for (int j = 0; j < 2 * n; ++j) {
                    std::swap(augmented[i * (2 * n) + j], augmented[maxRow * (2 * n) + j]);
                }
            }

            // Make the pivot element 1
            double pivot = augmented[i * (2 * n) + i];
            if (pivot == 0) {
                return false; // Singular matrix, can't invert
            }
            for (int j = 0; j < 2 * n; ++j) {
                augmented[i * (2 * n) + j] /= pivot;
            }

            // Make the elements below the pivot 0
            for (int k = i + 1; k < n; ++k) {
                double factor = augmented[k * (2 * n) + i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k * (2 * n) + j] -= augmented[i * (2 * n) + j] * factor;
                }
            }
        }

        // Perform back substitution to eliminate entries above the pivot
        for (int i = n - 1; i >= 0; --i) {
            for (int k = i - 1; k >= 0; --k) {
                double factor = augmented[k * (2 * n) + i];
                for (int j = 0; j < 2 * n; ++j) {
                    augmented[k * (2 * n) + j] -= augmented[i * (2 * n) + j] * factor;
                }
            }
        }

        // Extract the inverse matrix from the augmented matrix
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inverse[i * n + j] = augmented[i * (2 * n) + (j + n)];
            }
        }

        return true;
    }

    void populate_hessian_lower_tri(std::vector<double> &H, double lambda){
        for (int row = 0; row < 6; row++) {
            for (int col = row + 1; col < 6; col++) {
                H[col * 6 + row] = H[row * 6 + col];
            }
            H[row * 6 + row] += lambda * H[row * 6 + row]; // diagonal
        }
    }
    
    void update_lambda(double costp, double costpdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda){

        if (costp < costpdp){
            lambda *= 10.0;
        }
        else{
            lambda *= 0.1;
            for (int i = 0; i < 6; i++){
                p[i] = pdp[i];
            }
        }
        
    }

    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &g){

        // multiply inverse with gradient
        for (int i = 0; i < 6; ++i) {
            dp[i] = 0.0;
            for (int j = 0; j < 6; ++j) {
                dp[i] +=  1.0 * invH[i*6 + j] * g[j];
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

    void affine_parameters_to_displacement(Results *results, double ss_x, double ss_y, std::vector<double> &p){
        results->u = ss_x - (p[0] + (1 + p[2]) * ss_x + p[3] * ss_y);
        results->v = ss_y - (p[1] + (1 + p[5]) * ss_y + p[4] * ss_x);
        results->mag = std::sqrt(results->u*results->u + results->v*results->v);
    }



    void setCostFunction(const std::string& corr_crit) {
        if (corr_crit == "SSD") optimize_cost = ssd;
        else if (corr_crit == "NSSD") optimize_cost = nssd;
        else if (corr_crit == "ZNSSD") optimize_cost = znssd;
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

    void init_parameters(Parameters *params, int ss_size){
        params->g.resize(6,0.0); 
        params->dfdp.resize(6,0.0);
        params->H.resize(36,0.0);
        params->invH.resize(36,0.0); 
        params->ss_ref.resize(ss_size*ss_size,0.0); 
        params->ss_ref_x.resize(ss_size*ss_size,0.0);
        params->ss_ref_y.resize(ss_size*ss_size,0.0); 
        params->p.resize(6,0.0); 
        params->dp.resize(6,0.0); 
        params->pdp.resize(6,0.0);
        params->iter = 0;
        params->lambda = 0.01;
        params->ftol = 0.0;
        params->xtol = 0.0;
    }


}