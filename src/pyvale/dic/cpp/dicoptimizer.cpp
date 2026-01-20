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
#include <array>
#include <omp.h>



// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicoptimizer.hpp"
#include "./dicshapefunc.hpp"
#include "./dicresults.hpp"


namespace optimizer {




    // function pointer for correlation criteria
    void (*optimize_cost)(const subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &Interp, optimizer::Parameters &opt, const int global_x, const int global_y);


    OptResult solve(const double ss_x, const double ss_y, subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &interp_def, optimizer::Parameters &opt, const std::string &corr_crit){

        int iter = 0;
        double ftol = 0;
        double xtol = 0;
        opt.lambda = 0.001;
        uint8_t converged = false;
        const double eps = 1e-10;


        // trying relative instead of global coordinates for the optimization
        int global_x = ss_ref.x[0];
        int global_y = ss_ref.y[0];
        for (int px = 0; px < ss_ref.num_px; px++){
            ss_ref.x[px] -= global_x;
            ss_ref.y[px] -= global_y;
        }

        while (iter < opt.max_iter) {

            // perform the optimization
            optimize_cost(ss_ref, ss_def, interp_def, opt, global_x, global_y);

            // set new damping value
            update_lambda(opt.costp, opt.costpdp, opt.p, opt.pdp, opt.lambda, opt.num_params);

            // relative change of all parameters
            const double dp_norm = std::sqrt(std::inner_product(opt.dp.begin(), opt.dp.end(), opt.dp.begin(), 0.0));
            const double p_norm  = std::sqrt(std::inner_product( opt.p.begin(),  opt.p.end(),  opt.p.begin(), 0.0));
            xtol = dp_norm / (p_norm+eps);

            // variation on correlation coefficient
            ftol = std::abs(opt.costpdp - opt.costp) / (std::abs(opt.costp) + eps);


            // Check converged
            if ((xtol < opt.precision) && (ftol < opt.precision)) {
                //debug_print(ss_x, ss_y, iter, opt.costp, ftol, xtol, opt.p);
                converged=true;
                break;
            }
            iter++;
        }


        // calculate zncc value
        double mean_def = 0.0;
        double mean_ref = 0.0;

        for (int i = 0; i < ss_def.num_px; ++i) {
            mean_ref += ss_ref.vals[i];
            mean_def += ss_def.vals[i];
        }

        mean_ref /= ss_ref.num_px;
        mean_def /= ss_def.num_px;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < ss_def.num_px; ++i) {
            sum_squared_ref += (ss_ref.vals[i] - mean_ref) * (ss_ref.vals[i] - mean_ref);
            sum_squared_def += (ss_def.vals[i] - mean_def) * (ss_def.vals[i] - mean_def);
        }

        const double inv_sum_squared = 1.0 / sqrt(sum_squared_ref*sum_squared_def);

        double zncc = 0.0;
        for (int i = 0; i < ss_def.num_px; ++i) {
            const double def_norm = (ss_def.vals[i] - mean_def);
            const double ref_norm = (ss_ref.vals[i] - mean_ref);
            zncc += ref_norm*def_norm; 
        }
        zncc *= inv_sum_squared;


        OptResult res(opt.num_params);
        shapefunc::get_displacement(res, ss_x-global_x, ss_y-global_y, opt.p);
        res.iter = iter;
        res.ftol = ftol;
        res.xtol = xtol;
        res.p = opt.p;
        res.cost = zncc;
        res.converged = converged;
        if (zncc >= opt.threshold) res.above_threshold = true;

        // debugging
        //if (iter == opt.max_iter) {
        //  debug_print(ss_x, ss_y, iter, opt.costp, ftol, xtol, opt.p);
        //}

        return res;
    }




    void ssd(const subset::Pixels &ss_ref,
             subset::Pixels &ss_def,
             const Interpolator &interp_def,
             optimizer::Parameters &opt,
             const int global_x,
             const int global_y){

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;
        double dfdx;
        double dfdy;

        // interpolation data struct
        InterpVals interp_vals;

        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        // loop over the subset values
        for (int i = 0; i < num_px; i++){

            // apply shape function parameters to deformed subset
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.p);

            // x and y coordinates of reference subset
            double def_x = ss_def.x[i];
            double def_y = ss_def.y[i];

            // get the subset value and derivitives
            interp_vals = interp_def.eval_bicubic_and_derivs(global_x, global_y, def_x+global_x, def_y+global_y);
            ss_def.vals[i] = interp_vals.f;
            double def = ss_def.vals[i];

            dfdx = interp_vals.dfdx;
            dfdy = interp_vals.dfdy;

            // derivative of shape function with repsect to parameters
            shapefunc::get_dfdp(opt.dfdp, def_x, def_y, dfdx, dfdy);

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += dfdp_row * opt.dfdp[col];
                }
            }

            const double dCost_df = - (ss_ref.vals[i] - def);

            for (int j = 0; j < num_params; j++) {
                opt.g[j] += dCost_df * opt.dfdp[j];
            }

        }

        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);

        // calculate cost function for current and updated parameter values 
        opt.costp = 0.0;
        for (int i = 0; i < num_px; i++){
            opt.costp += (ss_ref.vals[i] - ss_def.vals[i]) * (ss_ref.vals[i] - ss_def.vals[i]);
        }


        // calculate cost function for updated parameter values
        opt.costpdp = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.pdp);
            ss_def.vals[i] = interp_def.eval_bicubic(global_x, global_y, ss_def.x[i]+global_x, ss_def.y[i]+global_y);
            opt.costpdp += (ss_ref.vals[i] - ss_def.vals[i]) * (ss_ref.vals[i] - ss_def.vals[i]);
        }
    }


    void nssd(const subset::Pixels &ss_ref,
              subset::Pixels &ss_def,
              const Interpolator &interp_def,
              optimizer::Parameters &opt,
              const int global_x,
              const int global_y){

        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;

        std::vector<double> dfdx(num_px);
        std::vector<double> dfdy(num_px);

        double sum_squared_def = 0.0;
        double sum_squared_ref = 0.0; 
        double inv_sum_squared_def;
        double inv_sum_squared_ref;

        // interpolation data struct
        InterpVals interp_vals;

        // reset cost function
        opt.costp = 0.0;
        opt.costpdp = 0.0;

        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < num_px; ++i) {

            // apply shape function parameters to deformed subset
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.p);

            interp_vals = interp_def.eval_bicubic_and_derivs(global_x, global_y, ss_def.x[i]+global_x, ss_def.y[i]+global_y);
            ss_def.vals[i] = interp_vals.f;
            dfdx[i] = interp_vals.dfdx;
            dfdy[i] = interp_vals.dfdy;
            sum_squared_def += ss_def.vals[i] * ss_def.vals[i];
            sum_squared_ref += ss_ref.vals[i] * ss_ref.vals[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        // loop over the subset values
        for (int i = 0; i < num_px; i++){

            const double def_x_i = ss_def.x[i];
            const double def_y_i = ss_def.y[i];
            const double dfdx_i = dfdx[i];
            const double dfdy_i = dfdy[i];

            // derivative of shape function with repsect to parameters
            shapefunc::get_dfdp(opt.dfdp, def_x_i, def_y_i, dfdx_i, dfdy_i);

            const double dCostdf = - inv_sum_squared_def * (ss_ref.vals[i] * inv_sum_squared_ref - ss_def.vals[i] * inv_sum_squared_def);

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                opt.g[row] += dCostdf * opt.dfdp[row];
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += inv_sum_squared_def * inv_sum_squared_def * dfdp_row * opt.dfdp[col];
                }
            }
        }

        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);


        // calculate cost function for current parameter values
        for (int i = 0; i < num_px; i++){
            const double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            const double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            opt.costp += (ref_norm - def_norm) * (ref_norm - def_norm);
        }


        // calculate cost function for updated parameter values
        sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.pdp);
            ss_def.vals[i] = interp_def.eval_bicubic(global_x, global_y, ss_def.x[i]+global_x, ss_def.y[i]+global_y);
            sum_squared_def += ss_def.vals[i] * ss_def.vals[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);

        for (int i = 0; i < num_px; ++i) {
            const double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            const double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            opt.costpdp += (ref_norm - def_norm) * (ref_norm - def_norm);
        }

    }


    void znssd(const subset::Pixels &ss_ref,
               subset::Pixels &ss_def,
               const Interpolator &interp_def,
               optimizer::Parameters &opt,
               const int global_x,
               const int global_y){


        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;

        std::vector<double> dfdx(num_px);
        std::vector<double> dfdy(num_px);

        double mean_ref = 0.0;
        double mean_def = 0.0;

        // interpolation data struct
        InterpVals interp_vals;

        // reset cost function
        opt.costp = 0.0;
        opt.costpdp = 0.0;

        // get the normalisation values for both reference and deformed subsets
        for (int i = 0; i < num_px; ++i) {

            // apply shape function parameters to deformed subset
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.p);

            interp_vals = interp_def.eval_bicubic_and_derivs(global_x, global_y, ss_def.x[i]+global_x, ss_def.y[i]+global_y);
            ss_def.vals[i] = interp_vals.f;
            dfdx[i] = interp_vals.dfdx;
            dfdy[i] = interp_vals.dfdy;

            mean_ref += ss_ref.vals[i];
            mean_def += ss_def.vals[i];

        }

        mean_def /= num_px;
        mean_ref /= num_px;

        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_def += (ss_def.vals[i] - mean_def) * (ss_def.vals[i] - mean_def);
            sum_squared_ref += (ss_ref.vals[i] - mean_ref) * (ss_ref.vals[i] - mean_ref);
        }

        double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        // loop over the subset values
        for (int i = 0; i < num_px; i++){

            const double def_x_i = ss_def.x[i];
            const double def_y_i = ss_def.y[i];
            const double dfdx_i = dfdx[i];
            const double dfdy_i = dfdy[i];

            // derivative of shape function with repsect to parameters
            shapefunc::get_dfdp(opt.dfdp, def_x_i, def_y_i, dfdx_i, dfdy_i);

            const double dCost_df = - inv_sum_squared_def * ((ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref - (ss_def.vals[i] - mean_def) * inv_sum_squared_def);


            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                opt.g[row] += dCost_df * opt.dfdp[row];
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += inv_sum_squared_def * inv_sum_squared_def * dfdp_row * opt.dfdp[col];
                }
            }
        }


        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);

        // calculate cost function for current parameter values
        for (int i = 0; i < num_px; i++){
            const double def_norm = (ss_def.vals[i] - mean_def) * inv_sum_squared_def;
            const double ref_norm = (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref;
            opt.costp += (ref_norm - def_norm) * (ref_norm - def_norm);
        }

        // calculate cost function for updated parameter values
        mean_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shapefunc::get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], opt.pdp);
            ss_def.vals[i] = interp_def.eval_bicubic(global_x, global_y, ss_def.x[i]+global_x, ss_def.y[i]+global_y);
            mean_def += ss_def.vals[i];
        }

        mean_def /= num_px;

        sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_def += (ss_def.vals[i] - mean_def) * (ss_def.vals[i] - mean_def);
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);


        for (int i = 0; i < num_px; ++i) {
            const double def_norm = (ss_def.vals[i] - mean_def) * inv_sum_squared_def;
            const double ref_norm = (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref;
            opt.costpdp += (ref_norm - def_norm) * (ref_norm - def_norm);
        }

    }

    // Inv matrix using Gauss Elim.
    bool invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse, std::vector<double>& augmented, int num_params) {

        const int n = num_params;
        
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

    void populate_hessian_lower_tri(std::vector<double> &H, double lambda, int num_params){
        for (int row = 0; row < num_params; row++) {
            for (int col = row + 1; col < num_params; col++) {
                H[col * num_params + row] = H[row * num_params + col];
            }
            H[row * num_params + row] += lambda * H[row * num_params + row]; // diagonal
        }
    }

    void update_lambda(double costp, double costpdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda, int num_params){

        if (costp < costpdp){
            lambda *= 10.0;
        }
        else{
            lambda *= 0.1;
            for (int i = 0; i < num_params; i++){
                p[i] = pdp[i];
            }
        }
    }

    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &g, int num_params){

        // multiply inverse with gradient
        for (int i = 0; i < num_params; ++i) {
            dp[i] = 0.0;
            for (int j = 0; j < num_params; ++j) {
                dp[i] +=  1.0 * invH[i*num_params + j] * g[j];
            }
        }

        // add p to delta p
        for (int i = 0; i < num_params; ++i) {
            pdp[i] = p[i] - dp[i];
        }
    }


    void set_cost_function(const std::string& corr_crit) {
        if (corr_crit == "SSD") optimize_cost = ssd;
        else if (corr_crit == "NSSD") optimize_cost = nssd;
        else if (corr_crit == "ZNSSD") optimize_cost = znssd;
        else {
            std::cerr << "Unexpected Correlation Criteria: '" << corr_crit << "'" << std::endl;
            std::cerr << "Allowed Values: 'SSD', 'NSSD', 'ZNSSD'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }



    void debug_print(int ss_x, int ss_y, int iter, double costp, double ftol, double xtol, const std::vector<double>& p) {
        #pragma omp critical
        {
            std::cout << omp_get_thread_num() << " ";
            std::cout << ss_x << " " << ss_y << " ";
            std::cout << iter << " " << costp << " " << ftol << " " << xtol << " ";
            for (size_t i = 0; i < p.size(); ++i) {
                std::cout << p[i] << " ";
            }
            std::cout << std::endl;
        }
    }


}
