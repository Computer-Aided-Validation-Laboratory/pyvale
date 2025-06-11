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
#include "./dicutil.hpp"
#include "./defines.hpp"


namespace optimizer {




    // function pointer for correlation criteria
    void (*optimize_cost)(util::Subset &ss_def, util::Subset &ss_ref,optimizer::Parameters &opt, const Interpolator &Interp);
    void (*shape_function)(double &, double &, double , double , std::vector<double> &);
    void (*dshape_dp)(std::vector<double>&, double, double, double, double, int);
    void (*params_to_displacement)(util::Results &results, double ss_x, double ss_y, std::vector<double> &p);






    void init(std::string &corr_crit, std::string &shape_func){
        setCostFunction(corr_crit);
        setShapeFunction(shape_func);
    }



    util::Results solve(const double ss_x, const double ss_y, util::Subset &ss_def, util::Subset &ss_ref, const Interpolator &interp_ref, optimizer::Parameters &opt) {

        int iter = 0;
        double ftol = 0;
        double xtol = 0;
        opt.lambda = 0.001;

        while (iter < opt.max_iter) {

            // perform the optimization
            optimize_cost(ss_def, ss_ref, opt, interp_ref);
            update_lambda(opt.costp, opt.costpdp, opt.p, opt.pdp, opt.lambda, opt.num_params);



            // relative change of all parameters
            xtol = std::sqrt(std::inner_product(opt.dp.begin(), opt.dp.end(), opt.dp.begin(), 0.0)) / 
                          std::sqrt(std::inner_product( opt.p.begin(),  opt.p.end(),  opt.p.begin(), 0.0));



            // variation on correlation coefficient
            ftol = std::abs(opt.costpdp - opt.costp);

            // convergence criteria
            // - rel change in parameters is less than user precision
            // - change in corr coeff is less than precision
            // - corr is less than threshold
            if ((xtol < opt.precision) && (ftol < opt.precision) && (opt.costp < opt.threshold_lm)) {
                //debugPrint(ss_x, ss_y, iter, opt.costp, ftol, xtol, opt.p);
                break;
            }


            iter++;
        }

        util::Results res;
        params_to_displacement(res, ss_x, ss_y, opt.p);
        res.iter = iter;
        res.ftol = ftol;
        res.xtol = xtol;
        res.p = opt.p;
        res.cost = opt.costp;

        // if its a bad subset and hits max_iterations then reset p values to 0.0 to prevent bad seeding.
        if (iter == opt.max_iter) {
            //debugPrint(ss_x, ss_y, iter, opt.costp, ftol, xtol, opt.p);
            std::fill(opt.p.begin(), opt.p.end(), 0.0);
        }

        return res;
    }




    void ssd(util::Subset &ss_def,
             util::Subset &ss_ref,
             optimizer::Parameters &opt,
             const Interpolator &interp_ref){

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;
        double dfdx;
        double dfdy;

        // interpolation data struct
        InterpVals interp_vals;

        double gtemp[6] = {0,0,0,0,0,0};


        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        // loop over the subset values
        for (int i = 0; i < num_px; i++){

            // get subset coordinates based on shape function parameters
            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.p);

            double ref_x = ss_ref.x[i];
            double ref_y = ss_ref.y[i];

            // // Inside ssd function before interpolation
            // if (ref_x < 0 || ref_x > opt.px_hori-1 || ref_y < 0 || ref_y > opt.px_vert-1) {
            //     // Apply a penalty proportional to how far outside the bounds
            //     double penalty = 0;
            //     if (ref_x < 0) penalty += std::abs(ref_x);
            //     if (ref_x > opt.px_hori-1) penalty += ref_x - (opt.px_hori-1);
            //     if (ref_y < 0) penalty += std::abs(ref_y);
            //     if (ref_y > opt.px_vert-1) penalty += ref_y - (opt.px_vert-1);
                
            //     // Add this penalty to the cost function
            //     opt.costp += penalty * penalty;
                
            //     // Skip this point for derivative calculations
            //     continue;
            // }

            // get the subset value and derivitives
            interp_vals = interp_ref.eval_bicubic_and_derivs(ref_x, ref_y);
            ss_ref.vals[i] = interp_vals.f;
            double ref = ss_ref.vals[i];

            dfdx = interp_vals.dfdx;
            dfdy = interp_vals.dfdy;

            // derivative of shape function with repsect to parameters
            dshape_dp(opt.dfdp, ref_x, ref_y, dfdx, dfdy, num_px);
            
            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += dfdp_row * opt.dfdp[col];
                }
            }

            double dshape_df = - (ss_def.vals[i] - ref);
            
            gtemp[0] = dshape_df * dfdx;
            gtemp[1] = dshape_df * dfdy;
            gtemp[2] = dshape_df * dfdx * ref_x;
            gtemp[3] = dshape_df * dfdx * ref_y;
            gtemp[4] = dshape_df * dfdy * ref_x;
            gtemp[5] = dshape_df * dfdy * ref_y;

            for (int i = 0; i < num_params; ++i) {
                opt.g[i] += gtemp[i];
            }

        }

        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);

        // calculate cost function for current and updated parameter values 
        opt.costp = 0.0;
        for (int i = 0; i < num_px; i++){
            opt.costp += (ss_def.vals[i] - ss_ref.vals[i]) * (ss_def.vals[i] - ss_ref.vals[i]);
        }


        // calculate cost function for updated parameter values
        opt.costpdp = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.pdp);
            ss_ref.vals[i] = interp_ref.eval_bicubic(ss_ref.x[i], ss_ref.y[i]);
            opt.costpdp += (ss_def.vals[i] - ss_ref.vals[i]) * (ss_def.vals[i] - ss_ref.vals[i]);
        }
    }


    void nssd(util::Subset &ss_def,
              util::Subset &ss_ref,
              optimizer::Parameters &opt,
              const Interpolator &interp_ref){

        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;

        double gtemp[6] = {0,0,0,0,0,0};

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

            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.p);
            
            // // Inside ssd function before interpolation
            // if (ss_ref.x[i] < 0 || ss_ref.x[i] > opt.px_hori-1 || ss_ref.y[i] < 0 || ss_ref.y[i] > opt.px_vert-1) {
            //     // Apply a penalty proportional to how far outside the bounds
            //     double penalty = 0;
            //     if (ss_ref.x[i] < 0) penalty += std::abs(ss_ref.x[i]);
            //     if (ss_ref.x[i] > opt.px_hori-1) penalty += ss_ref.x[i] - (opt.px_hori-1);
            //     if (ss_ref.y[i] < 0) penalty += std::abs(ss_ref.y[i]);
            //     if (ss_ref.y[i] > opt.px_vert-1) penalty += ss_ref.y[i] - (opt.px_vert-1);
                
            //     // Add this penalty to the cost function
            //     opt.costp += 10000.0 * penalty * penalty;
                
            //     // Skip this point for derivative calculations
            //     continue;
            // }

            interp_vals = interp_ref.eval_bicubic_and_derivs(ss_ref.x[i], ss_ref.y[i]);
            ss_ref.vals[i] = interp_vals.f;
            dfdx[i] = interp_vals.dfdx;
            dfdy[i] = interp_vals.dfdy;
            sum_squared_def += ss_def.vals[i] * ss_def.vals[i];
            sum_squared_ref += ss_ref.vals[i] * ss_ref.vals[i];
        }

        inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        // loop over the subset values
        for (int i = 0; i < num_px; i++){
            
            // derivative of shape function with repsect to parameters
            dshape_dp(opt.dfdp, ss_ref.x[i], ss_ref.y[i], dfdx[i], dfdy[i], num_px);

            double dshape_df = - inv_sum_squared_ref * (ss_def.vals[i] * inv_sum_squared_def - ss_ref.vals[i] * inv_sum_squared_ref);
            
            
            gtemp[0] = dshape_df * dfdx[i];
            gtemp[1] = dshape_df * dfdy[i];
            gtemp[2] = dshape_df * dfdx[i] * ss_ref.x[i];
            gtemp[3] = dshape_df * dfdx[i] * ss_ref.y[i];
            gtemp[4] = dshape_df * dfdy[i] * ss_ref.x[i];
            gtemp[5] = dshape_df * dfdy[i] * ss_ref.y[i];

            for (int i = 0; i < num_params; ++i) {
                opt.g[i] += gtemp[i];
            }

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * opt.dfdp[col];
                }
            }
        }

        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);


        // calculate cost function for current parameter values
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            opt.costp += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        sum_squared_ref = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.pdp);
            ss_ref.vals[i] = interp_ref.eval_bicubic(ss_ref.x[i], ss_ref.y[i]);
            sum_squared_ref += ss_ref.vals[i] * ss_ref.vals[i];
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (int i = 0; i < num_px; ++i) {
            double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            opt.costpdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

    }


    void znssd(util::Subset &ss_def,
               util::Subset &ss_ref,
               optimizer::Parameters &opt,
               const Interpolator &interp_ref){


        // reset derivative and hessian values
        std::fill(opt.g.begin(), opt.g.end(), 0.0);
        std::fill(opt.H.begin(), opt.H.end(), 0.0);

        const int num_px = ss_def.num_px;
        const int num_params = opt.num_params;

        double gtemp[6] = {0,0,0,0,0,0};


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
        // std::cout << ss_x << " " << ss_y << " ";

        for (int i = 0; i < num_px; ++i) {

            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.p);

            // // Inside ssd function before interpolation
            // if (ss_ref.x[i] < 0 || ss_ref.x[i] > opt.px_hori-1 || ss_ref.y[i] < 0 || ss_ref.y[i] > opt.px_vert-1) {
            //     // Apply a penalty proportional to how far outside the bounds
            //     double penalty = 0;
            //     if (ss_ref.x[i] < 0) penalty += std::abs(ss_ref.x[i]);
            //     if (ss_ref.x[i] > opt.px_hori-1) penalty += ss_ref.x[i] - (opt.px_hori-1);
            //     if (ss_ref.y[i] < 0) penalty += std::abs(ss_ref.y[i]);
            //     if (ss_ref.y[i] > opt.px_vert-1) penalty += ss_ref.y[i] - (opt.px_vert-1);

            //     // Add this penalty to the cost function
            //     opt.costp += 10000.0 * penalty * penalty;

            //     // Skip this point for derivative calculations
            //     continue;
            // }


            interp_vals = interp_ref.eval_bicubic_and_derivs(ss_ref.x[i], ss_ref.y[i]);
            ss_ref.vals[i] = interp_vals.f;
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
            
            // derivative of shape function with repsect to parameters
            dshape_dp(opt.dfdp, ss_ref.x[i], ss_ref.y[i], dfdx[i], dfdy[i], num_px);

            double dshape_df = - inv_sum_squared_ref * ((ss_def.vals[i] - mean_def) * inv_sum_squared_def - (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref);

            gtemp[0] = dshape_df * dfdx[i];
            gtemp[1] = dshape_df * dfdy[i];
            gtemp[2] = dshape_df * dfdx[i] * ss_ref.x[i];
            gtemp[3] = dshape_df * dfdx[i] * ss_ref.y[i];
            gtemp[4] = dshape_df * dfdy[i] * ss_ref.x[i];
            gtemp[5] = dshape_df * dfdy[i] * ss_ref.y[i];


            for (int i = 0; i < num_params; ++i) {
                opt.g[i] += gtemp[i];
            }

            // Upper triangle of Hessian Matrix
            for (int row = 0; row < num_params; row++) {
                double dfdp_row = opt.dfdp[row];
                for (int col = row; col < num_params; col++) {
                    opt.H[row * num_params + col] += inv_sum_squared_ref * inv_sum_squared_ref * dfdp_row * opt.dfdp[col];
                }
            }
        }


        populate_hessian_lower_tri(opt.H, opt.lambda, opt.num_params);
        invertMatrix(opt.H, opt.invH, opt.augmented, opt.num_params);
        update_shapefunc_parameters(opt.pdp, opt.p, opt.dp, opt.invH, opt.g, opt.num_params);

        //#pragma omp critical
        //{
            // if (ss_def.vals[0] == 17){
                // std::cout << "invs " << inv_sum_squared_def << " " << inv_sum_squared_ref << std::endl;
                // std::cout << "dfdx " << dfdx[0] << " " << dfdy[0] << std::endl;
                // std::cout << "dfdp " << opt.dfdp[0] << " " << opt.dfdp[1] << " " << opt.dfdp[2] << " " << opt.dfdp[3] << " " <<opt.dfdp[4] << " " << opt.dfdp[5] << std::endl;
                // std::cout << "g   " << opt.g[0] << " " << opt.g[1] << " " << opt.g[2] << " " << opt.g[3] << " " << opt.g[4] << " " << opt.g[5] << std::endl;
                // std::cout << "H   " << opt.H[0] << " " << opt.H[1] << " " << opt.H[2] << " " << opt.H[3] << " " << opt.H[4] << " " << opt.H[5] << std::endl;
                // std::cout << "Hi  " << opt.invH[0] << " " << opt.invH[1] << " " << opt.invH[2] << " " << opt.invH[3] << " " << opt.invH[4] << " " << opt.invH[5] << std::endl;
                // std::cout << "p   " << opt.p[0] << " " << opt.p[1] << " " << opt.p[2] << " " << opt.p[3] << " " << opt.p[4] << " " << opt.p[5] << std::endl;
                // std::cout << "pdp " << opt.pdp[0] << " " << opt.pdp[1] << " " << opt.pdp[2] << " " << opt.pdp[3] << " " << opt.pdp[4] << " " << opt.pdp[5] << std::endl;
                // std::cout << opt.g.size() << std::endl;
                // std::cout << "invs " << inv_sum_squared_def << " " << inv_sum_squared_ref << std::endl;
                // std::cout << "dfdx " << dfdx[0] << " " << dfdy[0] << std::endl;
                // std::cout << "dfdp " << opt.dfdp[0] << " " << opt.dfdp[1] << std::endl;
                // std::cout << "g   " << opt.g[0] << " " << opt.g[1] << std::endl;
                // std::cout << "H   " << opt.H[0] << " " << opt.H[1] << " " << opt.H[2] << " " << opt.H[3] << std::endl;
                // std::cout << "Hi  " << opt.invH[0] << " " << opt.invH[1] << " " << opt.invH[2] << " " << opt.invH[3] << std::endl;
                // std::cout << "p   " << opt.p[0] << " " << opt.p[1] << std::endl;
                // std::cout << "pdp " << opt.pdp[0] << " " << opt.pdp[1] <<  std::endl;
                // exit(0);
            // }
        //}

        // calculate cost function for current parameter values
        for (int i = 0; i < num_px; i++){
            double def_norm = (ss_def.vals[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref;
            opt.costp += (def_norm - ref_norm) * (def_norm - ref_norm);
        }


        // calculate cost function for updated parameter values
        mean_ref = 0.0;
        for (int i = 0; i < num_px; ++i) {
            shape_function(ss_ref.x[i], ss_ref.y[i], ss_def.x[i], ss_def.y[i], opt.pdp);
            ss_ref.vals[i] = interp_ref.eval_bicubic(ss_ref.x[i], ss_ref.y[i]);
            mean_ref += ss_ref.vals[i];
        }

        mean_ref /= num_px;

        sum_squared_ref = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_ref += (ss_ref.vals[i] - mean_ref) * (ss_ref.vals[i] - mean_ref);
        }

        inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        for (int i = 0; i < num_px; ++i) {
            double def_norm = (ss_def.vals[i] - mean_def) * inv_sum_squared_def;
            double ref_norm = (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref;
            opt.costpdp += (def_norm - ref_norm) * (def_norm - ref_norm);        
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



    inline void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p){
        x_new = p[0] + (1 + p[2]) * x + p[3] * y;
        y_new = p[1] + (1 + p[5]) * y + p[4] * x;
    }

    inline void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p){
        x_new = p[0] + x;
        y_new = p[1] + y;
    }

    inline void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p){

    }

    inline void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n){

        dfdp[0] = dfdx;
        dfdp[1] = dfdy;
        dfdp[2] = dfdx * x;
        dfdp[3] = dfdx * y;
        dfdp[4] = dfdy * x;
        dfdp[5] = dfdy * y;


    }

    inline void drigid_dp(std::vector<double> &dfdp, double x, double y,  double dfdx, double dfdy, int n){

            dfdp[0] = dfdx;
            dfdp[1] = dfdy;
    }

    inline void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p){

    }

    inline void affine_parameters_to_displacement(util::Results &res, double ss_x, double ss_y, std::vector<double> &p){
        res.u = ss_x - (p[0] + (1 + p[2]) * ss_x + p[3] * ss_y);
        res.v = ss_y - (p[1] + (1 + p[5]) * ss_y + p[4] * ss_x);
        res.mag = std::sqrt(res.u*res.u + res.v*res.v);
    }

    inline void rigid_parameters_to_displacement(util::Results &res, double ss_x, double ss_y, std::vector<double> &p){
        res.u = -p[0];
        res.v = -p[1];
        res.mag = std::sqrt(res.u*res.u + res.v*res.v);
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
        if (shape_func == "RIGID") {
            shape_function = rigid;
            dshape_dp = drigid_dp;
            params_to_displacement = rigid_parameters_to_displacement;
        } else if (shape_func == "AFFINE") {
            shape_function = affine;
            dshape_dp = daffine_dp;
            params_to_displacement = affine_parameters_to_displacement;
        } else {
            std::cerr << "Unexpected Shape Function: '" << shape_func << "'" << std::endl;
            std::cerr << "Allowed Values: 'RIGID', 'AFFINE'." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void debugPrint(int ss_x, int ss_y, int iter, double costp, double ftol, double xtol, const std::vector<double>& p) {
        #pragma omp critical
        {
            std::cout << omp_get_thread_num() << " ";
            std::cout << ss_x << " " << ss_y << " ";
            std::cout << iter << " " << costp << " " << ftol << " " << xtol << " ";
            for (size_t i = 0; i < p.size(); ++i) {
                std::cout << p[i] << " ";
            }

        }
    }


}
