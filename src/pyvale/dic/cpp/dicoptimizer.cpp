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
#include "./dicinterp.hpp"
#include "./dicoptimizer.hpp"
#include "./dicshapefunc.hpp"
#include "./dicresults.hpp"
#include "./dicsubset.hpp"


// Constructor
Optimizer::Optimizer(const std::string& shape_func, 
                        const std::string& cost_func,
                        int max_iter_,
                        double precision_,
                        double threshold_,
                        int num_px)

    : num_params(get_num_params(shape_func)),
        lambda(0.001),
        costp(0.0),
        costpdp(0.0),
        g(num_params, 0.0),
        dfdp(num_params, 0.0),
        dfdx(num_px),
        dfdy(num_px),
        H(num_params * num_params, 0.0),
        invH(num_params * num_params, 0.0),
        p(num_params, 0.0),
        dp(num_params, 0.0),
        pdp(num_params, 0.0),
        augmented(num_params * num_params * 2, 0.0),
        max_iter(max_iter_),
        precision(precision_),
        threshold(threshold_),
        optimize_cost(nullptr),
        get_pixel(nullptr),
        get_dfdp(nullptr),
        get_displacement(nullptr) {
    
    set_shape(shape_func);
    set_cost_function(cost_func);
}

// Get number of parameters from shape function name
int Optimizer::get_num_params(const std::string& shape_name) {
    if (shape_name == "RIGID") return Rigid::num_params;
    else if (shape_name == "AFFINE") return Affine::num_params;
    else if (shape_name == "QUAD") return Quad::num_params;
    else throw std::invalid_argument("Unknown shape function: " + shape_name);
}

// Set shape function
void Optimizer::set_shape(const std::string& shape_name) {
    if (shape_name == "AFFINE") {
        get_pixel = &Affine::get_pixel;
        get_dfdp = &Affine::get_dshape_dp;
        get_displacement = &Affine::get_displacement;
    } 
    else if (shape_name == "RIGID") {
        get_pixel = &Rigid::get_pixel;
        get_dfdp = &Rigid::get_dshape_dp;
        get_displacement = &Rigid::get_displacement;
    } 
    else if (shape_name == "QUAD") {
        get_pixel = &Quad::get_pixel;
        get_dfdp = &Quad::get_dshape_dp;
        get_displacement = &Quad::get_displacement;
    } 
    else {
        throw std::invalid_argument("Unknown shape function: " + shape_name);
    }
}

void Optimizer::set_rigid_displacement(double dx, double dy) {
    std::fill(p.begin(), p.end(), 0.0);
    p[0] = dx;
    p[1] = dy;
}

OptResult Optimizer::solve(const double cx, 
                           const double cy,
                        subset::Pixels &ss_ref,
                        subset::Pixels &ss_def,
                        const Interpolator &interp_def){

    int iter = 0;
    double ftol = 0;
    double xtol = 0;
    lambda = 0.001;
    uint8_t converged = false;
    const double eps = 1e-10;


    // trying relative instead of global coordinates for the optimization
    for (int px = 0; px < ss_ref.num_px; px++){
        ss_ref.x[px] -= cx;
        ss_ref.y[px] -= cy;
    }

    while (iter < max_iter) {

        // perform the optimization
        (this->*optimize_cost)(ss_ref, ss_def, interp_def, cx, cy);

        // set new damping value
        update_lambda(costp, costpdp, p, pdp, lambda, num_params);

        // relative change of all parameters
        const double dp_norm = std::sqrt(std::inner_product(dp.begin(), dp.end(), dp.begin(), 0.0));
        const double p_norm  = std::sqrt(std::inner_product( p.begin(),  p.end(),  p.begin(), 0.0));
        xtol = dp_norm / (p_norm+eps);

        // variation on correlation coefficient
        ftol = std::abs(costpdp - costp) / (std::abs(costp) + eps);


        // Check converged
        if ((xtol < precision) && (ftol < precision)) {
            //debug_print(ss_x, ss_y, iter, costp, ftol, xtol, p);
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

    OptResult res(num_params);
    get_displacement(res.u, res.v, 0.0, 0.0, p);
    res.iter = iter;
    res.ftol = ftol;
    res.xtol = xtol;
    res.p = p;
    res.cost = zncc;
    res.converged = converged;
    if (zncc >= threshold) res.above_threshold = true;

    // debugging
    //if (iter == max_iter) {
    //  debug_print(ss_x, ss_y, iter, costp, ftol, xtol, p);
    //}

    return res;
}

void Optimizer::ssd(const subset::Pixels &ss_ref,
            subset::Pixels &ss_def,
            const Interpolator &interp_def,
            const double cx,
            const double cy){

    const int num_px = ss_def.num_px;
    
    // dont need std::vector for ssd
    double dfdx_ssd;
    double dfdy_ssd;

    // interpolation data struct
    InterpVals interp_vals;

    // reset derivative and hessian values
    std::fill(g.begin(), g.end(), 0.0);
    std::fill(H.begin(), H.end(), 0.0);

    // loop over the subset values
    for (int i = 0; i < num_px; i++){

        // apply shape function parameters to deformed subset
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], p);

        // x and y coordinates of reference subset
        double def_x = ss_def.x[i];
        double def_y = ss_def.y[i];

        // get the subset value and derivitives
        interp_vals = interp_def.eval_and_derivs(cx, cy, def_x+cx, def_y+cy);
        ss_def.vals[i] = interp_vals.f;
        double def = ss_def.vals[i];

        dfdx_ssd = interp_vals.dfdx;
        dfdy_ssd = interp_vals.dfdy;

        // derivative of shape function with repsect to parameters
        get_dfdp(dfdp, def_x, def_y, dfdx_ssd, dfdy_ssd);

        // Upper triangle of Hessian Matrix
        for (int row = 0; row < num_params; row++) {
            double dfdp_row = dfdp[row];
            for (int col = row; col < num_params; col++) {
                H[row * num_params + col] += dfdp_row * dfdp[col];
            }
        }

        const double dCost_df = - (ss_ref.vals[i] - def);

        for (int j = 0; j < num_params; j++) {
            g[j] += dCost_df * dfdp[j];
        }

    }

    populate_hessian_lower_tri(H, lambda, num_params);
    invertMatrix(H, invH, augmented, num_params);
    update_shapefunc_parameters(pdp, p, dp, invH, g, num_params);

    // calculate cost function for current and updated parameter values 
    costp = 0.0;
    for (int i = 0; i < num_px; i++){
        costp += (ss_ref.vals[i] - ss_def.vals[i]) * (ss_ref.vals[i] - ss_def.vals[i]);
    }


    // calculate cost function for updated parameter values
    costpdp = 0.0;
    for (int i = 0; i < num_px; ++i) {
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], pdp);
        ss_def.vals[i] = interp_def.eval(cx, cy, ss_def.x[i]+cx, ss_def.y[i]+cy);
        costpdp += (ss_ref.vals[i] - ss_def.vals[i]) * (ss_ref.vals[i] - ss_def.vals[i]);
    }
}


void Optimizer::nssd(const subset::Pixels &ss_ref,
                        subset::Pixels &ss_def,
                        const Interpolator &interp_def,
                        const double cx,
                        const double cy){

    // reset derivative and hessian values
    std::fill(g.begin(), g.end(), 0.0);
    std::fill(H.begin(), H.end(), 0.0);

    const int num_px = ss_def.num_px;

    double sum_squared_def = 0.0;
    double sum_squared_ref = 0.0; 
    double inv_sum_squared_def;
    double inv_sum_squared_ref;

    // interpolation data struct
    InterpVals interp_vals;

    // reset cost function
    costp = 0.0;
    costpdp = 0.0;

    // get the normalisation values for both reference and deformed subsets
    for (int i = 0; i < num_px; ++i) {

        // apply shape function parameters to deformed subset
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], p);

        interp_vals = interp_def.eval_and_derivs(cx, cy, ss_def.x[i]+cx, ss_def.y[i]+cy);
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
        get_dfdp(dfdp, def_x_i, def_y_i, dfdx_i, dfdy_i);

        const double dCostdf = - inv_sum_squared_def * (ss_ref.vals[i] * inv_sum_squared_ref - ss_def.vals[i] * inv_sum_squared_def);

        // Upper triangle of Hessian Matrix
        for (int row = 0; row < num_params; row++) {
            g[row] += dCostdf * dfdp[row];
            double dfdp_row = dfdp[row];
            for (int col = row; col < num_params; col++) {
                H[row * num_params + col] += inv_sum_squared_def * inv_sum_squared_def * dfdp_row * dfdp[col];
            }
        }
    }

    populate_hessian_lower_tri(H, lambda, num_params);
    invertMatrix(H, invH, augmented, num_params);
    update_shapefunc_parameters(pdp, p, dp, invH, g, num_params);


    // calculate cost function for current parameter values
    for (int i = 0; i < num_px; i++){
        const double def_norm = ss_def.vals[i] * inv_sum_squared_def;
        const double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
        costp += (ref_norm - def_norm) * (ref_norm - def_norm);
    }


    // calculate cost function for updated parameter values
    sum_squared_def = 0.0;
    for (int i = 0; i < num_px; ++i) {
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], pdp);
        ss_def.vals[i] = interp_def.eval(cx, cy, ss_def.x[i]+cx, ss_def.y[i]+cy);
        sum_squared_def += ss_def.vals[i] * ss_def.vals[i];
    }

    inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);

    for (int i = 0; i < num_px; ++i) {
        const double def_norm = ss_def.vals[i] * inv_sum_squared_def;
        const double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
        costpdp += (ref_norm - def_norm) * (ref_norm - def_norm);
    }

}


void Optimizer::znssd(const subset::Pixels &ss_ref,
                        subset::Pixels &ss_def,
                        const Interpolator &interp_def,
                        const double cx,
                        const double cy){


    // reset derivative and hessian values
    std::fill(g.begin(), g.end(), 0.0);
    std::fill(H.begin(), H.end(), 0.0);

    const int num_px = ss_def.num_px;

    double mean_ref = 0.0;
    double mean_def = 0.0;

    // interpolation data struct
    InterpVals interp_vals;

    // reset cost function
    costp = 0.0;
    costpdp = 0.0;

    // get the normalisation values for both reference and deformed subsets
    for (int i = 0; i < num_px; ++i) {

        // apply shape function parameters to deformed subset
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], p);

        interp_vals = interp_def.eval_and_derivs(cx, cy, ss_def.x[i]+cx, ss_def.y[i]+cy);
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
        get_dfdp(dfdp, def_x_i, def_y_i, dfdx_i, dfdy_i);

        const double dCost_df = - inv_sum_squared_def * ((ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref - (ss_def.vals[i] - mean_def) * inv_sum_squared_def);


        // Upper triangle of Hessian Matrix
        for (int row = 0; row < num_params; row++) {
            g[row] += dCost_df * dfdp[row];
            double dfdp_row = dfdp[row];
            for (int col = row; col < num_params; col++) {
                H[row * num_params + col] += inv_sum_squared_def * inv_sum_squared_def * dfdp_row * dfdp[col];
            }
        }
    }


    populate_hessian_lower_tri(H, lambda, num_params);
    invertMatrix(H, invH, augmented, num_params);
    update_shapefunc_parameters(pdp, p, dp, invH, g, num_params);

    // calculate cost function for current parameter values
    for (int i = 0; i < num_px; i++){
        const double def_norm = (ss_def.vals[i] - mean_def) * inv_sum_squared_def;
        const double ref_norm = (ss_ref.vals[i] - mean_ref) * inv_sum_squared_ref;
        costp += (ref_norm - def_norm) * (ref_norm - def_norm);
    }

    // calculate cost function for updated parameter values
    mean_def = 0.0;
    for (int i = 0; i < num_px; ++i) {
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], pdp);
        ss_def.vals[i] = interp_def.eval(cx, cy, ss_def.x[i]+cx, ss_def.y[i]+cy);
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
        costpdp += (ref_norm - def_norm) * (ref_norm - def_norm);
    }

}

// Inv matrix using Gauss Elim.
bool Optimizer::invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse, std::vector<double>& augmented, int num_params) {

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

void Optimizer::populate_hessian_lower_tri(std::vector<double> &H, double lambda, int num_params){
    for (int row = 0; row < num_params; row++) {
        for (int col = row + 1; col < num_params; col++) {
            H[col * num_params + row] = H[row * num_params + col];
        }
        H[row * num_params + row] += lambda * H[row * num_params + row]; // diagonal
    }
}

void Optimizer::update_lambda(double costp, double costpdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda, int num_params){

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

void Optimizer::update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &g, int num_params){

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


void Optimizer::set_cost_function(const std::string& corr_crit) {
    if (corr_crit == "SSD") optimize_cost = &Optimizer::ssd;
    else if (corr_crit == "NSSD") optimize_cost = &Optimizer::nssd;
    else if (corr_crit == "ZNSSD") optimize_cost = &Optimizer::znssd;
    else {
        std::cerr << "Unexpected Correlation Criteria: '" << corr_crit << "'" << std::endl;
        std::cerr << "Allowed Values: 'SSD', 'NSSD', 'ZNSSD'." << std::endl;
        exit(EXIT_FAILURE);
    }
}



void Optimizer::debug_print(const int ss_x, const int ss_y, int iter, double costp, double ftol, double xtol) {
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
void Optimizer::copy_params_from_fft(const int idx,
                          const std::vector<double> &shift_x,
                          const std::vector<double> &shift_y) {
    std::fill(p.begin(), p.end(), 0.0);
    p[0] = shift_x[idx];
    p[1] = shift_y[idx];
}

void Optimizer::copy_params_from_neigh(const std::vector<double> &results_p,
                            const int idx_results_p) {

    for (int i = 0; i < p.size(); i++)
        p[i] = results_p[idx_results_p+i];

}

// Reset parameters to zero
void Optimizer::reset_params() {
    std::fill(p.begin(), p.end(), 0.0);
}
