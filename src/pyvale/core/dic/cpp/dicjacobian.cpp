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

// Program Header files
#include "./dicoptimization.hpp"
#include "./dicinterpolator.hpp"
#include "./dicjacobian.hpp"

namespace jacobian {


    int ssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J) {

        optimization::Data* jacdata = static_cast<optimization::Data*>(data);

        std::vector<double>& subset_coords_x = jacdata->subset_coords_x;
        std::vector<double>& subset_coords_y = jacdata->subset_coords_y;
        gsl_spline2d* spline = jacdata->spline;
        gsl_interp_accel* xacc = jacdata->xacc;
        gsl_interp_accel* yacc = jacdata->yacc;
        int px_vertical = jacdata->px_vertical;
        int px_horizontal = jacdata->px_horizontal;
        int p_length = jacdata->p_length;

        double p[p_length];
        for (int j = 0; j < p_length; j++) {
            p[j] = gsl_vector_get(p_gsl, j);
        }

        const size_t n = subset_coords_y.size();

        for (size_t i = 0; i < n; ++i) {
            
            double x = subset_coords_x[i];
            double y = subset_coords_y[i];
            
            // Affine
            double x_new = p[0] + (1 + p[2]) * x + p[3] * y;
            double y_new = p[1] + (1 + p[5]) * y + p[4] * x;

            if (x_new < 0 || x_new > px_horizontal || y_new < 0 || y_new > px_vertical) {
                for (int j = 0; j < p_length; ++j) {
                    gsl_matrix_set(J, i, j, 1.0e6);
                }
            }
            else{                      
                
                // partial derivatives derivatives 
                double df_dx = gsl_spline2d_eval_deriv_x(spline, x_new, y_new, xacc, yacc);
                double df_dy = gsl_spline2d_eval_deriv_y(spline, x_new, y_new, xacc, yacc);

                // Compute partial derivatives of r_i with respect to parameters
                gsl_matrix_set(J, i, 0, -df_dx);
                gsl_matrix_set(J, i, 1, -df_dy);
                gsl_matrix_set(J, i, 2, -df_dx * x_new);
                gsl_matrix_set(J, i, 3, -df_dx * y_new);
                gsl_matrix_set(J, i, 4, -df_dy * x_new);
                gsl_matrix_set(J, i, 5, -df_dy * y_new);
            }
        }

        return GSL_SUCCESS;

    }

    int nssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J) {

        optimization::Data* jacdata = static_cast<optimization::Data*>(data);


        double p[jacdata->p_length];
        for (int j = 0; j < jacdata->p_length; j++) {
            p[j] = gsl_vector_get(p_gsl, j);
        }

        const size_t n = jacdata->subset_coords_y.size();

        // squared sum of pixel values for the deformed subset
        double sum_squared = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_squared += jacdata->subset_values[i] * jacdata->subset_values[i];
        }

        // 1 over the gray level sum. Prevents multiple divisions when calculation of correlation criteria.
        double inv_sum_squared = 1.0 / sum_squared;

        for (size_t i = 0; i < n; ++i) {
            
            double x = jacdata->subset_coords_x[i];
            double y = jacdata->subset_coords_y[i];
            
            // Affine
            double x_new = p[0] + (1 + p[2]) * x + p[3] * y;
            double y_new = p[1] + (1 + p[5]) * y + p[4] * x;

            if (x_new < 0 || x_new > jacdata->px_horizontal || y_new < 0 || y_new > jacdata->px_vertical) {
                for (int j = 0; j < jacdata->p_length; ++j) {
                    gsl_matrix_set(J, i, j, 1.0e6);
                }
            }
            else{                      
                
                // partial derivatives derivatives 
                double df_dx = gsl_spline2d_eval_deriv_x(jacdata->spline, x_new, y_new, jacdata->xacc, jacdata->yacc);
                double df_dy = gsl_spline2d_eval_deriv_y(jacdata->spline, x_new, y_new, jacdata->xacc, jacdata->yacc);

                // Compute partial derivatives of r_i with respect to parameters
                gsl_matrix_set(J, i, 0, -df_dx * inv_sum_squared);
                gsl_matrix_set(J, i, 1, -df_dy * inv_sum_squared);
                gsl_matrix_set(J, i, 2, -df_dx * x_new * inv_sum_squared);
                gsl_matrix_set(J, i, 3, -df_dx * y_new * inv_sum_squared);
                gsl_matrix_set(J, i, 4, -df_dy * x_new * inv_sum_squared);
                gsl_matrix_set(J, i, 5, -df_dy * y_new * inv_sum_squared);
            }
        }

        return GSL_SUCCESS;
    }


    int znssd(const gsl_vector *p_gsl, void *data, gsl_matrix * f) {

        std::cout << "ZNSSD jacobian doesn't work. Issues with the maths. Exiting" << std::endl;
        exit(0);

        // optimization::Data* jacdata = static_cast<optimization::Data*>(data);

        // // get shape function parameter values
        // double p[jacdata->p_length];
        // for (int i = 0; i < jacdata->p_length; i++){
        //     p[i] = gsl_vector_get(p_gsl,i);
        // }
        
        // const size_t n = jacdata->subset_values.size();


        // // mean values of reference and deformed subset
        // double mean_def = 0.0;
        // double mean_ref = 0.0;

        // // store the interpolated values in a std::vector<double> (need to access them multiple times)
        // std::vector<double> transformed_vals(n, 0.0);

        // //get the mean values
        // for (size_t i = 0; i < n; ++i) {

        //     double x = jacdata->subset_coords_x[i];
        //     double y = jacdata->subset_coords_y[i];

        //     double x_new = p[0] + (1 + p[2]) * x + p[3] * y;
        //     double y_new = p[1] + (1 + p[5]) * y + p[4] * x;

        //     transformed_vals[i] = gsl_spline2d_eval(jacdata->spline, x_new, y_new, jacdata->xacc, jacdata->yacc);

        //     mean_ref += transformed_vals[i];
        //     mean_def += jacdata->subset_values[i];
        // }

        // // normalise the mean values
        // mean_def /= n;
        // mean_ref /= n;

        // // (f(x,y) - f_mean)**2
        // double sum_squared_ref = 0.0;
        // double sum_squared_def = 0.0;
        // for (size_t i = 0; i < n; ++i) {
        //     sum_squared_def += (jacdata->subset_values[i] -  mean_def) * (jacdata->subset_values[i] -  mean_def);
        //     sum_squared_ref += (transformed_vals[i] - mean_ref) * (transformed_vals[i] - mean_ref);
        // }

        // // 1.0 / (f(x,y) - f_mean)**2
        // double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
        // double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


        // // loop over pixels in the subset
        // for (size_t i = 0; i < n; ++i) {

        //     double x = jacdata->subset_coords_x[i];
        //     double y = jacdata->subset_coords_y[i];
            
        //     // Affine
        //     double x_new = p[0] + (1 + p[2]) * x + p[3] * y; // x-coordinate
        //     double y_new = p[1] + (1 + p[5]) * y + p[4] * x; // y-coordinate

        //     // partial derivatives derivatives 
        //     double df_dx = gsl_spline2d_eval_deriv_x(jacdata->spline, x_new, y_new, jacdata->xacc, jacdata->yacc);
        //     double df_dy = gsl_spline2d_eval_deriv_y(jacdata->spline, x_new, y_new, jacdata->xacc, jacdata->yacc);

        //     double norm_def = inv_sum_squared_def * (jacdata->subset_values[i] - mean_def);
        //     double norm_ref = inv_sum_squared_ref * (transformed_vals[i] - mean_ref);


        //     // prevent out of bounds - set cost function to massive value if out of bounds
        //     // this will be replaced with an ROI at some point
        //     if (x_new < 0 || x_new > jacdata->px_horizontal || y_new < 0 || y_new >  jacdata->px_vertical) {
        //         gsl_vector_set(f, i, 1.0e6);
        //     }
        //     else {
        //         double diff =  inv_sum_squared_def * (jacdata->subset_values[i] - mean_def) -
        //                     inv_sum_squared_ref * (transformed_vals[i] - mean_ref);

        //         gsl_vector_set(f, i, (diff * diff));
        //     }

        // }

        // return GSL_SUCCESS;

    }



}