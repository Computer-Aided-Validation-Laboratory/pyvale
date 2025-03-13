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

        // squared sum of pixel values for the deformed subset
        double sum_squared_ref = 0.0;
        std::vector<double> transformed_vals(n, 0.0);
        std::vector<double> x_new(n, 0.0);
        std::vector<double> y_new(n, 0.0);
        std::vector<double> df_dx(n, 0.0);
        std::vector<double> df_dy(n, 0.0);


        for (size_t i = 0; i < n; ++i) {

            double x = subset_coords_x[i];
            double y = subset_coords_y[i];

            x_new[i] = p[0] + (1 + p[2]) * x + p[3] * y;
            y_new[i] = p[1] + (1 + p[5]) * y + p[4] * x;

            // prevent out of bounds - set cost function to massive value if out of bounds
            // this will be replaced with an ROI at some point
            if (x_new[i] < 0 || x_new[i] > px_horizontal || y_new[i] < 0 || y_new[i] > px_vertical) {
                for (int j = 0; j < p_length; ++j) {
                    gsl_matrix_set(J, i, j, 1.0e6);
                }
            }
            else {

                transformed_vals[i] = gsl_spline2d_eval(spline, x_new[i], y_new[i], xacc, yacc);
                sum_squared_ref += transformed_vals[i] * transformed_vals[i];

                df_dx[i] = gsl_spline2d_eval_deriv_x(spline, x_new[i], y_new[i], xacc, yacc);
                df_dy[i] = gsl_spline2d_eval_deriv_y(spline, x_new[i], y_new[i], xacc, yacc);

            }


        }

        // 1 over the gray level sum. Prevents multiple divisions when calculation of correlation criteria.
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (size_t i = 0; i < n; ++i) {
    
            // Compute partial derivatives of r_i with respect to parameters
            gsl_matrix_set(J, i, 0, -df_dx[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 1, -df_dy[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 2, -df_dx[i] * x_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 3, -df_dx[i] * y_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 4, -df_dy[i] * x_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 5, -df_dy[i] * y_new[i] * inv_sum_squared_ref);
        }

        return GSL_SUCCESS;
    }


    int znssd(const gsl_vector *p_gsl, void *data, gsl_matrix *J) {

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

        // squared sum of pixel values for the deformed subset
        double mean_ref = 0.0;
        std::vector<double> transformed_vals(n, 0.0);
        std::vector<double> x_new(n, 0.0);
        std::vector<double> y_new(n, 0.0);
        std::vector<double> df_dx(n, 0.0);
        std::vector<double> df_dy(n, 0.0);

        for (size_t i = 0; i < n; ++i) {

            double x = subset_coords_x[i];
            double y = subset_coords_y[i];

            x_new[i] = p[0] + (1 + p[2]) * x + p[3] * y;
            y_new[i] = p[1] + (1 + p[5]) * y + p[4] * x;

            // prevent out of bounds - set cost function to massive value if out of bounds
            // this will be replaced with an ROI at some point
            if (x_new[i] < 0 || x_new[i] > px_horizontal || y_new[i] < 0 || y_new[i] > px_vertical) {
                for (int j = 0; j < p_length; ++j) {
                    gsl_matrix_set(J, i, j, 1.0e6);
                }
                return GSL_SUCCESS;
            }
            else {

                transformed_vals[i] = gsl_spline2d_eval(spline, x_new[i], y_new[i], xacc, yacc);
                mean_ref += transformed_vals[i];

                df_dx[i] = gsl_spline2d_eval_deriv_x(spline, x_new[i], y_new[i], xacc, yacc);
                df_dy[i] = gsl_spline2d_eval_deriv_y(spline, x_new[i], y_new[i], xacc, yacc);

            }
        }

        mean_ref /= n;

        double sum_squared_ref = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum_squared_ref += (transformed_vals[i] - mean_ref) * (transformed_vals[i] - mean_ref);
        }

        // 1 over the gray level sum. Prevents multiple divisions when calculation of correlation criteria.
        double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

        for (size_t i = 0; i < n; ++i) {
    
            // Compute partial derivatives of r_i with respect to parameters
            gsl_matrix_set(J, i, 0, -df_dx[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 1, -df_dy[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 2, -df_dx[i] * x_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 3, -df_dx[i] * y_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 4, -df_dy[i] * x_new[i] * inv_sum_squared_ref);
            gsl_matrix_set(J, i, 5, -df_dy[i] * y_new[i] * inv_sum_squared_ref);
        }

        return GSL_SUCCESS;

    }



}