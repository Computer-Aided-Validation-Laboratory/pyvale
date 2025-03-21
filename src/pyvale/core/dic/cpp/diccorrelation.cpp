// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <chrono>
#include <iostream>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>

// Program Header files
#include "./dicoptimization.hpp"
#include "./dicinterpolator.hpp"
#include "./diccorrelation.hpp"

namespace correlation {


    int ssd(const gsl_vector *p_gsl, void *data, gsl_vector * f) {

        optimization::Data* costdata = static_cast<optimization::Data*>(data);
        std::vector<double>& subset_coords_x = costdata->subset_coords_x;
        std::vector<double>& subset_coords_y = costdata->subset_coords_y;
        std::vector<double>& subset_values = costdata->subset_values;
        gsl_spline2d* spline = costdata->spline;   // Access spline
        gsl_interp_accel* xacc = costdata->xacc;    // Access x acceleration
        gsl_interp_accel* yacc = costdata->yacc;    // Access y acceleration
        int px_vertical = costdata->px_vertical;
        int px_horizontal = costdata->px_horizontal;
        int p_length = costdata->p_length;

        double p[p_length];
        for (int i = 0; i < p_length; i++){
            p[i] = gsl_vector_get(p_gsl,i);
        }
        
        const size_t n = subset_values.size();


        auto s0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {

            double x = subset_coords_x[i];
            double y = subset_coords_y[i];

            // Affine
            double x_new = p[0] + (1 + p[2]) * x + p[3] * y; // x-coordinate
            double y_new = p[1] + (1 + p[5]) * y + p[4] * x; // y-coordinate

            // prevent out of bounds - set cost function to massive value if out of bounds
            // this will be replaced with an ROI at some point
            if (x_new < 0 || x_new > px_horizontal || y_new < 0 || y_new > px_vertical) {
                gsl_vector_set(f, i, 1.0e6);
                std::cout << "OOOOOB" << std::endl;
            }
            else {
                double diff =  subset_values[i] - gsl_spline2d_eval(spline, x_new, y_new, xacc, yacc);
                // std::cout << diff << " ";
                gsl_vector_set(f, i, diff);
            }

        }
        std::cout << std::endl;
        auto f0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> e0 = f0 - s0;
        std::cout << p[0] << " " << p[1] <<  " " << p[2] << " "; 
        std::cout << p[3] << " " << p[4] <<  " " << p[5] << " "; 
        std::cout << "ssd:     " << e0.count() <<  " [s]     " << std::endl;



        return GSL_SUCCESS;

    }

    int nssd(const gsl_vector *p_gsl, void *data, gsl_vector * f) {

        optimization::Data* costdata = static_cast<optimization::Data*>(data);
        std::vector<double>& subset_coords_x = costdata->subset_coords_x;
        std::vector<double>& subset_coords_y = costdata->subset_coords_y;
        std::vector<double>& subset_values = costdata->subset_values;
        gsl_spline2d* spline = costdata->spline;   // Access spline
        gsl_interp_accel* xacc = costdata->xacc;    // Access x acceleration
        gsl_interp_accel* yacc = costdata->yacc;    // Access y acceleration
        int px_vertical = costdata->px_vertical;
        int px_horizontal = costdata->px_horizontal;
        int p_length = costdata->p_length;

        // get shape function parameter values
        double p[p_length];
        for (int i = 0; i < p_length; i++){
            p[i] = gsl_vector_get(p_gsl,i);
        }
        
        const size_t n = subset_values.size();


        // squared sum of pixel values for the deformed subset
        double sum_squared_def = 0.0;
        double sum_squared_ref = 0.0;
        // sum of squared values in deformed subset
        for (size_t i = 0; i < n; ++i) {
        }

        std::vector<double> transformed_vals(n, 0.0);
        bool out_of_bounds = false;
        for (size_t i = 0; i < n; ++i) {

            double x = subset_coords_x[i];
            double y = subset_coords_y[i];

            double x_new = p[0] + (1 + p[2]) * x + p[3] * y;
            double y_new = p[1] + (1 + p[5]) * y + p[4] * x;

            // prevent out of bounds - set cost function to massive value if out of bounds
            // this will be replaced with an ROI at some point
            if (x_new < 0 || x_new > px_horizontal || y_new < 0 || y_new > px_vertical) {
                out_of_bounds = true;
                break;
            }
            else {
                transformed_vals[i] = gsl_spline2d_eval(spline, x_new, y_new, xacc, yacc);
            }

            sum_squared_def += subset_values[i] * subset_values[i];
            sum_squared_ref += transformed_vals[i] * transformed_vals[i];

        }

        if (out_of_bounds) {
            for (size_t i = 0; i < n; ++i) {
                gsl_vector_set(f, i, 1.0e6);
            }
        }
        else {
        
            // 1 over the gray level sum. Prevents multiple divisions when calculation of correlation criteria.
            double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
            double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);


            // loop over pixels in the subset
            for (size_t i = 0; i < n; ++i) {
                double diff =  (subset_values[i] * inv_sum_squared_def) - (transformed_vals[i] * inv_sum_squared_ref);
                gsl_vector_set(f, i, diff);

            }
        }

        return GSL_SUCCESS;

    }


    int znssd(const gsl_vector *p_gsl, void *data, gsl_vector * f) {

        optimization::Data* costdata = static_cast<optimization::Data*>(data);
        std::vector<double>& subset_coords_x = costdata->subset_coords_x;
        std::vector<double>& subset_coords_y = costdata->subset_coords_y;
        std::vector<double>& subset_values = costdata->subset_values;
        gsl_spline2d* spline = costdata->spline;   // Access spline
        gsl_interp_accel* xacc = costdata->xacc;    // Access x acceleration
        gsl_interp_accel* yacc = costdata->yacc;    // Access y acceleration
        int px_vertical = costdata->px_vertical;
        int px_horizontal = costdata->px_horizontal;
        int p_length = costdata->p_length;

        // get shape function parameter values
        double p[p_length];
        for (int i = 0; i < p_length; i++){
            p[i] = gsl_vector_get(p_gsl,i);
        }
        
        const size_t n = subset_values.size();

        // mean values of reference and deformed subset
        double mean_def = 0.0;
        double mean_ref = 0.0;

        // store the interpolated values in a std::vector<double> (need to access them multiple times)
        std::vector<double> transformed_vals(n, 0.0);
        bool out_of_bounds = false;

        //get the mean values
        for (size_t i = 0; i < n; ++i) {

            double x = subset_coords_x[i];
            double y = subset_coords_y[i];

            double x_new = p[0] + (1 + p[2]) * x + p[3] * y;
            double y_new = p[1] + (1 + p[5]) * y + p[4] * x;

            // prevent out of bounds - set cost function to massive value if out of bounds
            // this will be replaced with an ROI at some point
            if (x_new < 0 || x_new > px_horizontal || y_new < 0 || y_new > px_vertical) {
                out_of_bounds = true;
                break;
            }
            else {
                transformed_vals[i] = gsl_spline2d_eval(spline, x_new, y_new, xacc, yacc);
            }

            mean_ref += transformed_vals[i];
            mean_def += subset_values[i];

        }
        
        
        if (out_of_bounds) {
            for (size_t i = 0; i < n; ++i) {
                gsl_vector_set(f, i, 1.0e6);
            }
        }
        else {
        
            // normalise the mean values
            mean_def /= n;
            mean_ref /= n;

            // (f(x,y) - f_mean)**2
            double sum_squared_ref = 0.0;
            double sum_squared_def = 0.0;
            for (size_t i = 0; i < n; ++i) {
                sum_squared_def += (subset_values[i] -  mean_def) * (subset_values[i] -  mean_def);
                sum_squared_ref += (transformed_vals[i] - mean_ref) * (transformed_vals[i] - mean_ref);
            }

            // 1.0 / (f(x,y) - f_mean)**2
            double inv_sum_squared_def = 1.0 / sqrt(sum_squared_def);
            double inv_sum_squared_ref = 1.0 / sqrt(sum_squared_ref);

            // loop over pixels in the subset and calculate residual 
            for (size_t i = 0; i < n; ++i) {
                double diff =  inv_sum_squared_def * (subset_values[i] - mean_def) - inv_sum_squared_ref * (transformed_vals[i] - mean_ref);
                gsl_vector_set(f, i, diff);
            }
        }

        return GSL_SUCCESS;

    }

}