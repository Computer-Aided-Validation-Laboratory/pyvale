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



namespace optimization {


    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    gsl_vector *p_arr;
    Data optData;
    int info;
    int p_length;


    void init(std::string &interp_routine, std::string &shape_function, int subset_size, int px_horizontal, int px_vertical, gsl_spline2d *spline){




        // resize number of paramters depending on the shape function used
        if (shape_function == "rigid") p_length = 2;
        else if (shape_function == "affine") p_length = 6;
        else {
            std::cerr << "Unexpected Shape Function Value: \'" << shape_function << "\'" << std::endl;
            std::cerr << "Allowed Values: \'rigid\', \'affine\'" << std::endl;
            exit(EXIT_FAILURE);
        }


        // populate p values with defaults
        p_arr = gsl_vector_alloc(p_length);
        for (int p = 0; p < p_length; p++){
            gsl_vector_set (p_arr, p, 0.0);
        }

        // assign struct variables
        optData.spline = spline;
        
        // returns a pointer to an accelerator object, which is a kind of iterator for interpolation lookups. 
        // It tracks the state of lookups, thus allowing for application of various acceleration strategies.
        optData.xacc = gsl_interp_accel_alloc();
        optData.yacc = gsl_interp_accel_alloc();
        optData.subset_coords_x.resize(subset_size*subset_size, 0.0);
        optData.subset_coords_y.resize(subset_size*subset_size, 0.0);
        optData.subset_values.resize(subset_size*subset_size, 0.0);
        optData.px_horizontal = px_horizontal;
        optData.px_vertical = px_horizontal;


        // funcs/vars for optimization routine
        fdf.f = cost_function;
        fdf.df = jacobian_function; 
        fdf.fvv = NULL;
        fdf.n = subset_size * subset_size;
        fdf.p = p_length;
        fdf.params = &optData;

        // alloc mem for multifit
        w = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &fdf_params, subset_size*subset_size, p_length);

    }

    int cost_function(const gsl_vector *p_gsl, void *data, gsl_vector * f) {

        Data* costdata = static_cast<Data*>(data);
        std::vector<double>& subset_coords_x = costdata->subset_coords_x;
        std::vector<double>& subset_coords_y = costdata->subset_coords_y;
        std::vector<double>& subset_values = costdata->subset_values;
        gsl_spline2d* spline = costdata->spline;   // Access spline
        gsl_interp_accel* xacc = costdata->xacc;    // Access x acceleration
        gsl_interp_accel* yacc = costdata->yacc;    // Access y acceleration
        int px_vertical = costdata->px_vertical;
        int px_horizontal = costdata->px_horizontal;

        double p[p_length];
        for (int i = 0; i < p_length; i++){
            p[i] = gsl_vector_get(p_gsl,i);
        }
        
        const size_t n = subset_values.size();

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
            }
            else {
                double diff =  subset_values[i] - gsl_spline2d_eval(spline, x_new, y_new, xacc, yacc);
                gsl_vector_set(f, i, diff);
            }

        }

        return GSL_SUCCESS;

    }

    int jacobian_function(const gsl_vector *p_gsl, void *data, gsl_matrix *J) {

        Data* jacdata = static_cast<Data*>(data);

        std::vector<double>& subset_coords_x = jacdata->subset_coords_x;
        std::vector<double>& subset_coords_y = jacdata->subset_coords_y;
        gsl_spline2d* spline = jacdata->spline;
        gsl_interp_accel* xacc = jacdata->xacc;
        gsl_interp_accel* yacc = jacdata->yacc;
        int px_vertical = jacdata->px_vertical;
        int px_horizontal = jacdata->px_horizontal;

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


    void set_data(std::vector<double> &subset_coords_x, std::vector<double> &subset_coords_y, std::vector<double> &subset_values){
        optData.update(subset_coords_x, subset_coords_y, subset_values);
    }



    void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w){

        gsl_vector *f = gsl_multifit_nlinear_residual(w);
        gsl_vector *x = gsl_multifit_nlinear_position(w);
        double rcond;

        gsl_multifit_nlinear_rcond(&rcond, w);
        
        std::cout << "iter = " << iter << " ";
        std::cout << "p0 = " << gsl_vector_get(x, 0) << " ";
        std::cout << "p1 = " << gsl_vector_get(x, 1) << " ";
        std::cout << "p2 = " << gsl_vector_get(x, 2) << " ";          
        std::cout << "p3 = " << gsl_vector_get(x, 3) << " ";
        std::cout << "p4 = " << gsl_vector_get(x, 4) << " ";
        std::cout << "p5 = " << gsl_vector_get(x, 5) << " ";
        std::cout << "cond(J) = " << 1.0 / rcond << " ";
        std::cout << "|f(x)| = " << gsl_blas_dnrm2(f) << "\n";
    }



    void execute(bool seed, 
                 double xtol, 
                 double gtol,
                 double ftol,
                 int max_iter){

        // if seed has been selected as true use the previous iterations to set the values of P.
        if (seed){
            for (int i = 0; i < p_length; i++){
                gsl_vector_set (p_arr, i,gsl_vector_get(w->x, i));
            }
        }
        gsl_multifit_nlinear_init(p_arr, &fdf, w);
        gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, NULL, NULL, &info, w);
    }

    void print_results(int ss_x, int ss_y){
        std::cout << "results: " << " ";
        std::cout << ss_x << " " << ss_y << " ";
        std::cout << gsl_multifit_nlinear_niter(w) << " ";
        std::cout << gsl_vector_get(w->x, 0) << " ";
        std::cout << gsl_vector_get(w->x, 1) << " ";
        std::cout << gsl_vector_get(w->x, 2) << " ";
        std::cout << gsl_vector_get(w->x, 3) << " ";
        std::cout << gsl_vector_get(w->x, 4) << " ";
        std::cout << gsl_vector_get(w->x, 5) << "\n";
        // std::cout << gsl_multifit_nlinear_name(w) << " ";
        // std::cout << gsl_multifit_nlinear_trs_name(w) << std::endl;
    }


}
