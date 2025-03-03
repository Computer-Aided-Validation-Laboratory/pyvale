// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_blas.h>


// Program Header files
#include "./dicoptimization.hpp"
#include "./dicinterpolator.hpp"



namespace optimization {


    // levenberg
    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    gsl_vector *p_arr;
    Data optData;
    int info;
    int p_length;


    void init(std::string interp_routine, std::string shape_function, int subset_size, gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc){

        // resize number of paramters depending on the shape function used
        if (shape_function == "rigid") p_length = 2;
        else if (shape_function == "affine") p_length = 6;
        else {
            std::cerr << "Unexpected Shape Function Value: \'" << shape_function << "\'" << std::endl;
            std::cerr << "Allowed Values: \'rigid\', \'affine\'" << std::endl;
            exit(0);
        }


        // populate p values with defaults
        p_arr = gsl_vector_alloc(6);
        for (int p = 0; p < p_length; p++){
            gsl_vector_set (p_arr, p, 0.0);
        }


        fdf.f = cost_function;
        fdf.df = jacobian_function; 
        fdf.fvv = NULL;
        fdf.n = subset_size * subset_size;
        fdf.p = p_length;
        fdf.params = &optData;

        w = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &fdf_params, subset_size*subset_size, p_length);

        // Struct that holds additional data for optimization routine.
        Data optdata;
        optdata.spline = spline;
        optdata.xacc = xacc;
        optdata.yacc = yacc;


    }

    int cost_function(const gsl_vector *p_gsl, void *data, gsl_vector * f) {

        Data* optdata = static_cast<Data*>(data);

        std::vector<double>& subset_coords_x = optdata->subset_coords_x;
        std::vector<double>& subset_coords_y = optdata->subset_coords_y;
        std::vector<double>& subset_values = optdata->subset_values;
        gsl_spline2d* spline = optdata->spline;   // Access spline
        gsl_interp_accel* xacc = optdata->xacc;    // Access x acceleration
        gsl_interp_accel* yacc = optdata->yacc;    // Access y acceleration
        
        double diff = 0.0;

        double p[p_length];
        for (int i = 0; i < p_length; i++){
            p[i] = gsl_vector_get(p_gsl,i);
        }

        for (unsigned int i = 0; i < subset_values.size(); ++i) {
            
            // Affine
            double x_new = p[0] + (1 + p[2]) * subset_coords_x[i] + p[3] * subset_coords_y[i]; // x-coordinate
            double y_new = p[1] + (1 + p[5]) * subset_coords_y[i] + p[4] * subset_coords_x[i]; // y-coordinate

            // prevent out of bounds
            if (x_new < 0 || x_new > 399 || y_new < 0 || y_new > 399) {
                gsl_vector_set(f, i, 1.0e6);
            }

            // use interpolator to get new pixel value
            diff =  subset_values[i] - gsl_spline2d_eval(spline, x_new, y_new, xacc, yacc);
            gsl_vector_set(f, i, diff);


        }

        return GSL_SUCCESS;

    }

    int jacobian_function(const gsl_vector *p_gsl, void *data, gsl_matrix *J) {

        Data* optdata = static_cast<Data*>(data);

        std::vector<double>& subset_coords_x = optdata->subset_coords_x;
        std::vector<double>& subset_coords_y = optdata->subset_coords_y;
        gsl_spline2d* spline = optdata->spline;
        gsl_interp_accel* xacc = optdata->xacc;
        gsl_interp_accel* yacc = optdata->yacc;

        double p[p_length];
        for (int j = 0; j < p_length; j++) {
            p[j] = gsl_vector_get(p_gsl, j);
        }

        for (size_t i = 0; i < subset_coords_x.size(); ++i) {
            
            double x_new = p[0] + (1 + p[2]) * subset_coords_x[i] + p[3] * subset_coords_y[i]; // x-coordinate
            double y_new = p[1] + (1 + p[5]) * subset_coords_y[i] + p[4] * subset_coords_x[i]; // y-coordinate

            // Compute spline derivatives 
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

        return GSL_SUCCESS;
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



    void execute(bool seed=true, 
                 double xtol=1e-15, 
                 double gtol=1e-20,
                 double ftol=1e-15,
                 int max_iter=100
                 ){

        // if seed has been selected as true use the previous iterations to set the values of P.
        if (seed == true){
            for (int i = 0; i < 6; i++){
                gsl_vector_set (p_arr, i,gsl_vector_get(w->x, i));
            }
        }


        gsl_multifit_nlinear_init(p_arr, &fdf, w);
        int status = gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, callback, NULL, &info, w);
    

    
    }


}
