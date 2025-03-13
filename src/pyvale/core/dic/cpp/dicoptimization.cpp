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


// // Program Header files
#include "./dicoptimization.hpp"
#include "./dicinterpolator.hpp"
#include "./diccorrelation.hpp"
#include "./dicjacobian.hpp"


namespace optimization {


    const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace *w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params = gsl_multifit_nlinear_default_parameters();
    gsl_vector *p_arr;
    Data optData;
    int info;
    int p_length;

    std::vector<double> p;
    std::vector<int> niter;
    std::vector<int> ss_coords;

    void init(int num_images, int num_subsets, std::string &corr_crit, std::string &interp_routine, std::string &shape_function, int subset_size, int px_horizontal, int px_vertical, gsl_spline2d *spline){

        // function pointer depending on user specified correlation criteria;
        int (*costfunc_ptr)(const gsl_vector *, void *, gsl_vector *);
        int (*jacbfunc_ptr)(const gsl_vector *, void *, gsl_matrix *);

        if (corr_crit == "SSD") {
            costfunc_ptr = correlation::ssd;
            jacbfunc_ptr = jacobian::ssd;
        } 
        else if (corr_crit == "NSSD") {
            costfunc_ptr = correlation::nssd;
            jacbfunc_ptr = jacobian::nssd;
        } 
        else if (corr_crit == "ZNSSD") {
            costfunc_ptr = correlation::znssd;
            jacbfunc_ptr = jacobian::znssd;
        }
        else {            
            std::cerr << "Unexpected Correlation Criteria: \'" << corr_crit << "\'" << std::endl;
            std::cerr << "Allowed Values: \'SSD\', \'NSSD\', \'ZNSSD\'. " << std::endl;
            exit(EXIT_FAILURE);
        }

        // resize number of paramters depending on the shape function used
        if (shape_function == "rigid") p_length = 2;
        else if (shape_function == "affine") p_length = 6;
        else {
            std::cerr << "Unexpected Shape Function Value: \'" << shape_function << "\'" << std::endl;
            std::cerr << "Allowed Values: \'rigid\', \'affine\'." << std::endl;
            exit(EXIT_FAILURE);
        }


        // populate p values with defaults
        p_arr = gsl_vector_alloc(p_length);
        for (int p = 0; p < p_length; p++){
            gsl_vector_set (p_arr, p, 0.0);
        }

        
        // returns a pointer to an accelerator object, which is a kind of iterator for interpolation lookups. 
        // It tracks the state of lookups, thus allowing for application of various acceleration strategies.
        optData.spline = spline;
        optData.xacc = gsl_interp_accel_alloc();
        optData.yacc = gsl_interp_accel_alloc();
        optData.subset_coords_x.resize(subset_size*subset_size, 0.0);
        optData.subset_coords_y.resize(subset_size*subset_size, 0.0);
        optData.subset_values.resize(subset_size*subset_size, 0.0);
        optData.px_horizontal = px_horizontal;
        optData.px_vertical = px_vertical;
        optData.p_length = p_length;


        // funcs/vars for optimization routine
        fdf.f = costfunc_ptr;
        fdf.df = jacbfunc_ptr; 
        fdf.fvv = NULL;
        fdf.n = subset_size * subset_size;
        fdf.p = p_length;
        fdf.params = &optData;

        // alloc mem for multifit
        w = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &fdf_params, subset_size*subset_size, p_length);


        // resize output arrays
        p.resize(num_images*num_subsets*6);
        niter.resize(num_images*num_subsets);
        ss_coords.resize(num_subsets);

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






    void execute(bool seed, double xtol, double gtol,double ftol,int max_iter){

        // if seed has been selected as true use the previous iterations to set the values of P.
        if (seed){
            for (int i = 0; i < p_length; i++){
                gsl_vector_set (p_arr, i,gsl_vector_get(w->x, i));
            }
        }
        gsl_multifit_nlinear_init(p_arr, &fdf, w);
        gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol, NULL, NULL, &info, w);
    }


    void collect_results(int n_img, int n_ss, int subset_num, int subset_x, int subset_y){

        p[n_img*n_ss + subset_num*6 + 0] = gsl_vector_get(w->x, 0);
        p[n_img*n_ss + subset_num*6 + 1] = gsl_vector_get(w->x, 1);
        p[n_img*n_ss + subset_num*6 + 2] = gsl_vector_get(w->x, 2);
        p[n_img*n_ss + subset_num*6 + 3] = gsl_vector_get(w->x, 3);
        p[n_img*n_ss + subset_num*6 + 4] = gsl_vector_get(w->x, 4);
        p[n_img*n_ss + subset_num*6 + 5] = gsl_vector_get(w->x, 5);
        niter[n_img*n_ss] = gsl_multifit_nlinear_niter(w);
        ss_coords[n_ss*2 + 0] = subset_x;
        ss_coords[n_ss*2 + 1] = subset_y;
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
