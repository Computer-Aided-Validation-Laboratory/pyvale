// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICOPTIMIZATION_H
#define DICOPTIMIZATION_H

// STD library Header files
#include <vector>
#include <iostream>


// GNU Scientific Library Header files
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>

// Program Header files

namespace optimization {

    struct Data {
        std::vector<double> subset_coords_x;
        std::vector<double> subset_coords_y;
        std::vector<double> subset_values;
        gsl_spline2d* spline;  
        gsl_interp_accel* xacc;
        gsl_interp_accel* yacc;
        int px_horizontal;
        int px_vertical;
        int p_length;

        // Member function to update data
        void update(const std::vector<double>& new_subset_coords_x,
                    const std::vector<double>& new_subset_coords_y,
                    const std::vector<double>& new_subset_values) {
            subset_coords_x = new_subset_coords_x;
            subset_coords_y = new_subset_coords_y;
            subset_values = new_subset_values;
        }
    };

    void init(int num_images, int num_subsets, std::string &corr_crit, std::string &interp_routine, std::string &shape_func, int subset_size, int px_horizontal, int px_vertical, gsl_spline2d *spline);
    int cost_function(const gsl_vector *p_gsl, void *data, gsl_vector * f);
    int jacobian_function(const gsl_vector *p_gsl, void *data, gsl_matrix *J);
    void set_data(std::vector<double> &subset_coords_x,std::vector<double> &subset_coords_y, std::vector<double> &subset);
    void execute(int subset_num, bool seed=true, double xtol=1e-15, double gtol=1e-20,double ftol=1e-15,int max_iter=100);
    void collect_results(int n_img, int n_ss, int subset_num, int subset_x, int subset_y);
    void print_results(int ss_x, int ss_y);

}

#endif //DICOPTIMIZATION_H