// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICENGINE_H
#define DICENGINE_H


// STD library Header files
#include <vector>
#include <iostream>

// Program Header files
#include "./dicoptimizer.hpp"


namespace dic {

    // result arrays. Not using std::vector because harder to handle with cython
    extern std::vector<int> ss_coord_list;
    extern std::vector<int> niter_arr;
    extern std::vector<double> u_arr;
    extern std::vector<double> v_arr;
    extern std::vector<double> p_arr;
    extern std::vector<double> ftol_arr;
    extern std::vector<double> xtol_arr;

    void engine_2d(int* image_ref, 
                    int* image_def_stack, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    int max_iter,
                    double tol,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method);

    
    void         image_scan(int *image_ref, util::Image *image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int ss_size, int max_iter, double tol);
    void image_scan_with_bf(int *image_ref, util::Image *image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int ss_size, int max_iter, double tol);;
    void reliability_guided(int *image_ref, util::Image *image_def, std::vector<int> &ss_coord_list, int num_def_images, int img_num, int ss_size, int max_iter, double tol);;

    void append_results(int num_def_images, int img_num, int ss, optimizer::Results *results);
}

#endif //DICENGINE_H
