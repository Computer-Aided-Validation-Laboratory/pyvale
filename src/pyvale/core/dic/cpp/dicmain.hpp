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
    // extern std::vector<int> ss_coord_list;
    // extern std::vector<int> niter_arr;
    // extern std::vector<double> u_arr;
    // extern std::vector<double> v_arr;
    // extern std::vector<double> p_arr;
    // extern std::vector<double> ftol_arr;
    // extern std::vector<double> xtol_arr;

    /**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def_stack 
     * @param image_roi 
     * @param px_vertical 
     * @param px_horizontal 
     * @param num_def_images 
     * @param subset_step 
     * @param subset_size 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     * @param corr_crit 
     * @param shape_func 
     * @param interp_routine 
     * @param scan_method 
     */
    void engine_2d(int* image_ref, 
                    int* image_def_stack, 
                    bool* image_roi, 
                    int px_vertical, 
                    int px_horizontal, 
                    int num_def_images,
                    int subset_step,
                    int subset_size,
                    int max_iter,
                    double precision,
                    double threshold_lm,
                    double threshold_bf,
                    int range_bf,
                    std::string& corr_crit, 
                    std::string& shape_func,
                    std::string& interp_routine,
                    std::string& scan_method);

    /**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param ss_size 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
    void image_scan(int *image_ref, 
                    util::Image *image_def, 
                    bool *image_roi,
                    util::SubsetList *ss_list, 
                    int num_def_images, 
                    int img_num, 
                    int ss_size, 
                    int max_iter, 
                    double precision,
                    double threshold_lm,
                    double threshold_bf,
                    double range_bf,
                    int num_params);


    /**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param ss_size 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
    void image_scan_with_bf(int *image_ref, 
                    util::Image *image_def, 
                    bool *image_roi,
                    util::SubsetList *ss_list, 
                    int num_def_images, 
                    int img_num, 
                    int ss_size, 
                    int max_iter, 
                    double precision,
                    double threshold_lm,
                    double threshold_bf,
                    double range_bf,
                    int num_params);


    /**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param ss_coord_list 
     * @param num_def_images 
     * @param img_num 
     * @param ss_size 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
    void reliability_guided(int *image_ref, 
                    util::Image *image_def, 
                    bool *image_roi,
                    util::SubsetList *ss_list, 
                    int num_def_images, 
                    int img_num, 
                    int ss_size, 
                    int max_iter, 
                    double precision,
                    double threshold_lm,
                    double threshold_bf,
                    double range_bf,
                    int num_params);



    /**
     * @brief 
     * 
     * @param num_def_images 
     * @param img_num 
     * @param ss 
     * @param results 
     */
    void append_results(int num_def_images, int img_num, int ss, optimizer::Results *results);
}

#endif //DICENGINE_H
