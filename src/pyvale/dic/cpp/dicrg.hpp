// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICRG_H
#define DICRG_H

// STD library Header files



// Program Header files
#include "./dicutil.hpp"
#include "./defines.hpp"


namespace rg {

    /**
     * @brief 
     * 
     */
    struct Point {
        int x;
        int y;
        double val;
        
        // Constructor
        Point(int x_coord, int y_coord, double _val) : 
            x(x_coord), y(y_coord), val(_val) {}
        
        // Comparison operator for priority queue (higher ZNCC first)
        bool operator<(const Point& other) const {
            return val < other.val;  // Note: priority_queue puts largest elements on top
        }
    };






    /**
     * @brief 
     * 
     * @param image_ref 
     * @param image_def 
     * @param seed_x
     * @param seed_y 
     * @param num_def_images 
     * @param img_num 
     * @param ss_size 
     * @param max_iter 
     * @param precision 
     * @param threshold_lm 
     * @param threshold_bf 
     * @param range_bf 
     */
    void reliability_guided_dic_single_seed(
        const double *image_ref,
        const double *image_def,
        const bool *image_roi,
        const int seed_x, const int seed_y,  // Single seed point coordinates
        util::SubsetData *ssdata,
        const int num_def_images,
        const int img_num,
        const int max_iter,
        const double precision,
        const double threshold_lm,
        const double threshold_bf,
        const double range_bf,
        const int num_params);


    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @param px_horizontal 
     * @param px_vertical 
     * @param ss_size 
     * @return true 
     * @return false 
     */
     bool is_valid_point(int ss_x, int ss_y, util::SubsetData &ssdata);




}


#endif // DICRG_H
