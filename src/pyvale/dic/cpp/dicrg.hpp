// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICRG_H
#define DICRG_H

// STD library Header files
#include <memory>


// Program Header files
#include "./dicutil.hpp"
#include "./defines.hpp"
#include "./dicfourier.hpp"

namespace rg {

    /**
     * @brief 
     * 
     */
    struct Point {
        int idx;
        double val;

        // Constructor
        Point(int _idx, double _val) : 
            idx(_idx), val(_val) {}

        // Comparison operator for priority queue (higher ZNCC first)
        bool operator<(const Point& other) const {
            return val < other.val;  // Note: priority_queue puts largest elements on top
        }
    };




    /**
     * @brief
     * 
     * @param[out] shift_x 
     * @param[out] shift_y 
     * @param ss_x 
     * @param ss_y 
     * @param fft_windows 
     * @param interp_ref 
     * @param img_def 
     */
    void get_rigid_shift(double &shift_x, double &shift_y,
                         const int ss_x, const int ss_y,
                         std::vector<std::unique_ptr<fourier::FFT>>& fft_windows,
                         const Interpolator &interp_ref,
                         const double *img_def);

    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @param px_hori 
     * @param px_vert 
     * @param ss_size 
     * @return true 
     * @return false 
     */
     bool is_valid_point(const int ss_x, const int ss_y, const util::SubsetData &ssdata);




}


#endif // DICRG_H
