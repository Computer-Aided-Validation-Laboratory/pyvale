// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICRG_H
#define DICRG_H

// STD library Header files

// Program Header files
#include "./dicsubset.hpp"

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
     * @param x 
     * @param y 
     * @param px_hori 
     * @param px_vert 
     * @param ss_size 
     * @return true 
     * @return false 
     */
     bool is_valid_point(const int ss_x, const int ss_y, const subset::Grid &ss_grid);

}


#endif // DICRG_H
