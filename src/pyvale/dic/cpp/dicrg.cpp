// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <csignal>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <omp.h>



// Program Header files
#include "./dicsubset.hpp"


namespace rg {

    bool is_valid_point(const int ss_x, const int ss_y, const subset::Grid &ss_grid) {

        int x = ss_x / ss_grid.step;
        int y = ss_y / ss_grid.step;

        int idx = y * ss_grid.num_ss_x + x;

        if ((ss_x % ss_grid.step) || (ss_y % ss_grid.step)){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "Subset ss_step size: " << ss_grid.step << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else if (ss_grid.mask[idx] == -1){
            std::cerr << "Subset coordinates (" << ss_x << ", " << ss_y << ") are not a valid subset location." << std::endl;
            std::cerr << "subset mask index: " << idx << std::endl;
            return false;
            exit(EXIT_FAILURE);
        }
        else return true;

        //auto it = ss_grid.coords_to_idx.find({ss_x, ss_y});

        //// check if coordinates are in the coordinate list
        //if (it == ss_grid.coords_to_idx.end()) {
        //   std::cerr << "Error: coordinates not found in the coordinate list." << std::endl;
        //   std::cerr << "Coordinates: " << ss_x << ", " << ss_y << std::endl;
        //   exit(EXIT_FAILURE);
        //}
        //else return true;
    }
}


