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
#include "./dicoptimizer.hpp"


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
    }


    void check_convergence_or_exit(const OptResult &res) {
        if (!res.converged || !res.above_threshold) {
            std::cout << "ERROR: unsuccessful convergence at seed or direct neighbour." << std::endl;
            std::cout << "Please select a different seed location." << std::endl;
            std::cout << std::endl;
            std::cout << "displacement: " << res.u << " " << res.v << std::endl;
            std::cout << "cost: " << res.cost << std::endl;
            std::cout << "xtol: " << res.xtol << std::endl;
            std::cout << "ftol: " << res.ftol << std::endl;
            std::cout << "above_threshold: " << static_cast<unsigned>(res.above_threshold) << std::endl;
            std::cout << "converged: " << static_cast<unsigned>(res.converged) << std::endl;
            std::cout << "iterations: " << res.iter << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}


