// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <omp.h>

// common_cpp Header files
#include "../../common_cpp/util.hpp"

// DIC Header files
#include "./dicutil.hpp"
#include "./dicsubset.hpp"


namespace util {


    std::vector<int> niter_arr;
    std::vector<double> u_arr;
    std::vector<double> v_arr;
    std::vector<double> p_arr;
    std::vector<double> ftol_arr;
    std::vector<double> xtol_arr;
    std::vector<double> cost_arr;
    std::vector<uint8_t> conv_arr;
    bool at_end;



    void extract_image(double *img_def_stack, 
                       int image_number,
                       int px_hori,
                       int px_vert){

        int count = 0;
        for (int px_y = 0; px_y < px_vert; px_y++){
            for (int px_x = 0; px_x < px_hori; px_x++){
                int idx = image_number * px_hori * px_vert + px_y * px_hori + px_x;
                std::cout << img_def_stack[idx] << " ";
                //img_def->vals[count] = img_def_stack[idx];
                count++;
            }
            std::cout << std::endl;
        }
        exit(0);
    }

    int next_pow2(int n) {
        if (n <= 0){
            std::cerr << __FILE__ << " " << __LINE__ << std::endl;
            std::cerr << "Expected a positive integer to calculate next power of 2 " << std::endl;
            std::cerr << "n = " << n << std::endl;
            exit(EXIT_FAILURE);
        }

        // If already a power of 2, return as-is
        if ((n & (n - 1)) == 0) return n;

        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n++;

        // Handle possible overflow
        if (n < 0) return std::numeric_limits<int>::max();

        return n;
    }


    void gen_size_and_step_vector(std::vector<int> &ss_sizes, std::vector<int> &ss_steps, 
                                  const int ss_size, const int ss_step, const int max_disp) {

        ss_sizes.clear();
        ss_steps.clear();

        int power = next_pow2(max_disp);

        // Generate sizes down to just above ss_size
        while (power > ss_size) {
            ss_sizes.push_back(power);
            ss_steps.push_back(power / 2);
            power /= 2;
        }

        // Finally, add the original ss_size and ss_step
        ss_sizes.push_back(ss_size);
        ss_steps.push_back(ss_step);

        // debugging
        //for (size_t i = 0; i < ss_sizes.size(); ++i) {
        //    std::cout << "ss_size = " << ss_sizes[i] << ", step = " << ss_steps[i] << std::endl;
        //}
    }
}
