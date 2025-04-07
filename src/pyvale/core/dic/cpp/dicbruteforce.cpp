// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <cmath>
#include <array>

// Program Header files
#include "./dicbruteforce.hpp"


namespace bruteforce {




    std::vector<int> dirs = {1,  0, //left 
                             0,  1, //right
                            -1,  0, 
                             0, -1}; 


    // function pointers
    double (*cost_function)(const int ss_x, 
               const int ss_y, 
               const int *image_ref, 
               const int px_vertical, 
               const int px_horizontal, 
               util::Subset *ss_def, 
               util::Subset *ss_ref,
               brute::Parameters *brute);

    std::array<int, 2> (*find_min)(const int ss_x, 
                                  const int ss_y, 
                                  const int *image_ref, 
                                  const int px_vertical, 
                                  const int px_horizontal, 
                                  util::Subset *ss_def, 
                                  util::Subset *ss_ref, 
                                  brute::Parameters *brute);

    void init(std::string &cost_function, std::string &search_method){

        // set brute force cost function
        if (cost_function == "SSD") {
            cost_function = ssd;
        } else if (cost_function == "NSSD") {
            cost_function = nssd;
        } else if (cost_function == "ZNSSD") {
            cost_function = znssd;
        } else {
            std::cerr << "Error: cost function not recognised. Using SSD." << std::endl;
            cost_function = ssd;
        }

        // set brute force search method
        if (search_method == "EXHAUSTIVE") {
            find_min = exhaustive;
        } else if (search_method == "SPIRAL") {
            find_min = spiral_search;
        } else {
            std::cerr << "Error: search method not recognised. Using SPIRAL." << std::endl;
            find_min = spiral_search;
        }
    }


    std::array<int, 2> spiral_search(const int ss_x, 
                                  const int ss_y, 
                                  const int *image_ref, 
                                  const int px_vertical, 
                                  const int px_horizontal, 
                                  util::Subset *ss_def, 
                                  util::Subset *ss_ref, 
                                  brute::Parameters *brute){

        const int range = brute->range;

        const int ymin = ss_y - range;
        const int ymax = ss_y + range;
        const int xmin = ss_x - range;
        const int xmax = ss_x + range;

        const int px_in_search_area = (range*2+1) * (range*2+1); // number of pixels in the search area
        
        // seed the start of the brute force search with the rigid parameters from the last subset
        int u = ss_x + brute->p_rigid[0];
        int v = ss_y + brute->p_rigid[1];

        int steps = 1;
        int step_in_dir = 0;
        int turns = 0;
        int dir_idx = 0;

        std::array<int, 2> p;

        for (int i = 0; i < px_in_search_area; i++){

            // reset cost function for new translation
            double cost_min = 1.0e6;
            p = {0,0};

            double cost = cost_function(ss_x, ss_y, image_ref, px_vertical, px_horizontal, ss_def, ss_ref, brute);

            // update minumum value. If Below tolerance then return.
            if (std::abs(cost) < cost_min) {
                cost_min = cost;
                p = {u,v};
                if (cost_min < brute->tol) return p;
            }

            // if we've not found a suitable match, update u,v from spiral path.
            u += dirs[2*dir_idx];
            v += dirs[2*dir_idx+1];
            ++step_in_dir;

            if (step_in_dir == steps) {
                // Change direction
                dir_idx = (dir_idx + 1) % 4;
                step_in_dir = 0;
                ++turns;

                // Every two turns, increase step size
                if (turns % 2 == 0) ++steps;
            }
        }


        // if we dont go below tolerance return min value of P at end of search
        return p; 
    }


    std::array<int, 2> exhaustive(const int ss_x, 
                                  const int ss_y, 
                                  const int *image_ref, 
                                  const int px_vertical, 
                                  const int px_horizontal, 
                                  util::Subset *ss_def, 
                                  util::Subset *ss_ref, 
                                  brute::Parameters *brute){

        const int range = brute->range;
        std::array<int, 2> p;


        for (int v = -range; v <= range; v++){
            for (int u = -range; u <= range; u++){

                // reset cost function for new translation
                double cost_min = 1.0e6;
                p = {0,0};

                double cost = cost_function(ss_x, ss_y, image_ref, px_vertical, px_horizontal, ss_def,ss_ref,brute);


                // update minumum value. If Below tolerance then return.
                if (std::abs(cost) < cost_min) {
                    cost_min = cost;
                    p = {u,v};
                    if (cost_min < brute->tol) return p;
                }

            }
        }

        // if we dont go below tolerance return min value of P at end of search
        return p; 

    }



    double ssd(const int ss_x, 
               const int ss_y, 
               const int *image_ref, 
               const int px_vertical, 
               const int px_horizontal, 
               util::Subset *ss_def, 
               util::Subset *ss_ref,
               brute::Parameters *brute){
        
        const int num_px = ss_def->num_px;
        double cost = 0.0;
        for (int i = 0; i < num_px; i++){

             // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + brute->p_rigid[0];
            ss_ref->y[i] = ss_def->y[i] + brute->p_rigid[1];
            ss_ref[i] = image_ref[ss_ref->y[i] * px_horizontal + ss_ref->x[i]];
            
            cost += (ss_def->vals[i] - ss_ref->vals[i]) *
                    (ss_def->vals[i] - ss_ref->vals[i]);      

        }
    }


    double nssd(const int ss_x, 
                const int ss_y, 
                const int *image_ref, 
                const int px_vertical, 
                const int px_horizontal, 
                util::Subset *ss_def,
                util::Subset *ss_ref,
                brute::Parameters *brute){


        const int num_px = ss_def->num_px;
        double cost = 0.0;
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;

        // get subset values and cost function denominators
        for (int i = 0; i < num_px; i++){
        
            // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + brute->p_rigid[0];
            ss_ref->y[i] = ss_def->y[i] + brute->p_rigid[1];
            ss_ref->vals[i] = image_ref[ss_ref->y[i] * px_horizontal + ss_ref->x[i]];

            sum_squared_ref += ss_ref->vals[i] * ss_ref->vals[i];
            sum_squared_def += ss_def->vals[i] * ss_def->vals[i];

        }
        double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
        double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);


        // calculate cost
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def->vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref->vals[i] * inv_sum_squared_ref;
            cost += (def_norm - ref_norm) *
                    (def_norm - ref_norm);        
        }

        return cost;

        
    }

    double znssd(const int ss_x, 
                 const int ss_y, 
                 const int *image_ref, 
                 const int px_vertical, 
                 const int px_horizontal, 
                 util::Subset *ss_def, 
                 util::Subset *ss_ref,
                 brute::Parameters *brute){

        const int num_px = ss_def->num_px;
        double cost = 0.0;
        double mean_ref = 0.0;
        double mean_def = 0.0;

        // loop over pixel values in reference image
        for (int i = 0; i < num_px; i++){

            // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + brute->p_rigid[0];
            ss_ref->y[i] = ss_def->y[i] + brute->p_rigid[1];
            ss_ref->vals[i] = image_ref[ss_ref->y[i] * px_horizontal + ss_ref->x[i]];
            mean_ref += ss_ref->vals[i];
            mean_def += ss_def[i];
        }
        mean_ref /= num_px;
        mean_def /= num_px;

        // get cost function denominators
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_ref += (ss_ref->vals[i] - mean_ref) * (ss_ref->vals[i] - mean_ref);
            sum_squared_def += (ss_def->vals[i] - mean_def) * (ss_def->vals[i] - mean_def);
        }
        double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
        double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);



        // calcualte cost 
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def->vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref->vals[i] * inv_sum_squared_ref;
            cost += (def_norm - ref_norm) * (def_norm - ref_norm);        
        }

        return cost;
    }


    // end of namespace
}