// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <cmath>
#include <array>
#include <chrono>

// Program Header files
#include "./dicbruteforce.hpp"
#include "./defines.hpp"


namespace brute {



    // directions of spiral.
    std::vector<int> dirs = {1,  0,  
                             0,  1,
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
               const int p0,
               const int p1);

    void (*find_min)(const int ss_x, 
                     const int ss_y, 
                     const int *image_ref, 
                     const int px_vertical, 
                     const int px_horizontal, 
                     util::Subset *ss_def, 
                     util::Subset *ss_ref, 
                     brute::Parameters *brute);




    void init(std::string &corr_crit, std::string &search_method){

        // set brute force cost function
        if (corr_crit == "SSD") {
            cost_function = brute::ssd;
        } else if (corr_crit == "NSSD") {
            cost_function = brute::nssd;
        } else if (corr_crit == "ZNSSD") {
            cost_function = brute::znssd;
        } else {
            std::cerr << "Error: cost function not recognised. Using SSD." << std::endl;
            cost_function = brute::ssd;
        }

        // set brute force search method
        if (search_method == "EXHAUSTIVE") {
            find_min = exhaustive;
        } else if (search_method == "EXPANDING_WAVEFRONT") {
            find_min = expanding_wavefront;
        } else {
            std::cerr << "Error: search method not recognised. Using EXPANDING_WAVEFRONT." << std::endl;
            find_min = expanding_wavefront;
        }
    }


    void expanding_wavefront(const int ss_x, 
                         const int ss_y, 
                         const int *image_ref, 
                         const int px_vertical, 
                         const int px_horizontal, 
                         util::Subset *ss_def, 
                         util::Subset *ss_ref, 
                         brute::Parameters *brute) {

        

        const int range = brute->range;
        double cost_min = 1.0e6;

        int r_xmin = ss_x - range;
        int r_ymin = ss_y - range;
        int r_xmax = ss_x + range;
        int r_ymax = ss_y + range;


        int offset_x = brute->p_rigid[0];
        int offset_y = brute->p_rigid[1];
        int count = 0;

        for (int r = 0; r <= range; r++) {

            // Go around the current ring at radius r
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {
            
                    // Only process the points on the perimeter of the square
                    if (std::abs(dx) != r && std::abs(dy) != r) continue;

                    int p0 = dx + offset_x;
                    int p1 = dy + offset_y;

                    // get the min and max values of the 'new' subset
                    int ss_xmin = ss_x + p0;
                    int ss_ymin = ss_y + p1;
                    int ss_xmax = ss_x + p0 + ss_def->size;
                    int ss_ymax = ss_y + p1 + ss_def->size;

                    // if the 'new' subset is outside the image bounds then skip it.
                    if (ss_xmin < 0 || ss_xmax >= px_horizontal || ss_ymin < 0 || ss_ymax >= px_vertical) continue;

                    // if the 'new' subset is outside the range bounds then skip it.
                    if (p0 < r_xmin || p0 >= r_xmax || p1 < r_ymin || p1 >= r_ymax) continue;

                    double cost = cost_function(ss_x, ss_y, image_ref, px_vertical, px_horizontal, ss_def, ss_ref, p0, p1);
                    if (std::abs(cost) < cost_min) {
                        cost_min = cost;
                        brute->p_rigid[0] = p0;
                        brute->p_rigid[1] = p1;
                        if (cost_min < brute->threshold_bf) return;
                    }
                }
            }
        }
        // std::cout << ss_x << " " << ss_y << " " << brute->p_rigid[0] << " " << brute->p_rigid[1] << " " << cost_min << std::endl;
        exit(0);
    }


        void exhaustive(const int ss_x, 
                        const int ss_y, 
                        const int *image_ref, 
                        const int px_vertical, 
                        const int px_horizontal, 
                        util::Subset *ss_def, 
                        util::Subset *ss_ref, 
                        brute::Parameters *brute){

        const int range = brute->range;
        double cost_min = 1.0e6;
        
        // clamp search area to within image bounds
        const int xmin = std::max(0, ss_x - range);
        const int ymin = std::max(0, ss_y - range);
        const int xmax = std::min(px_horizontal, ss_x + range);
        const int ymax = std::min(px_vertical, ss_y + range);


        for (int p1 = -ymin; p1 <= ymax; p1++){
            for (int p0 = -xmin; p0 <= xmax; p0++){                

                double cost = cost_function(ss_x, ss_y, image_ref, px_vertical, px_horizontal, ss_def,ss_ref,p0,p1);

                // update minumum value. If Below tolerance then return.
                if (std::abs(cost) < cost_min) {
                    cost_min = cost;
                    brute->p_rigid[0] = p0;
                    brute->p_rigid[1] = p1;
                    if (cost_min < brute->threshold_bf) return;
                }

            }
        }
    }



    double ssd(const int ss_x, 
               const int ss_y, 
               const int *image_ref, 
               const int px_vertical, 
               const int px_horizontal, 
               util::Subset *ss_def, 
               util::Subset *ss_ref,
               const int p0,
               const int p1){
        
        const int num_px = ss_def->num_px;
        double cost = 0.0;

        for (int i = 0; i < num_px; i++){

             // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + p0;
            ss_ref->y[i] = ss_def->y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref->x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref->y[i]);
            const int idx = ss_ref_y_int * px_horizontal + ss_ref_x_int;

            ss_ref[i] = image_ref[idx];
            
            cost += (ss_def->vals[i] - ss_ref->vals[i]) *
                    (ss_def->vals[i] - ss_ref->vals[i]);      

        }

        return cost;

    }


    double nssd(const int ss_x, 
                const int ss_y, 
                const int *image_ref, 
                const int px_vertical, 
                const int px_horizontal, 
                util::Subset *ss_def,
                util::Subset *ss_ref,
                const int p0,
                const int p1){


        const int num_px = ss_def->num_px;
        double cost = 0.0;
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;

        // get subset values and cost function denominators
        for (int i = 0; i < num_px; i++){
        
            // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + p0;
            ss_ref->y[i] = ss_def->y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref->x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref->y[i]);
            const int idx = ss_ref_y_int * px_horizontal + ss_ref_x_int;

            ss_ref->vals[i] = image_ref[idx];

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
                const int p0,
                const int p1){

        const int num_px = ss_def->num_px;
        double cost = 0.0;
        double mean_ref = 0.0;
        double mean_def = 0.0;

        // loop over pixel values in reference image
        for (int i = 0; i < num_px; i++){

            // integer coordinates of the subset in the reference image and extract pixel value
            ss_ref->x[i] = ss_def->x[i] + p0;
            ss_ref->y[i] = ss_def->y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref->x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref->y[i]);
            const int idx = ss_ref_y_int * px_horizontal + ss_ref_x_int;

            ss_ref->vals[i] = image_ref[idx];
            mean_ref += ss_ref->vals[i];
            mean_def += ss_def->vals[i];
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