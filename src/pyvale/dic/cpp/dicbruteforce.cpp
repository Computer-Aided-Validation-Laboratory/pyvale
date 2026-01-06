// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <vector>
#include <cmath>
#include <array>

// opencv header files
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"

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
    double (*cost_function)(const double *img_ref, 
               const int px_hori, 
               const int px_vert, 
               util::Subset &ss_def, 
               util::Subset &ss_ref,
               const int p0,
               const int p1);

    void (*find_min)(const int ss_x,
                     const int ss_y,
                     const double *img_ref, 
                     const int px_hori, 
                     const int px_vert, 
                     util::Subset &ss_def, 
                     util::Subset &ss_ref, 
                     brute::Parameters &brute);



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
            //std::cerr << "Error: search method not recognised. Using EXPANDING_WAVEFRONT." << std::endl;
            find_min = expanding_wavefront;
        }
    }


    void expanding_wavefront(const int ss_x,
                         const int ss_y,
                         const double *img_ref,
                         const int px_hori,
                         const int px_vert,
                         util::Subset &ss_def,
                         util::Subset &ss_ref,
                         brute::Parameters &brute) {



        const int range = brute.range;
        double cost_min = 1.0e6;

        int offset_x = 0; //brute.p_rigid_prevmatch[0];
        int offset_y = 0; //brute.p_rigid_prevmatch[1];

        for (int r = 0; r <= range; r++) {

            // Go around the current ring at radius r
            for (int dy = -r; dy <= r; dy++) {
                for (int dx = -r; dx <= r; dx++) {


                    if (!is_perimeter_point(dx, dy, r)) 
                        continue;

                    int p0 = dx + offset_x;
                    int p1 = dy + offset_y;

                    if (!is_within_range(p0, p1, range))
                        continue;

                    int ss_xmin = ss_x + p0;
                    int ss_ymin = ss_y + p1;
                    int ss_xmax = ss_x + p0 + ss_def.size;
                    int ss_ymax = ss_y + p1 + ss_def.size;

                    if (!is_within_image(ss_xmin, ss_ymin, ss_xmax, ss_ymax, 
                                         px_hori, px_vert))
                        continue;

                    double cost = cost_function(img_ref, px_hori, 
                                                px_vert, ss_def, 
                                                ss_ref, p0, p1);

                    if (std::abs(cost) < cost_min) {
                        cost_min = cost;
                        brute.p_rigid[0] = p0;
                        brute.p_rigid[1] = p1;

                        // if its below our threshold and considered a good match. we'll use these values for the next brute force.
                        if (cost_min < brute.bf_threshold) {
                            brute.p_rigid_prevmatch[0] = brute.p_rigid[0];
                            brute.p_rigid_prevmatch[1] = brute.p_rigid[1];
                            return;
                        }
                    }
                }
            }
        }
    }


    void exhaustive(const int ss_x, 
                    const int ss_y, 
                    const double *img_ref, 
                    const int px_hori, 
                    const int px_vert, 
                    util::Subset &ss_def, 
                    util::Subset &ss_ref, 
                    brute::Parameters &brute){

        const int range = brute.range;
        double cost_min = 1.0e6;
        
        // clamp search area to within image bounds
        const int xmin = std::max(0, ss_x - range);
        const int ymin = std::max(0, ss_y - range);
        const int xmax = std::min(px_hori, ss_x + range);
        const int ymax = std::min(px_vert, ss_y + range);


        for (int p1 = -ymin; p1 <= ymax; p1++){
            for (int p0 = -xmin; p0 <= xmax; p0++){

                double cost = cost_function(img_ref, px_hori, px_vert, ss_def,ss_ref,p0,p1);

                // update minumum value. If Below tolerance then return.
                if (std::abs(cost) < cost_min) {
                    cost_min = cost;
                    brute.p_rigid[0] = p0;
                    brute.p_rigid[1] = p1;
                    if (cost_min < brute.bf_threshold) return;
                }

            }
        }
    }



    double ssd(const double *img_ref, 
               const int px_hori, 
               const int px_vert, 
               util::Subset &ss_def, 
               util::Subset &ss_ref,
               const int p0,
               const int p1){
        
        const int num_px = ss_def.num_px;
        double cost = 0.0;

        for (int i = 0; i < num_px; i++){

            ss_ref.x[i] = ss_def.x[i] + p0;
            ss_ref.y[i] = ss_def.y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref.x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref.y[i]);
            const int idx = ss_ref_y_int * px_hori + ss_ref_x_int;

            ss_ref.vals[i] = img_ref[idx];
            
            cost += (ss_def.vals[i] - ss_ref.vals[i]) *
                    (ss_def.vals[i] - ss_ref.vals[i]);

        }

        return cost;

    }


    double nssd(const double *img_ref, 
                const int px_hori, 
                const int px_vert, 
                util::Subset &ss_def,
                util::Subset &ss_ref,
                const int p0,
                const int p1){


        const int num_px = ss_def.num_px;
        double cost = 0.0;
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;

        // get subset values and cost function denominators
        for (int i = 0; i < num_px; i++){
        
            ss_ref.x[i] = ss_def.x[i] + p0;
            ss_ref.y[i] = ss_def.y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref.x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref.y[i]);
            const int idx = ss_ref_y_int * px_hori + ss_ref_x_int;

            ss_ref.vals[i] = img_ref[idx];

            sum_squared_ref += ss_ref.vals[i] * ss_ref.vals[i];
            sum_squared_def += ss_def.vals[i] * ss_def.vals[i];

        }

        double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
        double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);


        // calculate cost
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            cost += (def_norm - ref_norm) *
                    (def_norm - ref_norm);
        }

        return cost;

    }

    double znssd(const double *img_ref, 
                const int px_hori, 
                const int px_vert, 
                util::Subset &ss_def,
                util::Subset &ss_ref,
                const int p0,
                const int p1){

        const int num_px = ss_def.num_px;
        double cost = 0.0;
        double mean_ref = 0.0;
        double mean_def = 0.0;

        // loop over pixel values in reference image
        for (int i = 0; i < num_px; i++){

            ss_ref.x[i] = ss_def.x[i] + p0;
            ss_ref.y[i] = ss_def.y[i] + p1;

            const int ss_ref_x_int = static_cast<int>(ss_ref.x[i]);
            const int ss_ref_y_int = static_cast<int>(ss_ref.y[i]);
            const int idx = ss_ref_y_int * px_hori + ss_ref_x_int;

            ss_ref.vals[i] = img_ref[idx];
            mean_ref += ss_ref.vals[i];
            mean_def += ss_def.vals[i];
        }

        mean_ref /= num_px;
        mean_def /= num_px;

        // get cost function denominators
        double sum_squared_ref = 0.0;
        double sum_squared_def = 0.0;
        for (int i = 0; i < num_px; ++i) {
            sum_squared_ref += (ss_ref.vals[i] - mean_ref)*
                               (ss_ref.vals[i] - mean_ref);
            sum_squared_def += (ss_def.vals[i] - mean_def)*
                               (ss_def.vals[i] - mean_def);
        }
        double inv_sum_squared_ref = 1.0 / std::sqrt(sum_squared_ref);
        double inv_sum_squared_def = 1.0 / std::sqrt(sum_squared_def);



        // calcualte cost 
        for (int i = 0; i < num_px; i++){
            double def_norm = ss_def.vals[i] * inv_sum_squared_def;
            double ref_norm = ss_ref.vals[i] * inv_sum_squared_ref;
            cost += (def_norm - ref_norm) * (def_norm - ref_norm);
        }

        return cost;
    }



    inline bool is_perimeter_point(int dx, int dy, int r) {
        return std::abs(dx) == r || std::abs(dy) == r;
    }

    inline bool is_within_image(int xmin, int ymin, int xmax, int ymax,
                         int width, int height) {
        return xmin >= 0 && xmax < width && ymin >= 0 && ymax < height;
    }

    inline bool is_within_range(int p0, int p1, int range) {
        return p0 >= -range && p0 < range && p1 >= -range && p1 < range;
    }



    // void cross_correlation(const int ss_x, 
    //                     const int ss_y, 
    //                     const double *img_ref, 
    //                     const int px_vert, 
    //                     const int px_hori, 
    //                     util::Subset *ss_def, 
    //                     util::Subset *ss_ref, 
    //                     brute::Parameters &brute) {

    //     cv::Mat image(px_vert, px_hori, CV_32S, const_cast<int*>(img_ref));
    //     cv::Mat ss(ss_def.size, ss_def.size, CV_64F, ss_def.vals.data());

    //     cv::Mat image_float;
    //     cv::Mat ss_float;
    //     image.convertTo(image_float, CV_32F);
    //     ss.convertTo(ss_float, CV_32F);


    //     cv::Mat result;
    //     cv::matchTemplate(image_float, ss_float, result, cv::TM_CCOEFF_NORMED);
        
    //     double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
    //     cv::Point matchLoc;
        
    //     cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    //     std::cout << "minVal: " << minVal << std::endl;
    //     std::cout << "maxVal: " << maxVal << std::endl;
    //     std::cout << "minLoc: " << minLoc.x << ", " << minLoc.y << std::endl;
    //     std::cout << "maxLoc: " << maxLoc.x << ", " << maxLoc.y << std::endl;
    //     brute.p_rigid[0] = maxLoc.x - ss_x;
    //     brute.p_rigid[1] = maxLoc.y - ss_y;
    // }

    
    // end of namespace
}
